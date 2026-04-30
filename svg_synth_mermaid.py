#!/usr/bin/env python3
"""
Synthesize a compact Mermaid diagram from a large Graphviz-style clustered SVG using an LLM,
processing the cluster tree **bottom-up by depth** (batch all nodes at the same depth, then
depth-1, ...). Requires the same env as utils.call_llm (LLM_PROVIDER + credentials).

Pipeline:
  1) Parse SVG → nested cluster tree (same logic as svg_to_mermaid.py).
  2) Flatten to nodes with stable ids (n00000, ...), depth, parent, children.
  3) For depth = max_depth .. 0: batch nodes at that depth (size --chunk-size), LLM adds
     a short `synthesis` per node from label + children summaries.
  4) Reduce pass A (merge names): LLM outputs ≤50 canonical groups; each original `nid` maps to one group.
  5) Reduce pass B (merge edges): derive group-level edges from parent/child tree, LLM simplifies and emits Mermaid (≤50 nodes).

Hard cap: **at most 50** Mermaid nodes (groups). `--max-mermaid-nodes` is clamped to 50.

Usage:
  python svg_synth_mermaid.py graph.svg -o synth.mmd
  python svg_synth_mermaid.py graph.svg -o synth.mmd --chunk-size 10 --max-parse-depth 30
  python svg_synth_mermaid.py graph.svg -o synth.mmd --no-cache --max-mermaid-nodes 50
"""
from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import dotenv
import yaml

from svg_to_mermaid import ClusterNode, _collect_clusters_from_svg
from utils.call_llm import call_llm

dotenv.load_dotenv()


def _local(tag: str) -> str:
    if tag.startswith("{"):
        return tag.split("}", 1)[-1]
    return tag


def extract_yaml_from_response(text: str) -> str:
    raw = text.strip()
    m = re.search(r"```ya?ml\s*\n(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"```\s*\n(.*?)```", raw, re.DOTALL)
    if m2:
        cand = m2.group(1).strip()
        try:
            yaml.safe_load(cand)
            return cand
        except yaml.YAMLError:
            pass
    try:
        yaml.safe_load(raw)
        return raw
    except yaml.YAMLError:
        pass
    raise ValueError("Could not extract YAML from model response")


def extract_mermaid_from_response(text: str) -> str:
    raw = text.strip()
    m = re.search(r"```mermaid\s*\n(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    if not m:
        raise ValueError("No ```mermaid ... ``` block in model response")
    return m.group(1).strip() + "\n"


@dataclass
class FlatNode:
    nid: str
    depth: int
    label: str
    parent_nid: Optional[str]
    child_nids: List[str] = field(default_factory=list)
    leaf_labels: List[str] = field(default_factory=list)
    synthesis: str = ""


def forest_to_flat(forest: List[ClusterNode]) -> List[FlatNode]:
    flat: List[FlatNode] = []

    def walk(node: ClusterNode, depth: int, parent_nid: Optional[str]) -> str:
        nid = f"n{len(flat):05d}"
        fn = FlatNode(
            nid=nid,
            depth=depth,
            label=node.label,
            parent_nid=parent_nid,
            child_nids=[],
            leaf_labels=[t[1] for t in node.leaf_nodes],
        )
        flat.append(fn)
        for ch in node.children_clusters:
            cid = walk(ch, depth + 1, nid)
            fn.child_nids.append(cid)
        return nid

    for root in forest:
        walk(root, 0, None)
    return flat


def _chunk_list(items: List[FlatNode], size: int) -> List[List[FlatNode]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _map_prompt_batch(batch: List[FlatNode], by_id: Dict[str, FlatNode]) -> str:
    payload: List[Dict[str, Any]] = []
    for n in batch:
        ch_sum = []
        for cid in n.child_nids:
            c = by_id.get(cid)
            if c:
                ch_sum.append(c.synthesis or c.label)
        leaf = (n.leaf_labels or [])[:8]
        payload.append(
            {
                "nid": n.nid,
                "label": n.label,
                "parent_nid": n.parent_nid,
                "children_summaries": ch_sum,
                "sample_leaf_labels": leaf,
            }
        )
    return yaml.safe_dump(
        {"clusters": payload},
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    )


def run_depth_batches(
    flat: List[FlatNode],
    chunk_size: int,
    use_cache: bool,
) -> None:
    by_id: Dict[str, FlatNode] = {n.nid: n for n in flat}
    by_depth: Dict[int, List[FlatNode]] = {}
    for n in flat:
        by_depth.setdefault(n.depth, []).append(n)
    max_d = max(by_depth.keys()) if by_depth else -1

    for d in range(max_d, -1, -1):
        level_nodes = sorted(by_depth.get(d, []), key=lambda x: x.nid)
        if not level_nodes:
            continue
        print(f"[depth {d}] {len(level_nodes)} cluster(s), batch size {chunk_size}")
        for bi, batch in enumerate(_chunk_list(level_nodes, chunk_size)):
            yin = _map_prompt_batch(batch, by_id)
            prompt = f"""You compress architecture cluster metadata extracted from a dependency/ call-graph SVG.

For EACH cluster in `clusters`, write a technical `synthesis`: 1–2 sentences naming its role
(responsibility, boundary) in the system. Use only information implied by `label`,
`children_summaries`, and `sample_leaf_labels`. Do not invent new subsystems or vendors.

Return ONLY valid YAML:

```yaml
items:
  - nid: n00000
    synthesis: |
      One or two sentences.
  - nid: n00001
    synthesis: |
      ...
```

Input YAML:

{yin}
"""
            raw = call_llm(prompt, use_cache=use_cache)
            ystr = extract_yaml_from_response(raw)
            data = yaml.safe_load(ystr)
            if not isinstance(data, dict) or "items" not in data:
                raise ValueError(f"Map phase: expected dict with 'items', got: {type(data)}")
            for it in data["items"]:
                if not isinstance(it, dict) or "nid" not in it:
                    continue
                nid = str(it["nid"]).strip()
                syn = str(it.get("synthesis", "")).strip()
                node = by_id.get(nid)
                if node and syn:
                    node.synthesis = syn
            # fallback: empty synthesis → label
            for n in batch:
                if not (n.synthesis or "").strip():
                    n.synthesis = n.label[:500]
            print(f"  batch {bi + 1}: ok ({len(batch)} nodes)")


def _build_outline(flat: List[FlatNode], max_chars: int) -> str:
    lines: List[str] = []
    for n in sorted(flat, key=lambda x: (x.depth, x.nid)):
        parent = n.parent_nid or "-"
        syn = (n.synthesis or n.label).replace("\n", " ")
        if len(syn) > 280:
            syn = syn[:277] + "..."
        lines.append(f"depth={n.depth}\t{n.nid}\tparent={parent}\t{syn}")
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 80] + "\n... [truncated; increase --max-outline-chars or reduce clusters]\n"


MAX_MERMAID_NODES_CAP = 50


def reduce_merge_names(
    flat: List[FlatNode],
    max_groups: int,
    max_outline_chars: int,
    use_cache: bool,
) -> Dict[str, Any]:
    """Reduce pass 1: merge original cluster ids into ≤ max_groups canonical groups."""
    outline = _build_outline(flat, max_outline_chars)
    all_nids = [n.nid for n in sorted(flat, key=lambda x: x.nid)]
    prompt = f"""You merge many SVG cluster summaries into a **small set of canonical architecture groups**.

Rules:
- Output **at most {max_groups}** groups (hard limit).
- Every cluster id listed under `all_nids` must appear in **exactly one** group under `members`.
- Each group has a short stable `id` (G0, G1, … or similar) and a human `name` (≤ 55 chars, technical).
- Merge synonyms and tiny leaves into a suitable parent theme; do not invent product names not hinted in the outline.

`all_nids` (complete list; every id must be assigned):
{yaml.safe_dump({"all_nids": all_nids}, allow_unicode=True, default_flow_style=False)}

Outline (depth, nid, parent, synthesis):
{outline}

Return ONLY valid YAML:

```yaml
groups:
  - id: G0
    name: "Short layer or subsystem name"
    members:
      - n00000
      - n00001
```
"""
    raw = call_llm(prompt, use_cache=use_cache)
    ystr = extract_yaml_from_response(raw)
    data = yaml.safe_load(ystr)
    if not isinstance(data, dict) or "groups" not in data:
        raise ValueError(f"Reduce names: expected dict with 'groups', got {type(data)}")
    groups = data["groups"]
    if not isinstance(groups, list) or len(groups) < 1 or len(groups) > max_groups:
        raise ValueError(
            f"Reduce names: need 1..{max_groups} groups, got {len(groups) if isinstance(groups, list) else 'invalid'}"
        )
    return data


def _nid_to_group_map(reduce_yaml: Dict[str, Any], flat: List[FlatNode]) -> Dict[str, str]:
    """nid -> canonical group id."""
    groups = reduce_yaml.get("groups") or []
    m: Dict[str, str] = {}
    for g in groups:
        if not isinstance(g, dict):
            continue
        gid = str(g.get("id", "")).strip()
        if not gid:
            continue
        for mem in g.get("members") or []:
            nid = str(mem).strip()
            if nid and nid not in m:
                m[nid] = gid
    expected = {n.nid for n in flat}
    missing = expected - set(m.keys())
    if missing:
        # Deterministic fallback: singleton groups for misses (may exceed cap — warn)
        for i, nid in enumerate(sorted(missing)):
            fake_gid = f"_X{i}"
            m[nid] = fake_gid
        print(f"[warn] reduce names: {len(missing)} nids missing from LLM output; assigned fallback groups")
    return m


def _group_display_names(reduce_yaml: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for g in reduce_yaml.get("groups") or []:
        if not isinstance(g, dict):
            continue
        gid = str(g.get("id", "")).strip()
        if gid:
            out[gid] = str(g.get("name", gid)).strip() or gid
    return out


def _structural_group_edges(flat: List[FlatNode], nid_to_gid: Dict[str, str]) -> List[tuple[str, str]]:
    """Parent/child cluster containment → directed group edge (parent_group, child_group), deduped."""
    seen: set[tuple[str, str]] = set()
    out: List[tuple[str, str]] = []
    for n in flat:
        p = n.parent_nid
        if not p:
            continue
        gp = nid_to_gid.get(p)
        gc = nid_to_gid.get(n.nid)
        if not gp or not gc or gp == gc:
            continue
        key = (gp, gc)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def reduce_merge_edges(
    reduce_names_yaml: Dict[str, Any],
    flat: List[FlatNode],
    nid_to_gid: Dict[str, str],
    max_nodes: int,
    use_cache: bool,
) -> str:
    """Reduce pass 2: simplify group-level edges and emit Mermaid (≤ max_nodes nodes)."""
    gnames = _group_display_names(reduce_names_yaml)
    # include fallback-only ids from nid_to_gid
    for gid in set(nid_to_gid.values()):
        gnames.setdefault(gid, gid)

    edges = _structural_group_edges(flat, nid_to_gid)
    edge_payload = [{"from_group": a, "to_group": b, "kind": "contains"} for a, b in edges]
    max_edges_in_prompt = 2500
    if len(edge_payload) > max_edges_in_prompt:
        print(f"[warn] truncating candidate edges for reduce-2: {len(edge_payload)} -> {max_edges_in_prompt}")
        edge_payload = edge_payload[:max_edges_in_prompt]

    unique_gids = sorted(set(nid_to_gid.values()))
    if len(unique_gids) > max_nodes:
        print(
            f"[warn] {len(unique_gids)} distinct groups after mapping (limit {max_nodes}); "
            f"reduce-1 should merge harder; reduce-2 still requests ≤{max_nodes} Mermaid nodes."
        )

    groups_payload = [{"id": gid, "name": gnames.get(gid, gid)} for gid in unique_gids]

    prompt = f"""You are given canonical architecture groups (already merged from a large SVG cluster tree) and **candidate directed edges**
derived from parent/child containment in that tree (kind \"contains\": parent cluster visually wraps the child in the SVG).

Tasks:
1) **Merge redundant / parallel edges** between the same pair (keep at most one edge per ordered pair; optional short label).
2) Drop ultra-noisy edges if they obscure the big picture; keep the backbone (major layers and dominant containment flow).
3) Produce ONE **Mermaid flowchart TB** using **at most {max_nodes} nodes** (each node is one group). Node id syntax: letters like A, B, C1 with bracket labels.
4) Use edges that read well top-to-bottom (outer/larger scope toward top as you see fit).
5) Do not add new named subsystems beyond the given groups.

Groups:
{yaml.safe_dump({"groups": groups_payload}, allow_unicode=True, default_flow_style=False, sort_keys=False)}

Candidate edges (merge/simplify from this list):
{yaml.safe_dump({"edges": edge_payload}, allow_unicode=True, default_flow_style=False, sort_keys=False)}

Output ONLY:

```mermaid
flowchart TB
  ...
```
"""
    raw = call_llm(prompt, use_cache=use_cache)
    return extract_mermaid_from_response(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM synthesis: large cluster SVG → compact Mermaid (depth batches).")
    parser.add_argument("svg", help="Input .svg (Graphviz-style clusters preferred)")
    parser.add_argument("-o", "--output", default="synth_architecture.mmd", help="Output .mmd path")
    parser.add_argument(
        "--max-parse-depth",
        type=int,
        default=999,
        help="Max nesting depth when parsing SVG clusters (default 999)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=12,
        help="Max clusters per LLM call within each depth level (default 12)",
    )
    parser.add_argument(
        "--max-mermaid-nodes",
        type=int,
        default=50,
        help="Max nodes in final Mermaid diagram (capped at 50, default 50)",
    )
    parser.add_argument(
        "--max-outline-chars",
        type=int,
        default=48000,
        help="Cap final outline size sent to the model (default 48000)",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable LLM cache for this run")
    parser.add_argument(
        "--dump-flat",
        type=str,
        default=None,
        metavar="JSON",
        help="Write flattened cluster JSON after synthesis (optional)",
    )
    parser.add_argument(
        "--dump-reduce1",
        type=str,
        default=None,
        metavar="YAML",
        help="Write reduce pass 1 (merged group names) YAML for debugging",
    )
    args = parser.parse_args()

    max_mermaid_nodes = min(MAX_MERMAID_NODES_CAP, max(1, args.max_mermaid_nodes))
    if args.max_mermaid_nodes > MAX_MERMAID_NODES_CAP:
        print(f"[info] --max-mermaid-nodes clamped to {MAX_MERMAID_NODES_CAP}")

    tree = ET.parse(args.svg)
    svg_root = tree.getroot()
    if _local(svg_root.tag) != "svg":
        raise SystemExit("Root element is not <svg>")

    forest = _collect_clusters_from_svg(svg_root, args.max_parse_depth)
    flat = forest_to_flat(forest)
    if not flat:
        raise SystemExit("No clusters found in SVG (expected Graphviz-style <g class='cluster'>).")

    depths = {n.depth for n in flat}
    print(f"Clusters: {len(flat)}, depth range: {min(depths)}..{max(depths)}")

    run_depth_batches(flat, chunk_size=args.chunk_size, use_cache=not args.no_cache)

    if args.dump_flat:
        with open(args.dump_flat, "w", encoding="utf-8") as f:
            json.dump([asdict(n) for n in flat], f, ensure_ascii=False, indent=2)
        print(f"Wrote flat dump: {args.dump_flat}")

    print("[reduce 1] merge names → canonical groups (≤50)")
    reduce1 = reduce_merge_names(
        flat,
        max_groups=max_mermaid_nodes,
        max_outline_chars=args.max_outline_chars,
        use_cache=not args.no_cache,
    )
    if args.dump_reduce1:
        with open(args.dump_reduce1, "w", encoding="utf-8") as f:
            yaml.safe_dump(reduce1, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        print(f"Wrote reduce-1 YAML: {args.dump_reduce1}")

    nid_to_gid = _nid_to_group_map(reduce1, flat)

    print("[reduce 2] merge edges → Mermaid")
    mermaid_body = reduce_merge_edges(
        reduce1,
        flat,
        nid_to_gid,
        max_nodes=max_mermaid_nodes,
        use_cache=not args.no_cache,
    )

    out_text = (
        f"flowchart TB\n{mermaid_body.removeprefix('flowchart TB').lstrip()}"
        if not mermaid_body.lstrip().lower().startswith("flowchart")
        else mermaid_body
    )

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(out_text if out_text.endswith("\n") else out_text + "\n")

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
