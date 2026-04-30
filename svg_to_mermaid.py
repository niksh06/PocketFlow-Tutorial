#!/usr/bin/env python3
"""
Convert a Graphviz-style SVG (many <g class="cluster">) into Mermaid flowchart syntax.

844+ clusters will not render sanely in most Mermaid viewers — use --max-subgraphs,
--max-depth, or emit multiple files and link from generated docs.

Typical integration with this repo: run after you have an SVG (e.g. from Graphviz),
then paste or include the .mmd output in docs, or split into chapter diagrams.

Usage:
  python svg_to_mermaid.py input.svg -o out.mmd
  python svg_to_mermaid.py huge.svg -o part1.mmd --max-subgraphs 50 --max-depth 4
"""
from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Optional


def _local(tag: str) -> str:
    if tag.startswith("{"):
        return tag.split("}", 1)[-1]
    return tag


def _cls(elem: ET.Element) -> str:
    return (elem.get("class") or "").strip()


def _title_child(elem: ET.Element) -> str:
    for c in elem:
        if _local(c.tag) == "title" and c.text:
            return c.text.strip()
    return ""


def _sanitize_mermaid_id(s: str) -> str:
    s = re.sub(r"[^\w]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s or s[0].isdigit():
        s = "N_" + s
    return s[:120]


def _escape_label(s: str) -> str:
    return s.replace('"', "#quot;").replace("\n", " ")


@dataclass
class ClusterNode:
    mid: str
    label: str
    children_clusters: List["ClusterNode"] = field(default_factory=list)
    leaf_nodes: List[tuple[str, str]] = field(default_factory=list)  # (mid, label)


def _parse_cluster_g(g: ET.Element, id_counter: list[int], max_depth: int, depth: int) -> Optional[ClusterNode]:
    """Parse one <g class='... cluster ...'> and its direct cluster children and node children."""
    classes = set(_cls(g).split())
    if "cluster" not in classes and not (g.get("id") or "").lower().startswith("cluster"):
        return None

    title = _title_child(g) or g.get("id") or "cluster"
    id_counter[0] += 1
    mid = _sanitize_mermaid_id(f"c_{id_counter[0]}_{title}")

    node = ClusterNode(mid=mid, label=title)

    if depth >= max_depth:
        return node

    for child in g:
        if _local(child.tag) != "g":
            continue
        cc = _cls(child)
        if "cluster" in cc.split() or (child.get("id") or "").lower().startswith("cluster"):
            sub = _parse_cluster_g(child, id_counter, max_depth, depth + 1)
            if sub:
                node.children_clusters.append(sub)
            continue
        if "node" in cc.split():
            nt = _title_child(child) or child.get("id") or "node"
            id_counter[0] += 1
            nid = _sanitize_mermaid_id(f"n_{id_counter[0]}_{nt}")
            node.leaf_nodes.append((nid, nt))

    return node


def _find_root_groups(svg_root: ET.Element) -> List[ET.Element]:
    """Graphviz often wraps everything in <g class='graph'> or similar."""
    out: List[ET.Element] = []
    for child in svg_root:
        if _local(child.tag) != "g":
            continue
        out.append(child)
    return out


def _collect_clusters_from_svg(svg_root: ET.Element, max_depth: int) -> List[ClusterNode]:
    id_counter = [0]
    roots = _find_root_groups(svg_root)
    clusters: List[ClusterNode] = []

    for g in roots:
        if "cluster" in _cls(g).split():
            c = _parse_cluster_g(g, id_counter, max_depth, 0)
            if c:
                clusters.append(c)
        else:
            for child in g:
                if _local(child.tag) != "g":
                    continue
                if "cluster" in _cls(child).split() or (child.get("id") or "").lower().startswith(
                    "cluster"
                ):
                    c = _parse_cluster_g(child, id_counter, max_depth, 0)
                    if c:
                        clusters.append(c)

    return clusters


def _emit_mermaid(clusters: List[ClusterNode], direction: str) -> str:
    lines = ["flowchart " + direction, ""]

    def emit_cluster(c: ClusterNode, indent: str) -> None:
        safe_label = _escape_label(c.label)
        lines.append(f'{indent}subgraph {c.mid}["{safe_label}"]')
        lines.append(f"{indent}  direction TB")
        for nid, nlabel in c.leaf_nodes:
            lines.append(f'{indent}  {nid}["{_escape_label(nlabel)}"]')
        for sub in c.children_clusters:
            emit_cluster(sub, indent + "  ")
        lines.append(f"{indent}end")
        lines.append("")

    for top in clusters:
        emit_cluster(top, "")
    return "\n".join(lines).rstrip() + "\n"


def _limit_subgraphs(clusters: List[ClusterNode], limit: Optional[int]) -> tuple[List[ClusterNode], int]:
    if limit is None:
        total = _count_clusters(clusters)
        return clusters, total

    out: List[ClusterNode] = []
    seen = 0

    def walk(c: ClusterNode) -> Optional[ClusterNode]:
        nonlocal seen
        if seen >= limit:
            return None
        seen += 1
        copy = ClusterNode(mid=c.mid, label=c.label, leaf_nodes=list(c.leaf_nodes))
        for sub in c.children_clusters:
            if seen >= limit:
                break
            sub_copy = walk(sub)
            if sub_copy:
                copy.children_clusters.append(sub_copy)
        return copy

    for c in clusters:
        if seen >= limit:
            break
        cc = walk(c)
        if cc:
            out.append(cc)

    return out, seen


def _count_clusters(clusters: List[ClusterNode]) -> int:
    n = 0

    def rec(c: ClusterNode) -> None:
        nonlocal n
        n += 1
        for s in c.children_clusters:
            rec(s)

    for c in clusters:
        rec(c)
    return n


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Graphviz-style clustered SVG → Mermaid flowchart (subgraphs)."
    )
    parser.add_argument("svg", help="Path to .svg file")
    parser.add_argument("-o", "--output", default="diagram.mmd", help="Output .mmd path")
    parser.add_argument(
        "--direction",
        default="TB",
        choices=["TB", "BT", "LR", "RL"],
        help="Mermaid flowchart direction (default TB)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=99,
        help="Max nesting depth for clusters (default 99)",
    )
    parser.add_argument(
        "--max-subgraphs",
        type=int,
        default=None,
        help="Stop after this many subgraphs ( breadth-first cut; use for huge SVGs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts only, do not write file",
    )
    args = parser.parse_args()

    tree = ET.parse(args.svg)
    root = tree.getroot()
    if _local(root.tag) != "svg":
        raise SystemExit("Root element is not <svg>")

    clusters = _collect_clusters_from_svg(root, max_depth=args.max_depth)
    raw_count = _count_clusters(clusters)
    limited, used = _limit_subgraphs(clusters, args.max_subgraphs)

    print(f"Clusters parsed (full tree): {raw_count}")
    if args.max_subgraphs is not None:
        print(f"Clusters emitted (after --max-subgraphs): {used}")

    if args.dry_run:
        return

    text = _emit_mermaid(limited, args.direction)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote {args.output} ({len(text)} bytes)")


if __name__ == "__main__":
    main()
