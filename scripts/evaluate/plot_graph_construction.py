#!/usr/bin/env python3
"""Generate graph-construction figures from audited graph artifacts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


GRAPH_DIR = Path("artifacts/graph")
RESULTS_DIR = Path("results/figures")
AUDIT_DIR = Path("results/audits")

NODES_PATH = GRAPH_DIR / "nodes.parquet"
EDGES_PATH = GRAPH_DIR / "edges.parquet"
MANIFEST_PATH = GRAPH_DIR / "graph_manifest.json"

NODE_ORDER = ["gene", "variant", "drug", "phenotype"]
NODE_SHAPES = {
    "drug": "o",
    "gene": "s",
    "variant": "^",
    "phenotype": "D",
}


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Save one figure in vector and high-resolution raster formats."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        RESULTS_DIR / f"{stem}.svg",
        bbox_inches="tight",
    )
    fig.savefig(
        RESULTS_DIR / f"{stem}.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        RESULTS_DIR / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def load_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    nodes = pd.read_parquet(NODES_PATH)
    edges = pd.read_parquet(EDGES_PATH)

    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    return nodes, edges, manifest


def plot_filtering_waterfall(manifest: dict) -> None:
    raw = manifest["raw_relationship_rows"]
    excluded = manifest["excluded_haplotype_rows"]
    final = manifest["total_edges"]

    labels = [
        "Raw relationships",
        "Haplotype-related\nexclusions",
        "Final KG edges",
    ]

    values = [raw, -excluded, final]
    starts = [0, raw, 0]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for index, (label, value, start) in enumerate(
        zip(labels, values, starts)
    ):
        ax.bar(
            index,
            value,
            bottom=start,
            width=0.62,
        )

        if value >= 0:
            label_y = start + value
            vertical_alignment = "bottom"
            offset = 1000
        else:
            label_y = start + value
            vertical_alignment = "top"
            offset = -1000

        prefix = "−" if value < 0 else ""
        ax.text(
            index,
            label_y + offset,
            f"{prefix}{abs(value):,}",
            ha="center",
            va=vertical_alignment,
            fontweight="bold",
        )

    ax.plot(
        [0.31, 0.69],
        [raw, raw],
        linestyle="--",
        linewidth=1,
    )
    ax.plot(
        [1.31, 1.69],
        [final, final],
        linestyle="--",
        linewidth=1,
    )

    retention = 100 * final / raw

    ax.text(
        1,
        final / 2,
        f"{retention:.1f}% retained",
        ha="center",
        va="center",
    )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Relationship records")
    ax.set_title("Relationship filtering during graph construction")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)

    save_figure(fig, "graph_filtering_waterfall")


def plot_node_composition(nodes: pd.DataFrame) -> None:
    counts = (
        nodes["node_type"]
        .value_counts()
        .reindex(NODE_ORDER)
        .dropna()
        .astype(int)
    )

    total = int(counts.sum())
    percentages = 100 * counts / total

    fig, ax = plt.subplots(figsize=(8, 5.5))

    bars = ax.barh(
        counts.index.str.title(),
        counts.values,
    )

    ax.invert_yaxis()

    for bar, count, percentage in zip(
        bars,
        counts.values,
        percentages.values,
    ):
        ax.text(
            bar.get_width() + total * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,} ({percentage:.1f}%)",
            va="center",
        )

    ax.set_xlabel("Number of nodes")
    ax.set_title(
        f"Node composition of the reconstructed graph (n={total:,})"
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, counts.max() * 1.25)

    save_figure(fig, "graph_node_composition")


def plot_metagraph(nodes: pd.DataFrame, edges: pd.DataFrame) -> None:
    node_counts = nodes["node_type"].value_counts().to_dict()

    pair_counts = (
        edges.groupby(["source_type", "target_type"])
        .size()
        .reset_index(name="edge_count")
    )

    graph = nx.DiGraph()

    for node_type, count in node_counts.items():
        graph.add_node(node_type, count=int(count))

    for row in pair_counts.itertuples(index=False):
        graph.add_edge(
            row.source_type,
            row.target_type,
            weight=int(row.edge_count),
        )

    positions = {
        "drug": (-1.6, 0.0),
        "gene": (0.0, 1.25),
        "variant": (0.0, -1.25),
        "phenotype": (1.6, 0.0),
    }

    fig, ax = plt.subplots(figsize=(10, 7))

    maximum_count = max(node_counts.values())

    node_sizes = [
        2800 + 5000 * node_counts[node] / maximum_count
        for node in graph.nodes
    ]

    nx.draw_networkx_nodes(
        graph,
        positions,
        node_size=node_sizes,
        linewidths=1.5,
        edgecolors="black",
        ax=ax,
    )

    maximum_edge_count = max(
        data["weight"]
        for _, _, data in graph.edges(data=True)
    )

    edge_widths = [
        0.8 + 4.2 * graph[u][v]["weight"] / maximum_edge_count
        for u, v in graph.edges
    ]

    nx.draw_networkx_edges(
        graph,
        positions,
        width=edge_widths,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=18,
        connectionstyle="arc3,rad=0.09",
        node_size=node_sizes,
        alpha=0.65,
        ax=ax,
    )

    node_labels = {
        node: f"{node.title()}\nn={node_counts[node]:,}"
        for node in graph.nodes
    }

    nx.draw_networkx_labels(
        graph,
        positions,
        labels=node_labels,
        font_size=10,
        font_weight="bold",
        ax=ax,
    )

    edge_labels = {
        (source, target): f"{data['weight']:,}"
        for source, target, data in graph.edges(data=True)
    }

    nx.draw_networkx_edge_labels(
        graph,
        positions,
        edge_labels=edge_labels,
        font_size=7,
        rotate=False,
        label_pos=0.48,
        bbox={
            "boxstyle": "round,pad=0.15",
            "alpha": 0.8,
            "linewidth": 0,
        },
        ax=ax,
    )

    ax.set_title(
        "Metagraph schema of the four-node pharmacogenomic graph"
    )
    ax.text(
        0.5,
        0.01,
        "Arrow labels show directed edge counts. "
        "Orientation supports message passing and does not necessarily "
        "imply biological causality.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
    )
    ax.axis("off")

    save_figure(fig, "graph_metagraph_schema")


def create_neighbour_maps(
    edges: pd.DataFrame,
) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    outgoing: dict[int, set[int]] = defaultdict(set)
    undirected: dict[int, set[int]] = defaultdict(set)

    for row in edges.itertuples(index=False):
        source = int(row.source_idx)
        target = int(row.target_idx)

        outgoing[source].add(target)
        undirected[source].add(target)
        undirected[target].add(source)

    return outgoing, undirected


def rank_candidate_drugs(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
) -> pd.DataFrame:
    node_type = nodes.set_index("node_index")["node_type"].to_dict()
    node_name = nodes.set_index("node_index")["node_name"].to_dict()

    _, neighbours = create_neighbour_maps(edges)

    evidence_by_drug: dict[int, set[str]] = defaultdict(set)

    for row in edges.itertuples(index=False):
        source = int(row.source_idx)
        target = int(row.target_idx)

        if node_type.get(source) == "drug":
            evidence_by_drug[source].add(row.evidence)
        if node_type.get(target) == "drug":
            evidence_by_drug[target].add(row.evidence)

    records = []

    drug_nodes = nodes[nodes["node_type"] == "drug"]

    for drug_row in drug_nodes.itertuples(index=False):
        drug_idx = int(drug_row.node_index)
        first_hop = neighbours.get(drug_idx, set())

        genes = {
            node
            for node in first_hop
            if node_type.get(node) == "gene"
        }
        variants = {
            node
            for node in first_hop
            if node_type.get(node) == "variant"
        }
        direct_phenotypes = {
            node
            for node in first_hop
            if node_type.get(node) == "phenotype"
        }

        mediated_phenotypes: set[int] = set()

        for intermediate in genes | variants:
            mediated_phenotypes.update(
                node
                for node in neighbours.get(intermediate, set())
                if node_type.get(node) == "phenotype"
            )

        phenotypes = direct_phenotypes | mediated_phenotypes

        evidence_count = len(evidence_by_drug.get(drug_idx, set()))

        # Reward all four node types and manageable local complexity.
        diversity_bonus = (
            5 * int(len(genes) >= 2)
            + 5 * int(len(variants) >= 1)
            + 5 * int(len(phenotypes) >= 1)
            + 3 * int(evidence_count >= 2)
        )

        size_penalty = max(
            0,
            len(first_hop) - 25,
        ) * 0.2

        score = (
            diversity_bonus
            + min(len(genes), 5)
            + min(len(variants), 4)
            + min(len(phenotypes), 4)
            + min(evidence_count, 4)
            - size_penalty
        )

        records.append(
            {
                "drug_idx": drug_idx,
                "drug_id": drug_row.node_id,
                "drug_name": node_name[drug_idx],
                "direct_degree": len(first_hop),
                "connected_genes": len(genes),
                "connected_variants": len(variants),
                "reachable_phenotypes": len(phenotypes),
                "evidence_categories": evidence_count,
                "subgraph_score": score,
            }
        )

    candidates = pd.DataFrame(records).sort_values(
        [
            "subgraph_score",
            "connected_genes",
            "connected_variants",
            "reachable_phenotypes",
        ],
        ascending=False,
    )

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(
        AUDIT_DIR / "representative_subgraph_candidates.csv",
        index=False,
    )

    return candidates


def choose_representative_nodes(
    drug_idx: int,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
) -> set[int]:
    node_type = nodes.set_index("node_index")["node_type"].to_dict()

    incident = edges[
        (edges["source_idx"] == drug_idx)
        | (edges["target_idx"] == drug_idx)
    ].copy()

    neighbour_ids = set(incident["source_idx"]) | set(
        incident["target_idx"]
    )
    neighbour_ids.discard(drug_idx)

    def ranked_neighbours(
        requested_type: str,
        limit: int,
    ) -> list[int]:
        candidates = [
            int(node)
            for node in neighbour_ids
            if node_type.get(int(node)) == requested_type
        ]

        degree = (
            pd.concat(
                [
                    edges["source_idx"],
                    edges["target_idx"],
                ]
            )
            .value_counts()
            .to_dict()
        )

        return sorted(
            candidates,
            key=lambda node: degree.get(node, 0),
            reverse=True,
        )[:limit]

    selected_genes = ranked_neighbours("gene", 3)
    selected_variants = ranked_neighbours("variant", 3)
    selected_phenotypes = ranked_neighbours("phenotype", 2)

    intermediate_nodes = selected_genes + selected_variants

    phenotype_candidates: set[int] = set(selected_phenotypes)

    for intermediate in intermediate_nodes:
        connected = edges[
            (edges["source_idx"] == intermediate)
            | (edges["target_idx"] == intermediate)
        ]

        connected_nodes = set(connected["source_idx"]) | set(
            connected["target_idx"]
        )

        phenotype_candidates.update(
            int(node)
            for node in connected_nodes
            if node_type.get(int(node)) == "phenotype"
        )

    phenotype_degree = (
        pd.concat(
            [
                edges.loc[
                    edges["source_type"] == "phenotype",
                    "source_idx",
                ],
                edges.loc[
                    edges["target_type"] == "phenotype",
                    "target_idx",
                ],
            ]
        )
        .value_counts()
        .to_dict()
    )

    selected_phenotypes = sorted(
        phenotype_candidates,
        key=lambda node: phenotype_degree.get(node, 0),
        reverse=True,
    )[:3]

    return {
        drug_idx,
        *selected_genes,
        *selected_variants,
        *selected_phenotypes,
    }


def collapse_reverse_edges(
    sub_edges: pd.DataFrame,
) -> pd.DataFrame:
    """Keep one readable record per node pair and evidence/association."""
    collapsed = sub_edges.copy()

    collapsed["pair_min"] = collapsed[
        ["source_idx", "target_idx"]
    ].min(axis=1)
    collapsed["pair_max"] = collapsed[
        ["source_idx", "target_idx"]
    ].max(axis=1)

    collapsed = collapsed.sort_values(
        [
            "pair_min",
            "pair_max",
            "evidence",
            "association",
            "source_idx",
            "target_idx",
        ]
    )

    collapsed = collapsed.drop_duplicates(
        [
            "pair_min",
            "pair_max",
            "evidence",
            "association",
        ]
    )

    return collapsed.drop(
        columns=["pair_min", "pair_max"]
    )


def plot_representative_subgraph(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    requested_drug_id: str | None = None,
) -> None:
    candidates = rank_candidate_drugs(nodes, edges)

    eligible = candidates[
        (candidates["connected_genes"] >= 2)
        & (candidates["connected_variants"] >= 1)
        & (candidates["reachable_phenotypes"] >= 1)
    ]

    if eligible.empty:
        raise RuntimeError(
            "No drug satisfies the representative-subgraph criteria."
        )

    if requested_drug_id is None:
        selected_drug = eligible.iloc[0]
    else:
        matched = eligible[
            eligible["drug_id"] == requested_drug_id
        ]

        if matched.empty:
            raise ValueError(
                f"Drug ID {requested_drug_id!r} was not found among "
                "eligible representative-subgraph candidates."
            )

        selected_drug = matched.iloc[0]

    drug_idx = int(selected_drug["drug_idx"])

    selected_nodes = choose_representative_nodes(
        drug_idx,
        nodes,
        edges,
    )

    sub_nodes = nodes[
        nodes["node_index"].isin(selected_nodes)
    ].copy()

    sub_edges = edges[
        edges["source_idx"].isin(selected_nodes)
        & edges["target_idx"].isin(selected_nodes)
    ].copy()

    sub_edges = collapse_reverse_edges(sub_edges)

    graph = nx.MultiDiGraph()

    for row in sub_nodes.itertuples(index=False):
        graph.add_node(
            int(row.node_index),
            node_name=row.node_name,
            node_type=row.node_type,
        )

    for row in sub_edges.itertuples(index=False):
        graph.add_edge(
            int(row.source_idx),
            int(row.target_idx),
            evidence=row.evidence,
            association=row.association,
        )

    position_groups = {
        "drug": (-2.4, 0.0),
        "gene": (-0.8, 0.0),
        "variant": (0.8, -1.1),
        "phenotype": (2.3, 0.0),
    }

    positions: dict[int, tuple[float, float]] = {}

    for node_type_name in [
        "drug",
        "gene",
        "variant",
        "phenotype",
    ]:
        members = [
            node
            for node, data in graph.nodes(data=True)
            if data["node_type"] == node_type_name
        ]

        base_x, base_y = position_groups[node_type_name]

        if len(members) == 1:
            y_values = [base_y]
        else:
            y_values = np.linspace(
                base_y + 1.2,
                base_y - 1.2,
                len(members),
            )

        for node, y_value in zip(members, y_values):
            positions[node] = (base_x, float(y_value))

    fig, ax = plt.subplots(figsize=(13, 8))

    for node_type_name, shape in NODE_SHAPES.items():
        members = [
            node
            for node, data in graph.nodes(data=True)
            if data["node_type"] == node_type_name
        ]

        if not members:
            continue

        sizes = [
            3300 if node == drug_idx else 2200
            for node in members
        ]

        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=members,
            node_shape=shape,
            node_size=sizes,
            linewidths=1.5,
            edgecolors="black",
            label=node_type_name.title(),
            ax=ax,
        )

    edge_list = list(graph.edges(keys=True))

    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=edge_list,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=16,
        width=1.3,
        alpha=0.7,
        connectionstyle="arc3,rad=0.08",
        ax=ax,
    )

    labels = {
        node: (
            data["node_name"]
            if len(str(data["node_name"])) <= 24
            else str(data["node_name"])[:21] + "..."
        )
        for node, data in graph.nodes(data=True)
    }

    nx.draw_networkx_labels(
        graph,
        positions,
        labels=labels,
        font_size=8,
        font_weight="bold",
        ax=ax,
    )

    edge_labels = {}

    for source, target, key, data in graph.edges(
        keys=True,
        data=True,
    ):
        association = data["association"].replace("_", " ")
        edge_labels[(source, target, key)] = (
            f"{data['evidence']} · {association}"
        )

    nx.draw_networkx_edge_labels(
        graph,
        positions,
        edge_labels=edge_labels,
        font_size=6.5,
        rotate=False,
        label_pos=0.5,
        bbox={
            "boxstyle": "round,pad=0.1",
            "alpha": 0.75,
            "linewidth": 0,
        },
        ax=ax,
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        frameon=False,
    )

    ax.set_title(
        "Representative drug-centred pharmacogenomic subgraph\n"
        f"Selected drug: {selected_drug['drug_name']}"
    )

    ax.text(
        0.5,
        0.01,
        "One relationship orientation is shown per matched "
        "evidence–association pair for readability.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
    )

    ax.axis("off")

    safe_drug_name = (
        str(selected_drug["drug_name"])
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
    )

    save_figure(
        fig,
        f"graph_representative_subgraph_{safe_drug_name}",
    )

    sub_nodes.to_csv(
        AUDIT_DIR / "representative_subgraph_nodes.csv",
        index=False,
    )
    sub_edges.to_csv(
        AUDIT_DIR / "representative_subgraph_edges.csv",
        index=False,
    )

    print("\nRepresentative subgraph")
    print("-----------------------")
    print(f"Drug: {selected_drug['drug_name']}")
    print(f"Drug ID: {selected_drug['drug_id']}")
    print(f"Displayed nodes: {len(sub_nodes)}")
    print(f"Displayed edges: {len(sub_edges)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate graph-construction figures."
    )
    parser.add_argument(
        "--drug-id",
        type=str,
        default=None,
        help=(
            "Optional PharmGKB drug ID for the representative "
            "subgraph. When omitted, the highest-ranked eligible "
            "drug is selected automatically."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nodes, edges, manifest = load_artifacts()

    plot_filtering_waterfall(manifest)
    plot_node_composition(nodes)
    plot_metagraph(nodes, edges)
    plot_representative_subgraph(
        nodes,
        edges,
        requested_drug_id=args.drug_id,
    )

    print("\nGenerated figures:")
    for path in sorted(RESULTS_DIR.glob("graph_*")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
