#!/usr/bin/env python3
"""Generate clean thesis-ready graph-construction visuals."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import networkx as nx
import pandas as pd


GRAPH_DIR = Path("artifacts/graph")
FIGURE_DIR = Path("results/figures")
AUDIT_DIR = Path("results/audits")

NODE_COLOURS = {
    "drug": "#4C78A8",
    "gene": "#F58518",
    "variant": "#54A24B",
    "phenotype": "#E45756",
}

EVIDENCE_COLOURS = {
    "clinical": "#4C78A8",
    "label": "#F58518",
    "pathway": "#54A24B",
    "guideline": "#B279A2",
    "literature": "#9D755D",
    "other": "#BAB0AC",
}

NODE_SHAPES = {
    "drug": "o",
    "gene": "s",
    "variant": "^",
    "phenotype": "D",
}


def set_plot_defaults() -> None:
    """Apply consistent figure defaults."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 11,
            "figure.dpi": 120,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Save a figure as PNG, PDF and SVG."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        FIGURE_DIR / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        FIGURE_DIR / f"{stem}.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        FIGURE_DIR / f"{stem}.svg",
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)


def wrap_label(value: object, width: int = 18) -> str:
    """Wrap long node labels across multiple lines."""
    return "\n".join(
        textwrap.wrap(
            str(value),
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load graph nodes, edges and manifest."""
    nodes = pd.read_parquet(GRAPH_DIR / "nodes.parquet")
    edges = pd.read_parquet(GRAPH_DIR / "edges.parquet")

    with (GRAPH_DIR / "graph_manifest.json").open(
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(handle)

    return nodes, edges, manifest


def plot_node_composition(nodes: pd.DataFrame) -> None:
    """Plot graph node counts and percentages."""
    order = ["gene", "variant", "drug", "phenotype"]

    counts = (
        nodes["node_type"]
        .value_counts()
        .reindex(order)
        .astype(int)
    )

    total = int(counts.sum())

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    bars = ax.barh(
        [item.title() for item in counts.index],
        counts.values,
        color=[NODE_COLOURS[item] for item in counts.index],
        height=0.62,
    )

    ax.invert_yaxis()

    for bar, count in zip(bars, counts.values):
        percentage = 100 * count / total

        ax.text(
            bar.get_width() + 400,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,} ({percentage:.1f}%)",
            va="center",
            fontsize=11,
        )

    ax.set_xlim(0, 29_500)
    ax.set_xlabel("Number of nodes")
    ax.set_title(
        "Node composition of the pharmacogenomic knowledge graph",
        pad=16,
    )

    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    ax.text(
        0.99,
        -0.14,
        f"Total nodes: {total:,}",
        transform=ax.transAxes,
        ha="right",
        fontsize=10,
    )

    fig.subplots_adjust(
        left=0.16,
        right=0.94,
        top=0.86,
        bottom=0.18,
    )

    save_figure(fig, "graph_node_composition")


def plot_filtering_flow(manifest: dict) -> None:
    """Plot the relationship filtering and retention process."""
    raw = int(manifest["raw_relationship_rows"])
    excluded = int(manifest["excluded_haplotype_rows"])
    retained = int(manifest["total_edges"])
    retention = 100 * retained / raw

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis("off")

    positions = [0.15, 0.50, 0.85]
    values = [raw, excluded, retained]

    headings = [
        "Raw relationships",
        "Excluded",
        "Final graph edges",
    ]

    descriptions = [
        "ClinPGx relationship records",
        "Haplotype-related records",
        "All in-scope records mapped",
    ]

    colours = [
        NODE_COLOURS["drug"],
        NODE_COLOURS["phenotype"],
        NODE_COLOURS["variant"],
    ]

    for x, value, heading, description, colour in zip(
        positions,
        values,
        headings,
        descriptions,
        colours,
    ):
        ax.text(
            x,
            0.66,
            f"{value:,}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=26,
            fontweight="bold",
            color=colour,
        )

        ax.text(
            x,
            0.46,
            heading,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
        )

        ax.text(
            x,
            0.33,
            description,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#555555",
        )

    for start, end in [(0.26, 0.39), (0.61, 0.74)]:
        arrow = FancyArrowPatch(
            (start, 0.52),
            (end, 0.52),
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=1.8,
            color="#555555",
        )
        ax.add_patch(arrow)

    ax.text(
        0.5,
        0.12,
        f"{retention:.1f}% of raw relationship records retained",
        transform=ax.transAxes,
        ha="center",
        fontsize=12,
        fontweight="bold",
    )

    ax.set_title(
        "Relationship filtering and graph retention",
        pad=14,
    )

    save_figure(fig, "graph_filtering_waterfall")


def plot_metagraph(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
) -> None:
    """Plot the four-node metagraph with paired directed arrows."""
    node_counts = nodes["node_type"].value_counts().to_dict()

    pair_counts = (
        edges.groupby(["source_type", "target_type"])
        .size()
        .to_dict()
    )

    positions = {
        "drug": (-2.2, 0.0),
        "gene": (0.0, 1.55),
        "variant": (0.0, -1.55),
        "phenotype": (2.2, 0.0),
    }

    graph = nx.DiGraph()
    graph.add_nodes_from(positions)

    node_sizes = {
        "drug": 4200,
        "gene": 6000,
        "variant": 4700,
        "phenotype": 3500,
    }

    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.axis("off")

    for node_type in positions:
        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=[node_type],
            node_color=[NODE_COLOURS[node_type]],
            node_size=node_sizes[node_type],
            edgecolors="black",
            linewidths=1.3,
            ax=ax,
        )

    labels = {
        node_type: (
            f"{node_type.title()}\n"
            f"n={node_counts[node_type]:,}"
        )
        for node_type in positions
    }

    nx.draw_networkx_labels(
        graph,
        positions,
        labels=labels,
        font_size=11,
        font_weight="bold",
        ax=ax,
    )

    node_pairs = [
        ("drug", "gene"),
        ("drug", "variant"),
        ("gene", "variant"),
        ("gene", "phenotype"),
        ("variant", "phenotype"),
    ]

    label_positions = {
        ("drug", "gene"): (-1.15, 0.90),
        ("drug", "variant"): (-1.15, -0.90),
        ("gene", "variant"): (0.34, 0.0),
        ("gene", "phenotype"): (1.15, 0.90),
        ("variant", "phenotype"): (1.15, -0.90),
    }

    for source, target in node_pairs:
        forward = int(pair_counts.get((source, target), 0))
        reverse = int(pair_counts.get((target, source), 0))

        if forward == 0 and reverse == 0:
            continue

        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=[(source, target)],
            arrowstyle="-|>",
            arrowsize=20,
            width=1.8,
            edge_color="#555555",
            connectionstyle="arc3,rad=0.13",
            node_size=4500,
            ax=ax,
        )

        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=[(target, source)],
            arrowstyle="-|>",
            arrowsize=20,
            width=1.8,
            edge_color="#555555",
            connectionstyle="arc3,rad=0.13",
            node_size=4500,
            ax=ax,
        )

        if forward == reverse:
            label = f"{forward:,} per direction"
        else:
            label = (
                f"{source.title()}→{target.title()}: {forward:,}\n"
                f"{target.title()}→{source.title()}: {reverse:,}"
            )

        x, y = label_positions[(source, target)]

        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "#BBBBBB",
            },
        )

    ax.set_title(
        "Four-node pharmacogenomic metagraph",
        pad=20,
    )

    ax.text(
        0.5,
        0.03,
        "Paired arrows preserve both ordered source–target orientations.",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color="#555555",
    )

    ax.set_xlim(-3.1, 3.1)
    ax.set_ylim(-2.35, 2.3)

    save_figure(fig, "graph_metagraph_schema")


def plot_representative_subgraph() -> None:
    """Plot the final panitumumab-centred representative subgraph."""
    nodes = pd.read_csv(
        AUDIT_DIR / "representative_subgraph_final_nodes.csv"
    )
    edges = pd.read_csv(
        AUDIT_DIR / "representative_subgraph_final_edges.csv"
    )

    graph = nx.DiGraph()

    for row in nodes.itertuples(index=False):
        graph.add_node(
            int(row.node_index),
            node_name=row.node_name,
            node_type=row.node_type,
        )

    display_edges = (
        edges.sort_values(
            [
                "source_idx",
                "target_idx",
                "evidence",
            ]
        )
        .drop_duplicates(
            [
                "source_idx",
                "target_idx",
            ]
        )
    )

    for row in display_edges.itertuples(index=False):
        graph.add_edge(
            int(row.source_idx),
            int(row.target_idx),
            evidence=row.evidence,
        )

    members = {
        node_type: nodes.loc[
            nodes["node_type"] == node_type,
            "node_index",
        ].astype(int).tolist()
        for node_type in NODE_COLOURS
    }

    x_positions = {
        "drug": -3.0,
        "gene": -1.1,
        "variant": 1.0,
        "phenotype": 3.0,
    }

    y_positions = {
        1: [0.0],
        2: [1.0, -1.0],
        3: [1.55, 0.0, -1.55],
    }

    positions: dict[int, tuple[float, float]] = {}

    for node_type, node_ids in members.items():
        for node_id, y_value in zip(
            node_ids,
            y_positions[len(node_ids)],
        ):
            positions[node_id] = (
                x_positions[node_type],
                y_value,
            )

    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.axis("off")

    for node_type, node_ids in members.items():
        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=node_ids,
            node_shape=NODE_SHAPES[node_type],
            node_color=NODE_COLOURS[node_type],
            node_size=2300 if node_type != "drug" else 2900,
            edgecolors="black",
            linewidths=1.3,
            ax=ax,
        )

    for evidence, colour in EVIDENCE_COLOURS.items():
        selected_edges = [
            (source, target)
            for source, target, data in graph.edges(data=True)
            if data["evidence"] == evidence
        ]

        if not selected_edges:
            continue

        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=selected_edges,
            edge_color=colour,
            width=1.8,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=15,
            alpha=0.8,
            connectionstyle="arc3,rad=0.04",
            ax=ax,
        )

    labels = {
        node: wrap_label(
            data["node_name"],
            width=17
            if data["node_type"] == "phenotype"
            else 13,
        )
        for node, data in graph.nodes(data=True)
    }

    nx.draw_networkx_labels(
        graph,
        positions,
        labels=labels,
        font_size=8.5,
        font_weight="bold",
        ax=ax,
    )

    node_handles = [
        Line2D(
            [0],
            [0],
            marker=NODE_SHAPES[node_type],
            linestyle="",
            markerfacecolor=NODE_COLOURS[node_type],
            markeredgecolor="black",
            markersize=9,
            label=node_type.title(),
        )
        for node_type in [
            "drug",
            "gene",
            "variant",
            "phenotype",
        ]
    ]

    present_evidence = sorted(
        display_edges["evidence"].unique()
    )

    evidence_handles = [
        Line2D(
            [0],
            [0],
            color=EVIDENCE_COLOURS[evidence],
            linewidth=2.5,
            label=evidence.title(),
        )
        for evidence in present_evidence
    ]

    node_legend = ax.legend(
        handles=node_handles,
        title="Node type",
        loc="upper center",
        bbox_to_anchor=(0.30, -0.05),
        ncol=4,
        frameon=False,
    )
    ax.add_artist(node_legend)

    ax.legend(
        handles=evidence_handles,
        title="Evidence",
        loc="upper center",
        bbox_to_anchor=(0.78, -0.05),
        ncol=len(evidence_handles),
        frameon=False,
    )

    ax.set_title(
        "Representative panitumumab-centred subgraph",
        pad=18,
    )

    ax.set_xlim(-3.9, 3.9)
    ax.set_ylim(-2.5, 2.35)

    fig.subplots_adjust(
        bottom=0.20,
        top=0.90,
        left=0.04,
        right=0.96,
    )

    save_figure(
        fig,
        "graph_representative_subgraph_final",
    )


def main() -> None:
    """Generate all graph-construction figures."""
    set_plot_defaults()

    nodes, edges, manifest = load_data()

    plot_node_composition(nodes)
    plot_filtering_flow(manifest)
    plot_metagraph(nodes, edges)
    plot_representative_subgraph()

    print("Overwritten final figures:")

    for filename in [
        "graph_filtering_waterfall.png",
        "graph_node_composition.png",
        "graph_metagraph_schema.png",
        "graph_representative_subgraph_final.png",
    ]:
        print(f"  {FIGURE_DIR / filename}")


if __name__ == "__main__":
    main()
