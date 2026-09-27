#!/usr/bin/env python

"""
Final dissertation figure builder.

Project:
Interpretable Graph Neural Networks for Drug-Gene Inference
and Phenotype Contextualisation in Pharmacogenomics

This script uses only outputs from MLPKGA007_MSC_clean.

Main figures:
1. Knowledge graph composition
2. Strict-cold predictive performance and molecular-feature analysis
3. Candidate prioritisation funnel
4. Cross-model interpretability ranks
5. Relation-family ablation vs matched random controls
6. Literature-supported candidate map
7. External-validation enrichment and sensitivity

Supplementary:
S1. RGAT score/rank instability
S2. Pathway-context coverage
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# PATHS
# ============================================================

BASE = Path.home() / "MLPKGA007_MSC_clean"

SUMMARY = (
    BASE
    / "results"
    / "consensus_inference"
    / "molecular"
    / "strict_cold"
    / "summary"
)

MOLECULAR_SUMMARY = BASE / "results" / "molecular_summary"

FIG_MAIN = BASE / "figures" / "main"
FIG_SUPP = BASE / "figures" / "supplementary"

FIG_MAIN.mkdir(parents=True, exist_ok=True)
FIG_SUPP.mkdir(parents=True, exist_ok=True)


# ============================================================
# STYLE
# ============================================================

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.titlesize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


MODEL_COLORS = {
    "GCN": "#4C78A8",
    "RGCN": "#59A14F",
    "RGAT": "#F28E2B",
}

NODE_COLORS = {
    "Drug": "#59A14F",
    "Gene": "#4C78A8",
    "Variant": "#F28E2B",
    "Phenotype": "#E15759",
}

EVIDENCE_COLORS = {
    "direct_supported": "#59A14F",
    "direct_mixed": "#F28E2B",
    "indirect_support": "#4C78A8",
}

RELATION_MARKERS = {
    "drug_gene_clinical_associated": "o",
    "drug_gene_pathway_associated": "s",
    "drug_gene_label_associated": "D",
    "drug_gene_clinical_ambiguous": "^",
}


# ============================================================
# GRAPH CONSTANTS
# ============================================================

GRAPH_COUNTS = {
    "Drug": 5210,
    "Gene": 25042,
    "Variant": 7205,
    "Phenotype": 1619,
}

TOTAL_NODES = 39076
DIRECTED_EDGES = 47072
RELATION_TYPES = 56

HOST_VARIANTS = 6427
HOST_GENES = 2088


# ============================================================
# HELPERS
# ============================================================

def read_csv(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")

    return pd.read_csv(path, low_memory=False)


def summary_csv(name):
    return read_csv(SUMMARY / name)


def molecular_csv(name):
    return read_csv(MOLECULAR_SUMMARY / name)


def save_figure(fig, stem, supplementary=False):
    outdir = FIG_SUPP if supplementary else FIG_MAIN

    pdf = outdir / f"{stem}.pdf"
    png = outdir / f"{stem}.png"

    fig.savefig(
        pdf,
        bbox_inches="tight",
        pad_inches=0.15,
    )

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.15,
    )

    plt.close(fig)

    print(f"[FIG] {pdf.relative_to(BASE)}")
    print(f"[FIG] {png.relative_to(BASE)}")



def annotate_bars(
    ax,
    bars,
    fmt="{:.3f}",
    fontsize=8.5,
    offset_fraction=0.02,
):
    """Add exact numerical values above vertical bars."""
    ymin, ymax = ax.get_ylim()
    offset = (ymax - ymin) * offset_fraction

    for bar in bars:
        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight="bold",
        )


def pretty_relation(x):
    mapping = {
        "phenotype_variant_clinical_associated":
            "Phenotype-variant clinical",
        "drug_variant_clinical_associated":
            "Drug-variant clinical",
        "drug_gene_clinical_associated":
            "Drug-gene clinical",
        "drug_gene_pathway_associated":
            "Drug-gene pathway",
        "drug_gene_label_associated":
            "Drug-gene label",
        "drug_gene_guideline_associated":
            "Drug-gene guideline associated",
        "drug_gene_guideline_ambiguous":
            "Drug-gene guideline ambiguous",
        "drug_gene_clinical_ambiguous":
            "Drug-gene clinical ambiguous",
        "gene_phenotype_clinical_associated":
            "Gene-phenotype clinical",
    }

    return mapping.get(
        str(x),
        str(x).replace("_", " "),
    )


# ============================================================
# FIGURE 1
# KNOWLEDGE GRAPH COMPOSITION
# ============================================================

def build_figure_01():

    labels = ["Gene", "Variant", "Drug", "Phenotype"]
    counts = [GRAPH_COUNTS[x] for x in labels]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    y = np.arange(len(labels))

    bars = ax.barh(
        y,
        counts,
        color=[NODE_COLORS[x] for x in labels],
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    ax.set_xlabel("Number of nodes")
    ax.set_xlim(0, 34000)
    ax.set_title("Heterogeneous pharmacogenomics knowledge graph")

    ax.grid(
        axis="x",
        alpha=0.25,
        linestyle="--",
    )

    xmax = max(counts)

    for bar, n in zip(bars, counts):
        ax.text(
            bar.get_width() + xmax * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{n:,}",
            va="center",
            fontweight="bold",
        )

    metadata = (
        f"Total nodes: {TOTAL_NODES:,}\n"
        f"Directed edges: {DIRECTED_EDGES:,}\n"
        f"Relation types: {RELATION_TYPES}\n\n"
        f"Host-gene correction\n"
        f"{HOST_VARIANTS:,} variants mapped to\n"
        f"{HOST_GENES:,} authoritative host genes"
    )

    ax.text(
        0.98,
        0.06,
        metadata,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="gray",
            alpha=0.95,
        ),
    )

    ax.set_xlim(0, xmax * 1.28)

    fig.tight_layout()

    save_figure(
        fig,
        "fig01_kg_composition",
    )


# ============================================================
# FIGURE 2
# STRICT-COLD MODEL PERFORMANCE AND MOLECULAR FEATURES
# ============================================================

def build_figure_02():

    identifier = read_csv(
        BASE / "results" / "model_test_results_summary.csv"
    )

    molecular = molecular_csv(
        "molecular_test_results_summary.csv"
    )

    molecular_seed = molecular_csv(
        "molecular_test_results_by_seed.csv"
    )

    subset = molecular_csv(
        "molecular_test_results_by_subset_summary.csv"
    )

    paired = molecular_csv(
        "paired_molecular_vs_id_by_subset.csv"
    )

    models = ["gcn", "rgcn", "rgat"]
    labels = ["GCN", "RGCN", "RGAT"]

    x = np.arange(len(models))
    width = 0.34

    # Identifier-only strict-cold results
    id_sc = (
        identifier[
            identifier["protocol"] == "strict_cold"
        ]
        .set_index("model")
        .loc[models]
    )

    # Molecular strict-cold results
    mol_sc = (
        molecular[
            molecular["protocol"] == "strict_cold"
        ]
        .set_index("model")
        .loc[models]
    )

    # Molecular Hits@10 SD from the three individual seeds
    mol_sc_seed = molecular_seed[
        molecular_seed["protocol"] == "strict_cold"
    ].copy()

    mol_hits10_sd = (
        mol_sc_seed
        .groupby("model")["hits_at_10"]
        .std()
        .reindex(models)
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.5, 8.8),
    )

    # --------------------------------------------------------
    # Panel A: MRR
    # --------------------------------------------------------

    ax = axes[0, 0]

    bars_id = ax.bar(
        x - width / 2,
        id_sc["mrr_mean"].values,
        width,
        yerr=id_sc["mrr_sd"].values,
        capsize=4,
        label="Identifier-only",
        alpha=0.85,
    )

    bars_mol = ax.bar(
        x + width / 2,
        mol_sc["mean_mrr"].values,
        width,
        yerr=mol_sc["sd_mrr"].values,
        capsize=4,
        label="Molecular",
        alpha=0.85,
    )

    max_a = max(
        (id_sc["mrr_mean"] + id_sc["mrr_sd"]).max(),
        (mol_sc["mean_mrr"] + mol_sc["sd_mrr"]).max(),
    )

    ax.set_ylim(0, max_a * 1.28)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Reciprocal Rank")
    ax.set_title("A. Strict-cold MRR across three seeds")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=False)

    annotate_bars(ax, bars_id)
    annotate_bars(ax, bars_mol)

    # --------------------------------------------------------
    # Panel B: Hits@10
    # --------------------------------------------------------

    ax = axes[0, 1]

    bars_id = ax.bar(
        x - width / 2,
        id_sc["hits_at_10_mean"].values,
        width,
        yerr=id_sc["hits_at_10_sd"].values,
        capsize=4,
        label="Identifier-only",
        alpha=0.85,
    )

    bars_mol = ax.bar(
        x + width / 2,
        mol_sc["mean_hits_at_10"].values,
        width,
        yerr=mol_hits10_sd.values,
        capsize=4,
        label="Molecular",
        alpha=0.85,
    )

    max_b = max(
        (
            id_sc["hits_at_10_mean"]
            + id_sc["hits_at_10_sd"]
        ).max(),
        (
            mol_sc["mean_hits_at_10"]
            + mol_hits10_sd
        ).max(),
    )

    ax.set_ylim(0, max_b * 1.28)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Hits@10")
    ax.set_title("B. Strict-cold Hits@10 across three seeds")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=False)

    annotate_bars(ax, bars_id)
    annotate_bars(ax, bars_mol)

    # --------------------------------------------------------
    # Panel C: with vs without SMILES
    # --------------------------------------------------------

    ax = axes[1, 0]

    smi = subset[
        (subset["protocol"] == "strict_cold")
        & subset["model"].isin(models)
        & subset["subset"].isin(
            ["with_smiles", "without_smiles"]
        )
    ].copy()

    with_smiles = (
        smi[
            smi["subset"] == "with_smiles"
        ]
        .set_index("model")
        .loc[models]
    )

    without_smiles = (
        smi[
            smi["subset"] == "without_smiles"
        ]
        .set_index("model")
        .loc[models]
    )

    bars_with = ax.bar(
        x - width / 2,
        with_smiles["mean_mrr"].values,
        width,
        yerr=with_smiles["sd_mrr"].values,
        capsize=4,
        label="With SMILES",
        alpha=0.85,
    )

    bars_without = ax.bar(
        x + width / 2,
        without_smiles["mean_mrr"].values,
        width,
        yerr=without_smiles["sd_mrr"].values,
        capsize=4,
        label="Without SMILES",
        alpha=0.85,
    )

    max_c = max(
        (
            with_smiles["mean_mrr"]
            + with_smiles["sd_mrr"]
        ).max(),
        (
            without_smiles["mean_mrr"]
            + without_smiles["sd_mrr"]
        ).max(),
    )

    ax.set_ylim(0, max_c * 1.30)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Reciprocal Rank")
    ax.set_title(
        "C. Molecular runs stratified by SMILES availability"
    )
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=False)

    annotate_bars(ax, bars_with)
    annotate_bars(ax, bars_without)

    rgcn_with = float(
        with_smiles.loc["rgcn", "mean_mrr"]
    )

    rgcn_without = float(
        without_smiles.loc["rgcn", "mean_mrr"]
    )

    ax.text(
        0.98,
        0.76,
        (
            "RGCN three-seed mean\n"
            f"With SMILES: {rgcn_with:.4f}\n"
            f"Without SMILES: {rgcn_without:.4f}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="gray",
        ),
    )

    # --------------------------------------------------------
    # Panel D: RGCN molecular minus ID-only by seed
    # --------------------------------------------------------

    ax = axes[1, 1]

    rgcn_pair = paired[
        (paired["model"] == "rgcn")
        & (paired["protocol"] == "strict_cold")
        & (
            paired["subset"]
            == "with_smiles_structure_clean"
        )
    ].copy()

    rgcn_pair = rgcn_pair.sort_values("seed")

    x_seed = np.arange(len(rgcn_pair))
    delta = rgcn_pair["delta_mrr"].values

    lower = (
        delta
        - rgcn_pair["bootstrap_ci_low"].values
    )

    upper = (
        rgcn_pair["bootstrap_ci_high"].values
        - delta
    )

    ax.errorbar(
        x_seed,
        delta,
        yerr=[lower, upper],
        fmt="o",
        capsize=5,
        markersize=7,
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xticks(x_seed)
    ax.set_xticklabels(
        [
            f"Seed {seed}"
            for seed in rgcn_pair["seed"]
        ]
    )

    ax.set_ylabel(
        "ΔMRR: molecular minus identifier-only"
    )

    ax.set_title(
        "D. RGCN molecular effect is seed-dependent"
    )

    ax.grid(axis="y", alpha=0.25, linestyle="--")

    for i, row in enumerate(
        rgcn_pair.itertuples()
    ):
        vertical_offset = (
            0.004
            if row.delta_mrr >= 0
            else -0.006
        )

        ax.text(
            i,
            row.delta_mrr + vertical_offset,
            f"{row.delta_mrr:+.3f}",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )

    # Standardise panel tick-label size
    for ax in axes.flat:
        ax.tick_params(labelsize=9)

    fig.suptitle(
        (
            "Strict-cold predictive performance and "
            "molecular-feature availability"
        ),
        fontweight="bold",
        y=1.01,
    )

    fig.text(
        0.5,
        0.008,
        (
            "Molecular fingerprints did not uniformly improve whole-test-set "
            "performance. Within molecular runs, the subset of drugs with "
            "usable SMILES showed higher ranking performance than the subset "
            "without molecular features."
        ),
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(
        rect=[0, 0.045, 1, 1]
    )

    save_figure(
        fig,
        "fig02_strict_cold_predictive_performance",
    )


def build_figure_03():

    triaged = summary_csv(
        "triaged_unlabelled_candidates.csv"
    )

    phenotype = summary_csv(
        "phenotype_propagation_candidates.csv"
    )

    shortlist = summary_csv(
        "final_external_validation_evidence_table.csv"
    )

    supported = summary_csv(
        "manuscript_supported_candidate_case_table.csv"
    )

    # Unique candidate drugs represented in phenotype-context candidates
    drug_level_evaluations = (
        phenotype["drug_name"]
        .dropna()
        .nunique()
    )

    counts = [
        len(triaged),
        len(phenotype),
        drug_level_evaluations,
        len(shortlist),
        len(supported),
    ]

    stages = [
        "Eligible unlabelled pairs",
        "Phenotype-context candidates",
        "Unique candidate drugs",
        "External-validation shortlist",
        "Literature-supported candidates",
    ]

    fig, ax = plt.subplots(
        figsize=(9.5, 5.3)
    )

    y = np.arange(len(stages))

    bars = ax.barh(
        y,
        counts,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(stages)

    ax.invert_yaxis()

    ax.set_xlabel("Number of candidates / evaluations")

    ax.set_title(
        "Candidate prioritisation pipeline"
    )

    ax.grid(
        axis="x",
        alpha=0.25,
        linestyle="--",
    )

    max_count = max(counts)

    for bar, n in zip(bars, counts):

        retained = (
            100 * n / counts[0]
        )

        ax.text(
            bar.get_width()
            + max_count * 0.012,
            bar.get_y()
            + bar.get_height() / 2,
            f"{n:,}  ({retained:.1f}%)",
            va="center",
            fontweight="bold",
        )

    ax.set_xlim(
        0,
        max_count * 1.25,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "fig03_candidate_prioritisation",
    )


# ============================================================
# FIGURE 4
# CROSS-MODEL INTERPRETABILITY
# ============================================================

def build_figure_04():

    df = summary_csv(
        "model_interpretability_comparison.csv"
    )

    focus = [
        "phenotype_variant_clinical_associated",
        "drug_variant_clinical_associated",
        "drug_gene_clinical_associated",
        "drug_gene_pathway_associated",
        "drug_gene_label_associated",
        "drug_gene_guideline_ambiguous",
        "gene_phenotype_clinical_associated",
    ]

    df = df[
        df["relation_family"].isin(focus)
    ].copy()

    pivot = df.pivot_table(
        index="relation_family",
        columns="model",
        values="relation_rank_within_model",
        aggfunc="first",
    )

    for model in ["GCN", "RGCN", "RGAT"]:
        if model not in pivot.columns:
            pivot[model] = np.nan

    pivot = pivot[
        ["GCN", "RGCN", "RGAT"]
    ]

    order = [
        x
        for x in focus
        if x in pivot.index
    ]

    pivot = pivot.loc[order]

    data = pivot.values.astype(float)

    fig, ax = plt.subplots(
        figsize=(8.4, 5.8)
    )

    masked = np.ma.masked_invalid(data)

    im = ax.imshow(
        masked,
        aspect="auto",
        cmap="viridis_r",
        vmin=1,
        vmax=max(30, int(np.nanmax(data))),
    )

    ax.set_xticks(
        np.arange(3)
    )

    ax.set_xticklabels(
        ["GCN", "RGCN", "RGAT"],
        fontweight="bold",
    )

    ax.set_yticks(
        np.arange(len(order))
    )

    ax.set_yticklabels(
        [
            pretty_relation(x)
            for x in order
        ]
    )

    ax.set_title(
        "Within-model rank of selected relation families"
    )

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):

            value = data[i, j]

            if np.isnan(value):
                label = ""
            else:
                label = str(int(value))

            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontweight="bold",
                color=(
                    "black"
                    if (
                        not np.isnan(value)
                        and value < 17
                    )
                    else "white"
                ),
            )

    cbar = fig.colorbar(
        im,
        ax=ax,
        pad=0.02,
    )

    cbar.set_label(
        "Within-model rank (1 = strongest)"
    )

    if (
        "drug_gene_pathway_associated"
        in order
    ):

        row = order.index(
            "drug_gene_pathway_associated"
        )

        ax.annotate(
            (
                "Drug-gene pathway: "
                "RGCN rank 2, GCN rank 29, RGAT rank 26"
            ),
            xy=(1, row),
            xycoords="data",
            xytext=(0.50, 1.07),
            textcoords="axes fraction",
            ha="center",
            va="bottom",
            arrowprops=dict(
                arrowstyle="->",
                connectionstyle="arc3,rad=0.10",
            ),
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor="gray",
            ),
            fontsize=8.5,
        )

    fig.subplots_adjust(
        left=0.35,
        right=0.88,
        top=0.84,
    )

    save_figure(
        fig,
        "fig04_cross_model_relation_ranks",
    )


# ============================================================
# FIGURE 5
# ABLATION CONTROLS
# ============================================================

def build_figure_05():

    rgcn = summary_csv(
        "rgcn_relation_ablation_random_control_summary.csv"
    )

    gcn = summary_csv(
        "gcn_relation_ablation_random_control_summary.csv"
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5.6),
    )

    for ax, df, model in [
        (axes[0], rgcn, "RGCN"),
        (axes[1], gcn, "GCN"),
    ]:

        work = df.sort_values(
            "mean_excess_drop_over_random",
            ascending=True,
        ).copy()

        y = np.arange(
            len(work)
        )

        values = (
            work[
                "mean_excess_drop_over_random"
            ]
        )

        ax.barh(
            y,
            values,
            color=MODEL_COLORS[model],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.4,
        )

        ax.set_yticks(y)

        ax.set_yticklabels(
            [
                pretty_relation(x)
                for x in work[
                    "relation_family"
                ]
            ]
        )

        ax.axvline(
            0,
            linewidth=0.8,
        )

        ax.set_xlabel(
            (
                "Mean excess score drop over\n"
                "size-matched random deletion"
            )
        )

        ax.set_title(model)

        ax.grid(
            axis="x",
            alpha=0.25,
            linestyle="--",
        )

        for i, value in enumerate(values):

            ax.text(
                value + 0.04,
                i,
                f"{value:.2f}",
                va="center",
                fontsize=8.5,
            )

    fig.suptitle(
        "Relation-family ablation specificity controls",
        fontweight="bold",
    )

    fig.text(
        0.5,
        0.015,
        (
            "Ablation measures model sensitivity to relation-family removal. "
            "It does not establish biological causality. "
            "Score-drop magnitudes are interpreted within model and are not "
            "directly comparable across architectures."
        ),
        ha="center",
        fontsize=9,
    )

    fig.subplots_adjust(
        left=0.28,
        right=0.98,
        bottom=0.20,
        top=0.86,
        wspace=0.55,
    )

    save_figure(
        fig,
        "fig05_ablation_random_controls",
    )


# ============================================================
# FIGURE 6
# SUPPORTED CANDIDATES
# ============================================================

def build_figure_06():

    df = summary_csv(
        "manuscript_supported_candidate_case_table.csv"
    )

    evidence_order = [
        "direct_supported",
        "direct_mixed",
        "indirect_support",
    ]

    xmap = {
        name: i
        for i, name
        in enumerate(evidence_order)
    }

    work = df.copy()

    work["pair"] = (
        work["drug_name"].astype(str)
        + " - "
        + work["gene_name"].astype(str)
    )

    work = work.sort_values(
        "external_validation_rank"
    ).reset_index(drop=True)

    fig, ax = plt.subplots(
        figsize=(9.4, 6.3)
    )

    for i, row in work.iterrows():

        tier = row[
            "external_support_tier"
        ]

        if tier not in xmap:
            continue

        marker = RELATION_MARKERS.get(
            row[
                "rgcn_top_relation_family"
            ],
            "o",
        )

        color = EVIDENCE_COLORS.get(
            tier,
            "#999999",
        )

        ax.scatter(
            xmap[tier],
            i,
            s=150,
            marker=marker,
            color=color,
            edgecolor="black",
            linewidth=0.8,
        )

        n_seed = int(
            row[
                "strict_cold_eligible_seed_count"
            ]
        )

        ax.text(
            xmap[tier] + 0.08,
            i,
            (
                f"{n_seed} eligible "
                f"seed{'s' if n_seed != 1 else ''}"
            ),
            va="center",
            fontsize=8,
        )

    ax.set_yticks(
        np.arange(len(work))
    )

    ax.set_yticklabels(
        work["pair"]
    )

    ax.set_xticks(
        [0, 1, 2]
    )

    ax.set_xticklabels(
        [
            "Direct support",
            "Mixed direct evidence",
            "Indirect support",
        ]
    )

    ax.set_xlim(
        -0.35,
        2.8,
    )

    ax.set_xlabel(
        "External literature evidence class"
    )

    ax.set_title(
        (
            "Candidate drug-gene hypotheses with "
            "positive literature support"
        )
    )

    handles = []

    for relation, marker in RELATION_MARKERS.items():

        handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                linestyle="None",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=8,
                label=pretty_relation(
                    relation
                ),
            )
        )

    relation_legend = ax.legend(
        handles=handles,
        title="Dominant RGCN relation family",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=8.5,
        title_fontsize=9,
        borderaxespad=0.0,
        handletextpad=0.6,
        labelspacing=0.5,
    )

    ax.add_artist(relation_legend)

    evidence_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=EVIDENCE_COLORS["direct_supported"],
            markeredgecolor="black",
            markersize=8,
            label="Direct support",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=EVIDENCE_COLORS["direct_mixed"],
            markeredgecolor="black",
            markersize=8,
            label="Mixed direct evidence",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=EVIDENCE_COLORS["indirect_support"],
            markeredgecolor="black",
            markersize=8,
            label="Indirect support",
        ),
    ]

    ax.legend(
        handles=evidence_handles,
        title="Literature evidence class",
        loc="upper left",
        bbox_to_anchor=(1.01, 0.50),
        frameon=False,
        fontsize=8.5,
        title_fontsize=9,
        borderaxespad=0.0,
        handletextpad=0.6,
        labelspacing=0.5,
    )

    ax.grid(
        axis="x",
        alpha=0.2,
    )

    fig.subplots_adjust(
        left=0.28,
        right=0.76,
    )

    save_figure(
        fig,
        "fig06_supported_candidate_map",
    )


# ============================================================
# FIGURE 7
# EXTERNAL VALIDATION
# ============================================================

def build_figure_07():

    topk = summary_csv(
        "external_validation_topk_enrichment.csv"
    )

    sensitivity = summary_csv(
        "external_validation_sensitivity_summary.csv"
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5.1),
    )

    # --------------------------------------------------------
    # Panel A
    # --------------------------------------------------------

    ax = axes[0]

    ax.plot(
        topk["top_k"],
        topk["positive_support_rate"] * 100,
        marker="o",
        linewidth=2,
    )

    baseline = (
        topk[
            "overall_positive_support_rate"
        ].iloc[0]
        * 100
    )

    ax.axhline(
        baseline,
        linestyle="--",
        linewidth=1,
        label=f"Overall shortlist: {baseline:.0f}%",
    )

    ax.set_xticks(
        topk["top_k"]
    )

    ax.set_xlabel(
        "Top K shortlisted candidates"
    )

    ax.set_ylabel(
        "Positive literature support (%)"
    )

    ax.set_ylim(29, 42)

    ax.set_title(
        "A. Literature support enrichment"
    )

    ax.grid(
        alpha=0.25,
        linestyle="--",
    )

    ax.legend(
        frameon=False
    )

    for row in topk.itertuples():

        ax.text(
            row.top_k,
            row.positive_support_rate
            * 100
            + 1.5,
            f"{row.positive_support_rate * 100:.0f}%",
            ha="center",
            fontsize=8,
        )

    # --------------------------------------------------------
    # Panel B
    # --------------------------------------------------------

    ax = axes[1]

    sens = sensitivity[
        sensitivity["metric"]
        == "weighted_rrf_score"
    ].copy()

    sens = sens.sort_values(
        "direction_adjusted_mean_difference",
        ascending=True,
    )

    labels = (
        sens[
            "support_definition"
        ]
        .astype(str)
        .str.replace(
            "_",
            " ",
            regex=False,
        )
        .str.title()
    )

    y = np.arange(
        len(sens)
    )

    ax.barh(
        y,
        sens[
            "direction_adjusted_mean_difference"
        ],
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)

    ax.axvline(
        0,
        linewidth=0.8,
    )

    ax.set_xlabel(
        (
            "Direction-adjusted mean difference\n"
            "in weighted RRF score"
        )
    )

    ax.set_title(
        "B. Sensitivity of enrichment to support definition"
    )

    ax.grid(
        axis="x",
        alpha=0.25,
        linestyle="--",
    )

    for i, row in enumerate(
        sens.itertuples()
    ):

        p = (
            row.one_sided_permutation_pvalue
        )

        star = (
            "*"
            if p <= 0.05
            else ""
        )

        value = (
            row.direction_adjusted_mean_difference
        )

        ax.text(
            value,
            i,
            f"  p={p:.3f}{star}",
            va="center",
            fontsize=8.5,
        )

    ax.text(
        0.98,
        0.02,
        "* exploratory p ≤ 0.05",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )

    fig.suptitle(
        (
            "External validation enrichment "
            "and evidence-definition sensitivity"
        ),
        fontweight="bold",
    )

    fig.tight_layout()

    save_figure(
        fig,
        "fig07_external_validation",
    )


# ============================================================
# SUPPLEMENTARY FIGURE S1
# RGAT INSTABILITY
# ============================================================

def build_figure_s1():

    df = summary_csv(
        "rgat_relation_family_ablation_by_pair_seed.csv"
    )

    x = (
        df[
            "absolute_score_change"
        ]
        .astype(float)
    )

    y = (
        df[
            "absolute_rank_change"
        ]
        .astype(float)
    )

    mask = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0)
        & (y >= 0)
    )

    x = x[mask]
    y = y[mask]

    fig, ax = plt.subplots(
        figsize=(8, 5.4)
    )

    ax.scatter(
        x,
        y + 1,
        alpha=0.35,
        s=18,
        color=MODEL_COLORS["RGAT"],
    )

    ax.set_xscale(
        "symlog",
        linthresh=1e-5,
    )

    ax.set_yscale(
        "log"
    )

    ax.set_xlabel(
        "Absolute candidate-score change"
    )

    ax.set_ylabel(
        "Absolute rank change + 1"
    )

    ax.set_title(
        (
            "RGAT: small score perturbations can "
            "produce large rank movements"
        )
    )

    ax.grid(
        alpha=0.2,
    )

    ax.text(
        0.02,
        0.97,
        (
            "Near-tied scores make rank-only explanations unstable.\n"
            "RGAT is therefore retained as a descriptive comparator."
        ),
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="gray",
        ),
    )

    fig.tight_layout()

    save_figure(
        fig,
        "figS01_rgat_instability",
        supplementary=True,
    )


# ============================================================
# SUPPLEMENTARY FIGURE S2
# PATHWAY CONTEXT
# ============================================================

def build_figure_s2():

    kegg = summary_csv(
        "kegg_gene_pathway_results.csv"
    )

    reactome = summary_csv(
        "reactome_gene_pathway_results.csv"
    )

    pairs = summary_csv(
        "kegg_pair_pathway_validation_summary.csv"
    )

    candidate_genes = sorted(
        set(
            kegg[
                "gene_name"
            ]
            .dropna()
            .astype(str)
        )
        |
        set(
            reactome[
                "gene_name"
            ]
            .dropna()
            .astype(str)
        )
    )

    kegg_mapped = set(
        kegg.loc[
            kegg[
                "kegg_pathway_id"
            ].notna(),
            "gene_name",
        ].astype(str)
    )

    reactome_mapped = set(
        reactome.loc[
            reactome[
                "reactome_pathway_id"
            ].notna(),
            "gene_name",
        ].astype(str)
    )

    biological_drug_pathways = int(
        (
            pairs[
                "kegg_biological_drug_pathway_count"
            ]
            > 0
        ).sum()
    )

    exact_overlaps = int(
        pairs[
            "has_kegg_drug_gene_pathway_overlap"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.5, 4.6),
    )

    # Panel A
    ax = axes[0]

    mapping_counts = [
        len(kegg_mapped),
        len(reactome_mapped),
    ]

    bars = ax.bar(
        ["KEGG", "Reactome"],
        mapping_counts,
    )

    ax.set_ylabel(
        "Candidate genes with pathway mapping"
    )

    ax.set_title(
        "A. Gene-level pathway coverage"
    )

    ax.set_ylim(
        0,
        max(
            len(candidate_genes) + 1.5,
            16.5,
        ),
    )

    for bar, n in zip(
        bars,
        mapping_counts,
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            n + 0.25,
            f"{n}/{len(candidate_genes)}",
            ha="center",
            fontweight="bold",
        )

    # Panel B
    ax = axes[1]

    values = [
        len(pairs),
        biological_drug_pathways,
        exact_overlaps,
    ]

    labels = [
        "Shortlisted\npairs",
        "Biological KEGG\ndrug pathway",
        "Exact drug-gene\npathway overlap",
    ]

    bars = ax.bar(
        labels,
        values,
    )

    ax.set_ylabel(
        "Number of pairs"
    )

    ax.set_title(
        "B. Pair-level KEGG context"
    )

    ax.set_ylim(0, 33)

    for bar, n in zip(
        bars,
        values,
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            n + 0.4,
            str(n),
            ha="center",
            fontweight="bold",
        )

    fig.suptitle(
        (
            "Pathway analysis provides biological context, "
            "not direct pair validation"
        ),
        fontweight="bold",
    )

    fig.tight_layout()

    save_figure(
        fig,
        "figS02_pathway_context",
        supplementary=True,
    )


# ============================================================
# RUN
# ============================================================

def main():

    print("=" * 72)
    print("BUILDING FINAL DISSERTATION FIGURES")
    print("=" * 72)

    build_figure_01()
    build_figure_02()
    build_figure_03()
    build_figure_04()
    build_figure_05()
    build_figure_06()
    build_figure_07()

    build_figure_s1()
    build_figure_s2()

    print()
    print("=" * 72)
    print("FIGURE BUILD COMPLETE")
    print("=" * 72)

    print("\nMain figures:")

    for path in sorted(
        FIG_MAIN.glob("*.png")
    ):
        print(
            " ",
            path.relative_to(BASE),
        )

    print("\nSupplementary figures:")

    for path in sorted(
        FIG_SUPP.glob("*.png")
    ):
        print(
            " ",
            path.relative_to(BASE),
        )


if __name__ == "__main__":
    main()
