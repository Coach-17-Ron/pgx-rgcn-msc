#!/usr/bin/env python
"""Combine external validation and interpretability for shortlisted pairs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(
    "results/consensus_inference/molecular/"
    "strict_cold/summary"
)

EXTERNAL_PATH = (
    ROOT / "final_external_validation_evidence_table.csv"
)

COVERAGE_PATH = (
    ROOT / "interpretability_candidate_seed_coverage.csv"
)

RGCN_PATH = (
    ROOT / "rgcn_relation_family_ablation_pair_summary.csv"
)

GCN_PATH = (
    ROOT / "gcn_relation_family_ablation_pair_summary.csv"
)

RGAT_ATTENTION_PATH = (
    ROOT / "rgat_candidate_attention_relation_summary.csv"
)

OUTPUT_PATH = (
    ROOT / "integrated_candidate_evidence_table.csv"
)

MANUSCRIPT_PATH = (
    ROOT / "manuscript_integrated_candidate_evidence_table.csv"
)


def select_top_ablation(
    path: Path,
    model_prefix: str,
) -> pd.DataFrame:
    table = pd.read_csv(
        path,
        low_memory=False,
    )

    top = table.loc[
        table["relation_rank_within_pair"].eq(1)
    ].copy()

    keep = [
        "external_validation_rank",
        "relation_family",
        "mean_score_drop",
        "mean_rank_worsening",
        "supportive_score_seed_fraction",
        "supportive_rank_seed_fraction",
        "eligible_seed_count",
    ]

    top = top[keep]

    top = top.rename(
        columns={
            "relation_family":
                f"{model_prefix}_top_relation_family",
            "mean_score_drop":
                f"{model_prefix}_top_relation_mean_score_drop",
            "mean_rank_worsening":
                f"{model_prefix}_top_relation_mean_rank_worsening",
            "supportive_score_seed_fraction":
                f"{model_prefix}_supportive_score_seed_fraction",
            "supportive_rank_seed_fraction":
                f"{model_prefix}_supportive_rank_seed_fraction",
            "eligible_seed_count":
                f"{model_prefix}_eligible_seed_count",
        }
    )

    return top


def select_rgat_gene_attention() -> pd.DataFrame:
    table = pd.read_csv(
        RGAT_ATTENTION_PATH,
        low_memory=False,
    )

    top = table.loc[
        table["layer"].eq(2)
        & table["candidate_endpoint"].eq(
            "gene_incoming"
        )
        & table[
            "relation_rank_within_pair_layer_endpoint"
        ].eq(1)
    ].copy()

    keep = [
        "external_validation_rank",
        "relation_type",
        "mean_endpoint_attention_fraction",
        "maximum_attention",
        "eligible_seed_count",
    ]

    top = top[keep].rename(
        columns={
            "relation_type":
                "rgat_top_gene_incoming_relation",
            "mean_endpoint_attention_fraction":
                "rgat_mean_gene_attention_fraction",
            "maximum_attention":
                "rgat_maximum_gene_attention",
            "eligible_seed_count":
                "rgat_eligible_seed_count",
        }
    )

    return top


def main() -> None:
    external = pd.read_csv(
        EXTERNAL_PATH,
        low_memory=False,
    )

    coverage = pd.read_csv(
        COVERAGE_PATH,
        low_memory=False,
    )

    eligible_coverage = (
        coverage.loc[
            coverage[
                "eligible_for_strict_cold_explanation"
            ].fillna(False)
        ]
        .groupby(
            "external_validation_rank",
            as_index=False,
        )
        .agg(
            strict_cold_eligible_seed_count=(
                "seed",
                "nunique",
            ),
            strict_cold_eligible_seeds=(
                "seed",
                lambda values: "|".join(
                    str(value)
                    for value in sorted(
                        set(values)
                    )
                ),
            ),
        )
    )

    rgcn = select_top_ablation(
        RGCN_PATH,
        "rgcn",
    )

    gcn = select_top_ablation(
        GCN_PATH,
        "gcn",
    )

    rgat = select_rgat_gene_attention()

    integrated = (
        external.merge(
            eligible_coverage,
            on="external_validation_rank",
            how="left",
            validate="one_to_one",
        )
        .merge(
            rgcn,
            on="external_validation_rank",
            how="left",
            validate="one_to_one",
        )
        .merge(
            gcn,
            on="external_validation_rank",
            how="left",
            validate="one_to_one",
        )
        .merge(
            rgat,
            on="external_validation_rank",
            how="left",
            validate="one_to_one",
        )
    )

    integrated["external_support_tier"] = "none_found"

    integrated.loc[
        integrated[
            "pubmed_pair_evidence_class"
        ].eq("direct_supported"),
        "external_support_tier",
    ] = "direct_supported"

    integrated.loc[
        integrated[
            "pubmed_pair_evidence_class"
        ].eq("direct_mixed_evidence"),
        "external_support_tier",
    ] = "direct_mixed"

    integrated.loc[
        integrated[
            "pubmed_pair_evidence_class"
        ].eq("indirect_mechanistic_support"),
        "external_support_tier",
    ] = "indirect_support"

    integrated.loc[
        integrated[
            "pubmed_pair_evidence_class"
        ].eq("direct_null_only"),
        "external_support_tier",
    ] = "direct_null_only"

    integrated[
        "rgcn_primary_relation_aware_signal"
    ] = (
        integrated[
            "rgcn_supportive_score_seed_fraction"
        ].fillna(0) > 0.5
    )

    integrated[
        "gcn_topology_support_signal"
    ] = (
        integrated[
            "gcn_supportive_score_seed_fraction"
        ].fillna(0) > 0.5
    )

    integrated[
        "interpretability_summary"
    ] = (
        "RGCN: "
        + integrated[
            "rgcn_top_relation_family"
        ].fillna("not_available")
        + "; GCN: "
        + integrated[
            "gcn_top_relation_family"
        ].fillna("not_available")
        + "; RGAT gene incoming: "
        + integrated[
            "rgat_top_gene_incoming_relation"
        ].fillna("not_available")
    )

    integrated = integrated.sort_values(
        "external_validation_rank"
    )

    integrated.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    manuscript_columns = [
        "external_validation_rank",
        "drug_name",
        "gene_name",
        "pubmed_pair_evidence_class",
        "external_support_tier",
        "has_positive_pubmed_support",
        "strict_cold_eligible_seed_count",
        "rgcn_top_relation_family",
        "rgcn_supportive_score_seed_fraction",
        "gcn_top_relation_family",
        "gcn_supportive_score_seed_fraction",
        "rgat_top_gene_incoming_relation",
        "rgat_mean_gene_attention_fraction",
        "interpretability_summary",
    ]

    manuscript = integrated[
        [
            column
            for column in manuscript_columns
            if column in integrated.columns
        ]
    ].copy()

    manuscript.to_csv(
        MANUSCRIPT_PATH,
        index=False,
    )

    print(
        "INTEGRATED CANDIDATE EVIDENCE TABLE COMPLETE"
    )
    print("Pairs:", len(integrated))

    print("\nEXTERNAL SUPPORT DISTRIBUTION")
    print(
        integrated[
            "external_support_tier"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nDIRECT/INDIRECT SUPPORTED PAIRS WITH "
        "RGCN EXPLANATIONS"
    )

    supported = integrated.loc[
        integrated[
            "external_support_tier"
        ].isin(
            [
                "direct_supported",
                "direct_mixed",
                "indirect_support",
            ]
        ),
        [
            "external_validation_rank",
            "drug_name",
            "gene_name",
            "external_support_tier",
            "rgcn_top_relation_family",
            "rgcn_supportive_score_seed_fraction",
            "gcn_top_relation_family",
            "rgat_top_gene_incoming_relation",
        ],
    ]

    print(
        supported.to_string(index=False)
    )

    print("\nSaved:")
    print(OUTPUT_PATH)
    print(MANUSCRIPT_PATH)


if __name__ == "__main__":
    main()
