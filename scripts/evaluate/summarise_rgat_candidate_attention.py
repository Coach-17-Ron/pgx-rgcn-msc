#!/usr/bin/env python
"""Summarise RGAT attention around candidate drugs and genes in eligible seeds."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(
    "results/consensus_inference/molecular/"
    "strict_cold/summary"
)

ATTENTION_PATH = ROOT / "rgat_candidate_incident_attention.csv"
COVERAGE_PATH = ROOT / "interpretability_candidate_seed_coverage.csv"
EVIDENCE_PATH = ROOT / "final_external_validation_evidence_table.csv"

LONG_OUTPUT = ROOT / "rgat_candidate_attention_eligible_edges.csv"
TOP_OUTPUT = ROOT / "rgat_candidate_attention_top_edges.csv"
RELATION_OUTPUT = ROOT / "rgat_candidate_attention_relation_summary.csv"


def combine_unique(values: pd.Series) -> str:
    return ";".join(
        sorted(
            {
                str(value).strip()
                for value in values.dropna()
                if str(value).strip()
                and str(value).strip().lower() != "nan"
            }
        )
    )


def main() -> None:
    attention = pd.read_csv(
        ATTENTION_PATH,
        low_memory=False,
    )

    coverage = pd.read_csv(
        COVERAGE_PATH,
        low_memory=False,
    )

    evidence = pd.read_csv(
        EVIDENCE_PATH,
        low_memory=False,
    )[
        [
            "external_validation_rank",
            "drug_name",
            "gene_name",
            "pubmed_pair_evidence_class",
            "has_positive_pubmed_support",
        ]
    ].drop_duplicates()

    eligible = coverage.loc[
        coverage[
            "eligible_for_strict_cold_explanation"
        ].fillna(False),
        [
            "seed",
            "external_validation_rank",
            "drug_idx",
            "drug_name",
            "gene_idx",
            "gene_name",
        ],
    ].drop_duplicates()

    eligible = eligible.merge(
        evidence,
        on=[
            "external_validation_rank",
            "drug_name",
            "gene_name",
        ],
        how="left",
        validate="many_to_one",
    )

    local = attention.merge(
        eligible,
        on=[
            "seed",
            "external_validation_rank",
        ],
        how="inner",
        suffixes=("", "_eligible"),
        validate="many_to_many",
    )

    # Guard against accidental candidate mismatch.
    local = local.loc[
        local["candidate_drug_name"].eq(
            local["drug_name"]
        )
        & local["candidate_gene_name"].eq(
            local["gene_name"]
        )
    ].copy()

    local["candidate_endpoint"] = "other"

    drug_incoming = local["target_idx"].eq(
        local["drug_idx"]
    )

    gene_incoming = local["target_idx"].eq(
        local["gene_idx"]
    )

    local.loc[
        drug_incoming,
        "candidate_endpoint",
    ] = "drug_incoming"

    local.loc[
        gene_incoming,
        "candidate_endpoint",
    ] = "gene_incoming"

    # Incoming attention is the cleanest interpretation because
    # RGAT normalises attention over messages received by targets.
    local = local.loc[
        local["candidate_endpoint"].isin(
            ["drug_incoming", "gene_incoming"]
        )
    ].copy()

    local[
        "attention_rank_within_candidate_endpoint"
    ] = (
        local.groupby(
            [
                "external_validation_rank",
                "seed",
                "layer",
                "candidate_endpoint",
            ]
        )["attention"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    endpoint_total = local.groupby(
        [
            "external_validation_rank",
            "seed",
            "layer",
            "candidate_endpoint",
        ]
    )["attention"].transform("sum")

    local[
        "candidate_endpoint_attention_fraction"
    ] = local["attention"] / endpoint_total

    local.to_csv(
        LONG_OUTPUT,
        index=False,
    )

    top = local.loc[
        local[
            "attention_rank_within_candidate_endpoint"
        ].le(5)
    ].copy()

    top = top.sort_values(
        [
            "external_validation_rank",
            "seed",
            "layer",
            "candidate_endpoint",
            "attention_rank_within_candidate_endpoint",
        ]
    )

    top.to_csv(
        TOP_OUTPUT,
        index=False,
    )

    relation_summary = (
        local.groupby(
            [
                "external_validation_rank",
                "drug_name",
                "gene_name",
                "pubmed_pair_evidence_class",
                "layer",
                "candidate_endpoint",
                "relation_type",
            ],
            as_index=False,
        )
        .agg(
            eligible_seed_count=(
                "seed",
                "nunique",
            ),
            edge_count=(
                "attention",
                "size",
            ),
            mean_attention=(
                "attention",
                "mean",
            ),
            maximum_attention=(
                "attention",
                "max",
            ),
            mean_endpoint_attention_fraction=(
                "candidate_endpoint_attention_fraction",
                "mean",
            ),
            source_nodes=(
                "source_name",
                combine_unique,
            ),
        )
    )

    relation_summary[
        "relation_rank_within_pair_layer_endpoint"
    ] = (
        relation_summary.groupby(
            [
                "external_validation_rank",
                "layer",
                "candidate_endpoint",
            ]
        )["mean_endpoint_attention_fraction"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    relation_summary = relation_summary.sort_values(
        [
            "external_validation_rank",
            "layer",
            "candidate_endpoint",
            "relation_rank_within_pair_layer_endpoint",
        ]
    )

    relation_summary.to_csv(
        RELATION_OUTPUT,
        index=False,
    )

    print("RGAT CANDIDATE ATTENTION SUMMARY COMPLETE")
    print(
        "Eligible pair-seed combinations:",
        eligible[
            ["seed", "external_validation_rank"]
        ].drop_duplicates().shape[0],
    )
    print(
        "Incoming attention rows:",
        len(local),
    )
    print(
        "Candidate pairs represented:",
        local["external_validation_rank"].nunique(),
    )

    print("\nTOP RELATION PER PAIR — LAYER 2 GENE INCOMING")
    print(
        relation_summary.loc[
            relation_summary["layer"].eq(2)
            & relation_summary[
                "candidate_endpoint"
            ].eq("gene_incoming")
            & relation_summary[
                "relation_rank_within_pair_layer_endpoint"
            ].eq(1),
            [
                "external_validation_rank",
                "drug_name",
                "gene_name",
                "relation_type",
                "mean_endpoint_attention_fraction",
                "maximum_attention",
                "eligible_seed_count",
            ],
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(LONG_OUTPUT)
    print(TOP_OUTPUT)
    print(RELATION_OUTPUT)


if __name__ == "__main__":
    main()
