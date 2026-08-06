#!/usr/bin/env python
"""Collapse host-confirmed variant–phenotype paths into canonical records."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


INPUT_PATH = Path(
    "results/consensus_inference/molecular/strict_cold/"
    "summary/host_confirmed_variant_phenotype_paths.csv"
)

OUTPUT_DIR = Path(
    "results/consensus_inference/molecular/strict_cold/summary"
)


def combine_values(values: pd.Series) -> str:
    collected: set[str] = set()

    for value in values.dropna():
        for item in re.split(r"[;,|]", str(value)):
            item = item.strip()
            if item and item.lower() != "nan":
                collected.add(item)

    return ";".join(sorted(collected))


def main() -> None:
    paths = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    key_columns = [
        "seed",
        "drug_idx",
        "drug_id",
        "drug_name",
        "gene_idx",
        "gene_id",
        "gene_name",
        "variant_idx",
        "variant_id",
        "variant_name",
        "variant_location",
        "phenotype_idx",
        "phenotype_id",
        "phenotype_name",
    ]

    prediction_columns = [
        column
        for column in [
            "weighted_consensus_rank",
            "weighted_rrf_score",
            "candidate_class",
            "within_drug_priority_rank",
            "specificity_priority_score",
            "models_top_25",
            "gcn_rank",
            "rgcn_rank",
            "rgat_rank",
        ]
        if column in paths.columns
    ]

    aggregation: dict[str, object] = {
        column: "first"
        for column in prediction_columns
    }

    if "relation_type" in paths.columns:
        aggregation["relation_type"] = combine_values

    if "evidence" in paths.columns:
        aggregation["evidence"] = combine_values

    if "association" in paths.columns:
        aggregation["association"] = combine_values

    if "pmids" in paths.columns:
        aggregation["pmids"] = combine_values

    canonical = (
        paths.groupby(
            key_columns,
            dropna=False,
            as_index=False,
        )
        .agg(aggregation)
    )

    canonical["supporting_pmids_count"] = (
        canonical.get(
            "pmids",
            pd.Series("", index=canonical.index),
        )
        .fillna("")
        .map(
            lambda value: len(
                [
                    item
                    for item in str(value).split(";")
                    if item
                ]
            )
        )
    )

    pair_keys = [
        "seed",
        "drug_idx",
        "gene_idx",
    ]

    phenotype_summary = (
        canonical.groupby(
            pair_keys
            + [
                "drug_id",
                "drug_name",
                "gene_id",
                "gene_name",
            ],
            as_index=False,
        )
        .agg(
            verified_host_variants=(
                "variant_idx",
                "nunique",
            ),
            variant_supported_phenotypes=(
                "phenotype_idx",
                "nunique",
            ),
            total_variant_phenotype_records=(
                "phenotype_idx",
                "size",
            ),
            total_supporting_pmids=(
                "supporting_pmids_count",
                "sum",
            ),
        )
    )

    candidate_gene_summary = (
        canonical.groupby(
            ["gene_idx", "gene_id", "gene_name"],
            as_index=False,
        )
        .agg(
            unique_candidate_pairs=(
                "drug_idx",
                "nunique",
            ),
            candidate_evaluations=(
                "seed",
                "size",
            ),
            verified_host_variants=(
                "variant_idx",
                "nunique",
            ),
            variant_supported_phenotypes=(
                "phenotype_idx",
                "nunique",
            ),
        )
    )

    # Correct candidate-evaluation count:
    candidate_evaluations = (
        canonical[
            ["seed", "drug_idx", "gene_idx"]
        ]
        .drop_duplicates()
        .groupby("gene_idx")
        .size()
        .rename("candidate_evaluations")
        .reset_index()
    )

    unique_drugs = (
        canonical[
            ["drug_idx", "gene_idx"]
        ]
        .drop_duplicates()
        .groupby("gene_idx")
        .size()
        .rename("unique_candidate_drugs")
        .reset_index()
    )

    candidate_gene_summary = (
        candidate_gene_summary
        .drop(columns=["candidate_evaluations"])
        .merge(
            candidate_evaluations,
            on="gene_idx",
            how="left",
        )
        .merge(
            unique_drugs,
            on="gene_idx",
            how="left",
        )
        .sort_values(
            [
                "candidate_evaluations",
                "verified_host_variants",
            ],
            ascending=False,
        )
    )

    canonical_path = (
        OUTPUT_DIR
        / "canonical_host_variant_phenotype_paths.csv"
    )

    pair_summary_path = (
        OUTPUT_DIR
        / "candidate_variant_phenotype_summary.csv"
    )

    gene_summary_path = (
        OUTPUT_DIR
        / "host_variant_gene_summary.csv"
    )

    canonical.to_csv(
        canonical_path,
        index=False,
    )

    phenotype_summary.to_csv(
        pair_summary_path,
        index=False,
    )

    candidate_gene_summary.to_csv(
        gene_summary_path,
        index=False,
    )

    print("CANONICAL HOST-VARIANT PHENOTYPE SUMMARY")
    print("Raw path rows:", len(paths))
    print("Canonical path records:", len(canonical))
    print(
        "Candidate evaluations with support:",
        canonical[pair_keys]
        .drop_duplicates()
        .shape[0],
    )
    print(
        "Unique candidate drugs:",
        canonical["drug_idx"].nunique(),
    )
    print(
        "Candidate genes with support:",
        canonical["gene_idx"].nunique(),
    )
    print(
        "Verified host variants:",
        canonical["variant_idx"].nunique(),
    )
    print(
        "Supported phenotypes:",
        canonical["phenotype_idx"].nunique(),
    )

    print("\nCORRECTED GENE SUMMARY")
    print(
        candidate_gene_summary.head(30).to_string(
            index=False
        )
    )

    print("\nCANDIDATE-PAIR SUPPORT DISTRIBUTION")
    print(
        phenotype_summary[
            [
                "verified_host_variants",
                "variant_supported_phenotypes",
            ]
        ]
        .describe()
        .to_string()
    )

    print("\nSaved:")
    print(canonical_path)
    print(pair_summary_path)
    print(gene_summary_path)


if __name__ == "__main__":
    main()
