#!/usr/bin/env python
"""Rank gene-centred phenotype contexts for drug–gene candidates."""

from __future__ import annotations

from pathlib import Path
import math
import re

import numpy as np
import pandas as pd


ROOT = Path(
    "results/consensus_inference/molecular/"
    "strict_cold/summary"
)

INPUT_PATH = ROOT / "gene_centred_phenotype_context.csv"


def count_values(value: object) -> int:
    if pd.isna(value):
        return 0

    values = {
        item.strip()
        for item in re.split(r"[;,|]", str(value))
        if item.strip()
        and item.strip().lower() != "nan"
    }

    return len(values)


def association_weight(
    direct: object,
    variant: object,
) -> float:
    labels: set[str] = set()

    for value in (direct, variant):
        if pd.isna(value):
            continue

        labels.update(
            item.strip().lower()
            for item in re.split(
                r"[;,|]",
                str(value),
            )
            if item.strip()
        )

    if "associated" in labels:
        return 1.0

    if "ambiguous" in labels:
        return 0.65

    if "not_associated" in labels:
        return 0.30

    return 0.50


def primary_association(
    direct: object,
    variant: object,
) -> str:
    labels: set[str] = set()

    for value in (direct, variant):
        if pd.isna(value):
            continue

        labels.update(
            item.strip().lower()
            for item in re.split(
                r"[;,|]",
                str(value),
            )
            if item.strip()
        )

    if "associated" in labels:
        if "not_associated" in labels:
            return "conflicting"
        if "ambiguous" in labels:
            return "associated_with_ambiguity"
        return "associated"

    if "ambiguous" in labels:
        return "ambiguous"

    if "not_associated" in labels:
        return "not_associated"

    return "unknown"


def main() -> None:
    context = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    valid = context.loc[
        context["phenotype_context_type"]
        .ne("no_graph_context")
    ].copy()

    total_candidate_genes = valid[
        "gene_idx"
    ].nunique()

    phenotype_frequency = (
        valid[
            [
                "phenotype_idx",
                "phenotype_id",
                "phenotype_name",
                "gene_idx",
                "drug_idx",
                "seed",
            ]
        ]
        .drop_duplicates()
        .groupby(
            [
                "phenotype_idx",
                "phenotype_id",
                "phenotype_name",
            ],
            as_index=False,
        )
        .agg(
            candidate_gene_frequency=(
                "gene_idx",
                "nunique",
            ),
            candidate_drug_frequency=(
                "drug_idx",
                "nunique",
            ),
            candidate_evaluation_frequency=(
                "seed",
                "size",
            ),
        )
    )

    phenotype_frequency[
        "phenotype_specificity"
    ] = np.log(
        (total_candidate_genes + 1)
        / (
            phenotype_frequency[
                "candidate_gene_frequency"
            ]
            + 1
        )
    )

    valid = valid.merge(
        phenotype_frequency,
        on=[
            "phenotype_idx",
            "phenotype_id",
            "phenotype_name",
        ],
        how="left",
        validate="many_to_one",
    )

    context_weights = {
        "direct_and_host_variant": 1.00,
        "direct_gene_phenotype": 0.90,
        "host_variant_phenotype": 0.80,
    }

    valid["context_weight"] = (
        valid["phenotype_context_type"]
        .map(context_weights)
        .fillna(0.50)
    )

    valid["primary_association"] = valid.apply(
        lambda row: primary_association(
            row.get("direct_associations"),
            row.get("variant_associations"),
        ),
        axis=1,
    )

    valid["association_weight"] = valid.apply(
        lambda row: association_weight(
            row.get("direct_associations"),
            row.get("variant_associations"),
        ),
        axis=1,
    )

    valid["direct_pmids_count"] = (
        valid["direct_pmids"].map(count_values)
    )

    valid["variant_pmids_count"] = (
        valid["variant_pmids"].map(count_values)
    )

    valid["total_unique_pmids"] = valid.apply(
        lambda row: count_values(
            ";".join(
                [
                    str(row.get("direct_pmids", "")),
                    str(row.get("variant_pmids", "")),
                ]
            )
        ),
        axis=1,
    )

    valid["literature_weight"] = (
        1.0
        + np.log1p(
            valid["total_unique_pmids"]
        )
    )

    valid["drug_gene_strength"] = (
        1.0
        / valid["weighted_consensus_rank"]
        .clip(lower=1)
    )

    valid["phenotype_priority_score"] = (
        valid["drug_gene_strength"]
        * valid["phenotype_specificity"]
        * valid["context_weight"]
        * valid["association_weight"]
        * valid["literature_weight"]
    )

    pair_columns = [
        "seed",
        "drug_idx",
        "gene_idx",
    ]

    valid["within_pair_phenotype_rank"] = (
        valid.groupby(pair_columns)[
            "phenotype_priority_score"
        ]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    valid["within_drug_phenotype_rank"] = (
        valid.groupby(
            ["seed", "drug_idx"]
        )["phenotype_priority_score"]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    positive = valid.loc[
        valid["primary_association"].isin(
            [
                "associated",
                "associated_with_ambiguity",
                "ambiguous",
                "conflicting",
            ]
        )
    ].copy()

    negative = valid.loc[
        valid["primary_association"].eq(
            "not_associated"
        )
    ].copy()

    top_per_pair = positive.loc[
        positive[
            "within_pair_phenotype_rank"
        ].le(5)
    ].sort_values(
        [
            "seed",
            "drug_name",
            "gene_name",
            "within_pair_phenotype_rank",
        ]
    )

    top_per_drug = positive.loc[
        positive[
            "within_drug_phenotype_rank"
        ].le(10)
    ].sort_values(
        [
            "seed",
            "drug_name",
            "within_drug_phenotype_rank",
        ]
    )

    ranked_path = (
        ROOT / "ranked_gene_centred_phenotypes.csv"
    )

    pair_path = (
        ROOT / "top5_phenotypes_per_candidate_pair.csv"
    )

    drug_path = (
        ROOT / "top10_phenotypes_per_drug.csv"
    )

    negative_path = (
        ROOT / "not_associated_phenotype_context.csv"
    )

    prevalence_path = (
        ROOT / "phenotype_candidate_prevalence.csv"
    )

    valid.sort_values(
        "phenotype_priority_score",
        ascending=False,
    ).to_csv(
        ranked_path,
        index=False,
    )

    top_per_pair.to_csv(
        pair_path,
        index=False,
    )

    top_per_drug.to_csv(
        drug_path,
        index=False,
    )

    negative.to_csv(
        negative_path,
        index=False,
    )

    phenotype_frequency.sort_values(
        [
            "candidate_gene_frequency",
            "candidate_drug_frequency",
        ],
        ascending=False,
    ).to_csv(
        prevalence_path,
        index=False,
    )

    print("PHENOTYPE RANKING COMPLETE")
    print("Ranked positive/context rows:", len(positive))
    print(
        "Top-five pair rows:",
        len(top_per_pair),
    )
    print(
        "Candidate pairs represented:",
        top_per_pair[pair_columns]
        .drop_duplicates()
        .shape[0],
    )
    print(
        "Top-ten drug rows:",
        len(top_per_drug),
    )
    print(
        "Drug evaluations represented:",
        top_per_drug[
            ["seed", "drug_idx"]
        ]
        .drop_duplicates()
        .shape[0],
    )
    print(
        "Not-associated context rows:",
        len(negative),
    )

    print("\nPRIMARY ASSOCIATION COUNTS")
    print(
        valid["primary_association"]
        .value_counts()
        .to_string()
    )

    print("\nTOP RANKED PHENOTYPES")
    print(
        top_per_drug[
            [
                "seed",
                "drug_name",
                "gene_name",
                "phenotype_name",
                "weighted_consensus_rank",
                "phenotype_context_type",
                "primary_association",
                "candidate_gene_frequency",
                "total_unique_pmids",
                "phenotype_priority_score",
                "within_drug_phenotype_rank",
            ]
        ]
        .head(50)
        .to_string(index=False)
    )

    print("\nSaved:")
    print(ranked_path)
    print(pair_path)
    print(drug_path)
    print(negative_path)
    print(prevalence_path)


if __name__ == "__main__":
    main()
