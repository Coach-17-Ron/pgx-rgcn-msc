#!/usr/bin/env python
"""Select hypotheses for external pathway and literature validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(
    "results/consensus_inference/molecular/"
    "strict_cold/summary"
)

CANDIDATE_PATH = ROOT / "phenotype_propagation_candidates.csv"
PHENOTYPE_PATH = ROOT / "ranked_gene_centred_phenotypes.csv"
REPEATED_PATH = ROOT / "repeated_drug_gene_predictions.csv"


def main() -> None:
    candidates = pd.read_csv(
        CANDIDATE_PATH,
        low_memory=False,
    )

    phenotypes = pd.read_csv(
        PHENOTYPE_PATH,
        low_memory=False,
    )

    repeated = pd.read_csv(
        REPEATED_PATH,
        low_memory=False,
    )

    # One record per seed-specific candidate pair.
    pair_keys = [
        "seed",
        "drug_idx",
        "gene_idx",
    ]

    candidate_pairs = candidates.drop_duplicates(
        pair_keys
    ).copy()

    # Cross-seed recurrence is secondary because test-drug overlap is limited.
    repeated_columns = [
        column
        for column in [
            "drug_name",
            "gene_name",
            "seeds_observed",
            "seed_list",
            "mean_weighted_rank",
            "best_weighted_rank",
            "models_top_25_mean",
        ]
        if column in repeated.columns
    ]

    repeated_pairs = repeated[
        repeated_columns
    ].drop_duplicates(
        ["drug_name", "gene_name"]
    )

    candidate_pairs = candidate_pairs.merge(
        repeated_pairs,
        on=["drug_name", "gene_name"],
        how="left",
        suffixes=("", "_repeated"),
    )

    candidate_pairs["seeds_observed"] = (
        candidate_pairs["seeds_observed"]
        .fillna(1)
        .astype(int)
    )

    # Prefer drug-specific candidates, stronger model agreement,
    # better consensus rank, and cross-seed recurrence where available.
    candidate_pairs["validation_priority_score"] = (
        candidate_pairs["specificity_priority_score"]
        * candidate_pairs["models_top_25"]
        * (
            1.0
            + 0.25
            * (
                candidate_pairs["seeds_observed"]
                - 1
            )
        )
    )

    eligible_pairs = candidate_pairs.loc[
        candidate_pairs["candidate_class"].isin(
            [
                "drug_specific_candidate",
                "recurrent_candidate",
            ]
        )
        & candidate_pairs["has_molecular_features"].eq(True)
    ].copy()

    eligible_pairs = eligible_pairs.sort_values(
        [
            "validation_priority_score",
            "weighted_consensus_rank",
            "models_top_25",
        ],
        ascending=[False, True, False],
    )

    # Avoid allowing one recurrent gene to dominate the shortlist.
    selected_rows = []
    gene_counts: dict[int, int] = {}
    drug_counts: dict[int, int] = {}

    for _, row in eligible_pairs.iterrows():
        gene_idx = int(row["gene_idx"])
        drug_idx = int(row["drug_idx"])

        if gene_counts.get(gene_idx, 0) >= 3:
            continue

        if drug_counts.get(drug_idx, 0) >= 2:
            continue

        selected_rows.append(row)

        gene_counts[gene_idx] = (
            gene_counts.get(gene_idx, 0) + 1
        )
        drug_counts[drug_idx] = (
            drug_counts.get(drug_idx, 0) + 1
        )

        if len(selected_rows) >= 30:
            break

    selected_pairs = pd.DataFrame(selected_rows)

    if selected_pairs.empty:
        raise RuntimeError(
            "No candidate pairs selected."
        )

    selected_pairs["external_validation_rank"] = range(
        1,
        len(selected_pairs) + 1,
    )

    # Keep the top three positive/ambiguous phenotype contexts per pair.
    phenotype_subset = phenotypes.loc[
        phenotypes["primary_association"].isin(
            [
                "associated",
                "associated_with_ambiguity",
                "ambiguous",
                "conflicting",
            ]
        )
        & phenotypes[
            "within_pair_phenotype_rank"
        ].le(3)
    ].copy()

    hypotheses = selected_pairs.merge(
        phenotype_subset,
        on=pair_keys,
        how="left",
        suffixes=("_candidate", "_phenotype"),
    )

    hypotheses = hypotheses.sort_values(
        [
            "external_validation_rank",
            "within_pair_phenotype_rank",
        ]
    )

    pair_output = (
        ROOT
        / "external_validation_candidate_pairs.csv"
    )

    hypothesis_output = (
        ROOT
        / "external_validation_hypotheses.csv"
    )

    selected_pairs.to_csv(
        pair_output,
        index=False,
    )

    hypotheses.to_csv(
        hypothesis_output,
        index=False,
    )

    print("EXTERNAL VALIDATION SHORTLIST COMPLETE")
    print(
        "Selected drug–gene pairs:",
        len(selected_pairs),
    )
    print(
        "Unique drugs:",
        selected_pairs["drug_idx"].nunique(),
    )
    print(
        "Unique genes:",
        selected_pairs["gene_idx"].nunique(),
    )
    print(
        "Drug–gene–phenotype hypotheses:",
        len(hypotheses),
    )

    print("\nSELECTED PAIRS")
    print(
        selected_pairs[
            [
                "external_validation_rank",
                "seed",
                "drug_name",
                "gene_name",
                "weighted_consensus_rank",
                "models_top_25",
                "candidate_class",
                "seeds_observed",
                "validation_priority_score",
            ]
        ].to_string(index=False)
    )

    print("\nTOP PHENOTYPE CONTEXTS")
    print(
        hypotheses[
            [
                "external_validation_rank",
                "drug_name_candidate",
                "gene_name_candidate",
                "phenotype_name",
                "within_pair_phenotype_rank",
                "phenotype_context_type",
                "primary_association",
                "phenotype_priority_score",
            ]
        ]
        .head(90)
        .to_string(index=False)
    )

    print("\nSaved:")
    print(pair_output)
    print(hypothesis_output)


if __name__ == "__main__":
    main()
