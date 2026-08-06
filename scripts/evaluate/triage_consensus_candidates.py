#!/usr/bin/env python
"""Triage consensus predictions before phenotype propagation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


SEEDS = (42, 123, 2026)

RESULT_ROOT = Path(
    "results/consensus_inference/molecular/strict_cold"
)
SPLIT_ROOT = Path(
    "artifacts/splits/strict_cold"
)
TENSOR_ROOT = Path(
    "artifacts/tensors_molecular/strict_cold"
)
OUTPUT_DIR = RESULT_ROOT / "summary"


def load_seed_candidates(seed: int) -> pd.DataFrame:
    candidates = pd.read_csv(
        RESULT_ROOT
        / f"seed_{seed}"
        / "top_consensus_candidates.csv"
    )

    assignments = pd.read_parquet(
        SPLIT_ROOT
        / f"seed_{seed}"
        / "drug_assignments.parquet"
    )

    bundle = torch.load(
        TENSOR_ROOT
        / f"seed_{seed}"
        / "tensor_bundle.pt",
        map_location="cpu",
        weights_only=False,
    )

    has_features = (
        bundle["has_molecular_features"]
        .bool()
        .cpu()
        .numpy()
    )

    drug_coverage = assignments.loc[
        assignments["split"].eq("test"),
        ["drug_idx"],
    ].copy()

    drug_coverage["has_molecular_features"] = (
        drug_coverage["drug_idx"]
        .map(
            lambda index: bool(
                has_features[int(index)]
            )
        )
    )

    candidates = candidates.merge(
        drug_coverage,
        on="drug_idx",
        how="left",
        validate="many_to_one",
    )

    return candidates


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    frames = [
        load_seed_candidates(seed)
        for seed in SEEDS
    ]

    candidates = pd.concat(
        frames,
        ignore_index=True,
    )

    eligible = candidates.loc[
        candidates["association"].eq("unlabelled")
        & candidates["weighted_consensus_rank"].le(25)
        & candidates["models_top_25"].ge(2)
    ].copy()

    total_evaluations = (
        eligible[["seed", "drug_idx"]]
        .drop_duplicates()
        .shape[0]
    )

    gene_frequency = (
        eligible.groupby(
            ["gene_idx", "gene_id", "gene_name"]
        )
        .agg(
            top25_drug_evaluations=(
                "drug_idx",
                "size",
            ),
            unique_drugs=(
                "drug_id",
                "nunique",
            ),
            seeds_observed=(
                "seed",
                "nunique",
            ),
        )
        .reset_index()
    )

    gene_frequency["top25_prevalence"] = (
        gene_frequency[
            "top25_drug_evaluations"
        ]
        / total_evaluations
    )

    gene_frequency["specificity_weight"] = np.log(
        (total_evaluations + 1)
        / (
            gene_frequency[
                "top25_drug_evaluations"
            ]
            + 1
        )
    )

    eligible = eligible.merge(
        gene_frequency,
        on=["gene_idx", "gene_id", "gene_name"],
        how="left",
        validate="many_to_one",
    )

    eligible["rank_strength"] = (
        1.0
        / eligible["weighted_consensus_rank"]
    )

    eligible["agreement_fraction"] = (
        eligible["models_top_25"] / 3.0
    )

    eligible["specificity_priority_score"] = (
        eligible["rank_strength"]
        * eligible["agreement_fraction"]
        * eligible["specificity_weight"]
    )

    eligible["candidate_class"] = np.select(
        [
            ~eligible[
                "has_molecular_features"
            ].fillna(False),
            eligible["top25_prevalence"].ge(0.50),
            eligible["top25_prevalence"].ge(0.20),
        ],
        [
            "structure_limited",
            "global_prior",
            "recurrent_candidate",
        ],
        default="drug_specific_candidate",
    )

    eligible["within_drug_priority_rank"] = (
        eligible.groupby(
            ["seed", "drug_idx"]
        )["specificity_priority_score"]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    phenotype_candidates = (
        eligible.loc[
            eligible["has_molecular_features"].eq(True)
            & eligible["candidate_class"].isin(
                [
                    "drug_specific_candidate",
                    "recurrent_candidate",
                ]
            )
            & eligible[
                "within_drug_priority_rank"
            ].le(5)
        ]
        .sort_values(
            [
                "seed",
                "drug_name",
                "within_drug_priority_rank",
            ]
        )
    )

    global_prior = (
        eligible.loc[
            eligible["candidate_class"].eq(
                "global_prior"
            )
        ]
        .sort_values(
            [
                "top25_prevalence",
                "weighted_consensus_rank",
            ],
            ascending=[False, True],
        )
    )

    structure_limited = (
        eligible.loc[
            eligible["candidate_class"].eq(
                "structure_limited"
            )
        ]
        .sort_values(
            [
                "seed",
                "drug_name",
                "weighted_consensus_rank",
            ]
        )
    )

    eligible.to_csv(
        OUTPUT_DIR / "triaged_unlabelled_candidates.csv",
        index=False,
    )

    phenotype_candidates.to_csv(
        OUTPUT_DIR / "phenotype_propagation_candidates.csv",
        index=False,
    )

    global_prior.to_csv(
        OUTPUT_DIR / "global_prior_candidates.csv",
        index=False,
    )

    structure_limited.to_csv(
        OUTPUT_DIR / "structure_limited_candidates.csv",
        index=False,
    )

    gene_frequency.sort_values(
        "top25_drug_evaluations",
        ascending=False,
    ).to_csv(
        OUTPUT_DIR / "candidate_gene_prevalence.csv",
        index=False,
    )

    print("Candidate triage complete")
    print("Eligible unlabelled pairs:", len(eligible))
    print(
        "Phenotype propagation candidates:",
        len(phenotype_candidates),
    )
    print(
        "Phenotype candidate drugs:",
        phenotype_candidates[
            ["seed", "drug_idx"]
        ].drop_duplicates().shape[0],
    )
    print(
        "\nCandidate classes:"
    )
    print(
        eligible["candidate_class"]
        .value_counts()
        .to_string()
    )

    print("\nMost prevalent candidate genes:")
    print(
        gene_frequency.sort_values(
            "top25_drug_evaluations",
            ascending=False,
        )
        .head(20)[
            [
                "gene_name",
                "top25_drug_evaluations",
                "unique_drugs",
                "top25_prevalence",
                "specificity_weight",
            ]
        ]
        .to_string(index=False)
    )

    print(
        "\nSaved:",
        OUTPUT_DIR
        / "phenotype_propagation_candidates.csv",
    )


if __name__ == "__main__":
    main()
