#!/usr/bin/env python
"""Summarise three-seed molecular consensus inference results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


SEEDS = (42, 123, 2026)
MODELS = ("gcn", "rgcn", "rgat")
RANK_COLUMNS = (
    "gcn_rank",
    "rgcn_rank",
    "rgat_rank",
    "equal_rrf_rank",
    "weighted_consensus_rank",
)
THRESHOLDS = (1, 3, 10, 25, 100, 250)

RESULT_ROOT = Path(
    "results/consensus_inference/molecular/strict_cold"
)
SPLIT_ROOT = Path("artifacts/splits/strict_cold")
TENSOR_ROOT = Path("artifacts/tensors_molecular/strict_cold")
OUTPUT_DIR = Path(
    "results/consensus_inference/molecular/strict_cold/summary"
)


def recovery_rows(
    seed: int,
    labelled: pd.DataFrame,
) -> list[dict[str, object]]:
    associated = labelled.loc[
        labelled["association"].eq("associated")
    ]

    rows: list[dict[str, object]] = []

    for rank_column in RANK_COLUMNS:
        for threshold in THRESHOLDS:
            recovered = int(
                (
                    associated[rank_column]
                    <= threshold
                ).sum()
            )

            rows.append(
                {
                    "seed": seed,
                    "ranking": rank_column,
                    "threshold": threshold,
                    "recovered": recovered,
                    "associated_pairs": len(associated),
                    "rate": recovered / len(associated),
                }
            )

    return rows


def diversity_rows(
    seed: int,
    candidates: pd.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for rank_column in RANK_COLUMNS:
        for threshold in (1, 10, 25, 100):
            gene_sets = (
                candidates.loc[
                    candidates[rank_column].le(threshold),
                    ["drug_idx", "gene_idx"],
                ]
                .groupby("drug_idx")["gene_idx"]
                .apply(lambda values: tuple(sorted(values)))
            )

            rows.append(
                {
                    "seed": seed,
                    "ranking": rank_column,
                    "threshold": threshold,
                    "heldout_drugs": len(gene_sets),
                    "unique_gene_sets": int(
                        gene_sets.nunique()
                    ),
                }
            )

    return rows


def jaccard_rows(
    seed: int,
    candidates: pd.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for rank_column in RANK_COLUMNS:
        gene_sets = {
            int(drug_idx): set(group["gene_idx"])
            for drug_idx, group in candidates.loc[
                candidates[rank_column].le(100)
            ].groupby("drug_idx")
        }

        drugs = sorted(gene_sets)
        similarities: list[float] = []

        for position, first in enumerate(drugs):
            for second in drugs[position + 1:]:
                first_set = gene_sets[first]
                second_set = gene_sets[second]

                similarities.append(
                    len(first_set & second_set)
                    / len(first_set | second_set)
                )

        rows.append(
            {
                "seed": seed,
                "ranking": rank_column,
                "mean_top100_jaccard": float(
                    np.mean(similarities)
                ),
                "median_top100_jaccard": float(
                    np.median(similarities)
                ),
                "min_top100_jaccard": float(
                    np.min(similarities)
                ),
                "max_top100_jaccard": float(
                    np.max(similarities)
                ),
            }
        )

    return rows


def coverage_row(seed: int) -> dict[str, object]:
    assignments = pd.read_parquet(
        SPLIT_ROOT
        / f"seed_{seed}"
        / "drug_assignments.parquet"
    )

    test_drugs = assignments.loc[
        assignments["split"].eq("test"),
        ["drug_idx"],
    ]

    bundle = torch.load(
        TENSOR_ROOT
        / f"seed_{seed}"
        / "tensor_bundle.pt",
        map_location="cpu",
        weights_only=False,
    )

    has_features = bundle[
        "has_molecular_features"
    ].bool()

    indices = torch.tensor(
        test_drugs["drug_idx"].to_numpy(),
        dtype=torch.long,
    )

    covered = int(
        has_features[indices].sum().item()
    )

    return {
        "seed": seed,
        "heldout_drugs": len(test_drugs),
        "with_molecular_features": covered,
        "without_molecular_features": (
            len(test_drugs) - covered
        ),
        "coverage_rate": covered / len(test_drugs),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    recovery: list[dict[str, object]] = []
    diversity: list[dict[str, object]] = []
    jaccard: list[dict[str, object]] = []
    coverage: list[dict[str, object]] = []
    top_candidates: list[pd.DataFrame] = []
    labelled_all: list[pd.DataFrame] = []

    for seed in SEEDS:
        seed_dir = RESULT_ROOT / f"seed_{seed}"

        candidates = pd.read_csv(
            seed_dir / "top_consensus_candidates.csv"
        )

        labelled = pd.read_csv(
            seed_dir / "labelled_pair_rankings.csv"
        )

        recovery.extend(
            recovery_rows(seed, labelled)
        )
        diversity.extend(
            diversity_rows(seed, candidates)
        )
        jaccard.extend(
            jaccard_rows(seed, candidates)
        )
        coverage.append(coverage_row(seed))

        selected = candidates.loc[
            candidates[
                "weighted_consensus_rank"
            ].le(25)
        ].copy()

        top_candidates.append(selected)
        labelled_all.append(labelled)

    recovery_frame = pd.DataFrame(recovery)
    diversity_frame = pd.DataFrame(diversity)
    jaccard_frame = pd.DataFrame(jaccard)
    coverage_frame = pd.DataFrame(coverage)

    combined_candidates = pd.concat(
        top_candidates,
        ignore_index=True,
    )

    combined_labelled = pd.concat(
        labelled_all,
        ignore_index=True,
    )

    macro_recovery = (
        recovery_frame.groupby(
            ["ranking", "threshold"]
        )["rate"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    repeated_predictions = (
        combined_candidates.groupby(
            [
                "drug_idx",
                "drug_id",
                "drug_name",
                "gene_idx",
                "gene_id",
                "gene_name",
            ],
            dropna=False,
        )
        .agg(
            seeds_observed=("seed", "nunique"),
            seed_list=(
                "seed",
                lambda values: "|".join(
                    str(value)
                    for value in sorted(set(values))
                ),
            ),
            mean_weighted_rank=(
                "weighted_consensus_rank",
                "mean",
            ),
            best_weighted_rank=(
                "weighted_consensus_rank",
                "min",
            ),
            models_top_25_mean=(
                "models_top_25",
                "mean",
            ),
        )
        .reset_index()
        .loc[lambda frame: frame["seeds_observed"].ge(2)]
        .sort_values(
            [
                "seeds_observed",
                "mean_weighted_rank",
            ],
            ascending=[False, True],
        )
    )

    recovery_frame.to_csv(
        OUTPUT_DIR / "seed_recovery_metrics.csv",
        index=False,
    )
    macro_recovery.to_csv(
        OUTPUT_DIR / "macro_recovery_metrics.csv",
        index=False,
    )
    diversity_frame.to_csv(
        OUTPUT_DIR / "ranking_diversity.csv",
        index=False,
    )
    jaccard_frame.to_csv(
        OUTPUT_DIR / "top100_jaccard.csv",
        index=False,
    )
    coverage_frame.to_csv(
        OUTPUT_DIR / "molecular_coverage.csv",
        index=False,
    )
    combined_candidates.to_csv(
        OUTPUT_DIR / "all_seed_top25_candidates.csv",
        index=False,
    )
    combined_labelled.to_csv(
        OUTPUT_DIR / "all_seed_labelled_rankings.csv",
        index=False,
    )
    repeated_predictions.to_csv(
        OUTPUT_DIR / "repeated_drug_gene_predictions.csv",
        index=False,
    )

    weighted_macro = macro_recovery.loc[
        macro_recovery["ranking"].eq(
            "weighted_consensus_rank"
        )
    ]

    report_lines = [
        "# Molecular strict-cold consensus summary",
        "",
        "Results remain seed-specific because held-out drug "
        "sets differ across random splits.",
        "",
        "## Molecular coverage",
        "",
        "```",
        coverage_frame.to_string(index=False),
        "```",
        "",
        "## Weighted consensus recovery",
        "",
        "```",
        weighted_macro.to_string(index=False),
        "```",
        "",
        "## Interpretation",
        "",
        "- Molecular features restored drug-specific ranking "
        "variation for GCN and R-GCN.",
        "- RGAT was retained, but its consensus contribution "
        "was reduced using validation-derived weights.",
        "- Candidate rankings are prioritisation outputs, not "
        "confirmed novel pharmacogenomic associations.",
        "- Drugs without molecular features use the shared "
        "missing-feature representation and require separate "
        "interpretation.",
        "",
    ]

    report_path = (
        OUTPUT_DIR / "consensus_summary.md"
    )

    report_path.write_text(
        "\n".join(report_lines)
    )

    print("Consensus summary complete")
    print("Seeds:", SEEDS)
    print(
        "Held-out evaluations:",
        int(coverage_frame["heldout_drugs"].sum()),
    )
    print(
        "Mean molecular coverage:",
        f"{coverage_frame['coverage_rate'].mean():.4f}",
    )
    print(
        "Top-25 candidate rows:",
        len(combined_candidates),
    )
    print(
        "Repeated drug-gene predictions:",
        len(repeated_predictions),
    )
    print("Saved:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
