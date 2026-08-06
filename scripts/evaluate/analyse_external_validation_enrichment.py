#!/usr/bin/env python
"""Test whether externally supported pairs rank higher within the shortlist."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "results/consensus_inference/molecular/"
    "strict_cold/summary"
)

INPUT_PATH = ROOT / "final_external_validation_evidence_table.csv"
OUTPUT_PATH = ROOT / "external_validation_enrichment_summary.csv"
TOPK_PATH = ROOT / "external_validation_topk_enrichment.csv"

N_PERMUTATIONS = 100_000
RANDOM_SEED = 42


def permutation_pvalue(
    observed: float,
    values: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """One-sided test: supported pairs should have larger scores."""

    exceedances = 0

    for _ in range(N_PERMUTATIONS):
        shuffled = rng.permutation(labels)

        difference = (
            values[shuffled].mean()
            - values[~shuffled].mean()
        )

        if difference >= observed:
            exceedances += 1

    return (exceedances + 1) / (N_PERMUTATIONS + 1)


def main() -> None:
    df = pd.read_csv(INPUT_PATH, low_memory=False)

    df["supported"] = (
        df["has_positive_pubmed_support"]
        .fillna(False)
        .astype(bool)
    )

    supported = df.loc[df["supported"]]
    unsupported = df.loc[~df["supported"]]

    rng = np.random.default_rng(RANDOM_SEED)

    metrics = [
        (
            "validation_priority_score",
            True,
            "Higher validation-priority score",
        ),
        (
            "weighted_rrf_score",
            True,
            "Higher weighted RRF score",
        ),
        (
            "external_validation_rank",
            False,
            "Better external-validation rank",
        ),
        (
            "weighted_consensus_rank",
            False,
            "Better weighted consensus rank",
        ),
    ]

    rows = []

    for column, higher_is_better, description in metrics:
        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        usable = values.notna()

        metric_values = values.loc[usable].to_numpy()
        metric_labels = (
            df.loc[usable, "supported"]
            .to_numpy(dtype=bool)
        )

        if not higher_is_better:
            # Reverse rank direction so larger transformed values
            # always indicate a stronger model result.
            metric_values = -metric_values

        supported_values = metric_values[metric_labels]
        unsupported_values = metric_values[~metric_labels]

        observed_difference = (
            supported_values.mean()
            - unsupported_values.mean()
        )

        pvalue = permutation_pvalue(
            observed_difference,
            metric_values,
            metric_labels,
            rng,
        )

        original_supported = pd.to_numeric(
            supported[column],
            errors="coerce",
        )

        original_unsupported = pd.to_numeric(
            unsupported[column],
            errors="coerce",
        )

        rows.append(
            {
                "metric": column,
                "interpretation": description,
                "supported_n": original_supported.notna().sum(),
                "unsupported_n": original_unsupported.notna().sum(),
                "supported_mean": original_supported.mean(),
                "unsupported_mean": original_unsupported.mean(),
                "supported_median": original_supported.median(),
                "unsupported_median": original_unsupported.median(),
                "direction_adjusted_mean_difference": (
                    observed_difference
                ),
                "one_sided_permutation_pvalue": pvalue,
                "permutations": N_PERMUTATIONS,
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_PATH, index=False)

    topk_rows = []

    total_supported = int(df["supported"].sum())
    overall_rate = total_supported / len(df)

    ranked = df.sort_values(
        "external_validation_rank"
    ).reset_index(drop=True)

    for k in [5, 10, 15, 20, 30]:
        subset = ranked.head(k)
        supported_k = int(subset["supported"].sum())

        topk_rows.append(
            {
                "top_k": k,
                "supported_pairs": supported_k,
                "unsupported_pairs": k - supported_k,
                "positive_support_rate": supported_k / k,
                "overall_positive_support_rate": overall_rate,
                "fold_enrichment_over_shortlist": (
                    (supported_k / k) / overall_rate
                    if overall_rate > 0
                    else np.nan
                ),
            }
        )

    topk = pd.DataFrame(topk_rows)
    topk.to_csv(TOPK_PATH, index=False)

    print("EXTERNAL-VALIDATION ENRICHMENT ANALYSIS COMPLETE")
    print("Candidate pairs:", len(df))
    print("Positive-support pairs:", total_supported)
    print(
        "Overall positive-support rate:",
        f"{overall_rate:.3f}",
    )

    print("\nMETRIC COMPARISON")
    print(
        summary[
            [
                "metric",
                "supported_mean",
                "unsupported_mean",
                "supported_median",
                "unsupported_median",
                "direction_adjusted_mean_difference",
                "one_sided_permutation_pvalue",
            ]
        ].to_string(index=False)
    )

    print("\nTOP-K ENRICHMENT")
    print(topk.to_string(index=False))

    print("\nSUPPORTED PAIRS IN ORIGINAL VALIDATION ORDER")
    print(
        ranked.loc[
            ranked["supported"],
            [
                "external_validation_rank",
                "drug_name",
                "gene_name",
                "pubmed_pair_evidence_class",
                "validation_priority_score",
                "weighted_consensus_rank",
            ],
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(OUTPUT_PATH)
    print(TOPK_PATH)


if __name__ == "__main__":
    main()
