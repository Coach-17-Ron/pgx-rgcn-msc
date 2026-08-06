#!/usr/bin/env python
"""Sensitivity analysis using increasingly strict PubMed support definitions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "results/consensus_inference/molecular/"
    "strict_cold/summary"
)

INPUT_PATH = ROOT / "final_external_validation_evidence_table.csv"
OUTPUT_PATH = ROOT / "external_validation_sensitivity_summary.csv"

N_PERMUTATIONS = 100_000
RANDOM_SEED = 42


SUPPORT_DEFINITIONS = {
    "any_positive_support": {
        "direct_supported",
        "direct_mixed_evidence",
        "indirect_mechanistic_support",
    },
    "any_direct_support": {
        "direct_supported",
        "direct_mixed_evidence",
    },
    "strict_direct_support": {
        "direct_supported",
    },
}


METRICS = [
    ("weighted_rrf_score", True),
    ("weighted_consensus_rank", False),
    ("validation_priority_score", True),
    ("external_validation_rank", False),
]


def permutation_pvalue(
    values: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    observed = (
        values[labels].mean()
        - values[~labels].mean()
    )

    exceedances = 0

    for _ in range(N_PERMUTATIONS):
        shuffled = rng.permutation(labels)

        difference = (
            values[shuffled].mean()
            - values[~shuffled].mean()
        )

        if difference >= observed:
            exceedances += 1

    pvalue = (
        exceedances + 1
    ) / (
        N_PERMUTATIONS + 1
    )

    return observed, pvalue


def main() -> None:
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    rng = np.random.default_rng(RANDOM_SEED)

    rows = []

    for definition, positive_classes in SUPPORT_DEFINITIONS.items():
        labels = (
            df["pubmed_pair_evidence_class"]
            .isin(positive_classes)
            .to_numpy(dtype=bool)
        )

        for metric, higher_is_better in METRICS:
            numeric = pd.to_numeric(
                df[metric],
                errors="coerce",
            )

            usable = numeric.notna().to_numpy()

            values = numeric.to_numpy()[usable]
            metric_labels = labels[usable]

            original_values = values.copy()

            if not higher_is_better:
                values = -values

            observed, pvalue = permutation_pvalue(
                values,
                metric_labels,
                rng,
            )

            rows.append(
                {
                    "support_definition": definition,
                    "metric": metric,
                    "positive_n": int(metric_labels.sum()),
                    "comparison_n": int((~metric_labels).sum()),
                    "positive_mean": float(
                        original_values[
                            metric_labels
                        ].mean()
                    ),
                    "comparison_mean": float(
                        original_values[
                            ~metric_labels
                        ].mean()
                    ),
                    "positive_median": float(
                        np.median(
                            original_values[
                                metric_labels
                            ]
                        )
                    ),
                    "comparison_median": float(
                        np.median(
                            original_values[
                                ~metric_labels
                            ]
                        )
                    ),
                    "direction_adjusted_mean_difference": observed,
                    "one_sided_permutation_pvalue": pvalue,
                    "permutations": N_PERMUTATIONS,
                }
            )

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_PATH, index=False)

    print("VALIDATION SENSITIVITY ANALYSIS COMPLETE")

    print(
        result[
            [
                "support_definition",
                "metric",
                "positive_n",
                "positive_mean",
                "comparison_mean",
                "direction_adjusted_mean_difference",
                "one_sided_permutation_pvalue",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
