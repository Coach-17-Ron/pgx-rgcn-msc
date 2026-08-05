#!/usr/bin/env python3
"""Paired bootstrap comparison of calibrated models and gene-degree."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


MODELS = ["gcn", "rgcn", "rgat"]
PROTOCOLS = ["dg_context", "strict_cold"]
SEEDS = [42, 123, 2026]
BOOTSTRAPS = 10000


def bootstrap_mean_ci(
    differences: np.ndarray,
    seed: int,
) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    n_queries = differences.shape[0]

    estimates = np.empty(
        BOOTSTRAPS,
        dtype=np.float64,
    )

    for bootstrap_index in range(BOOTSTRAPS):
        sampled_indices = generator.integers(
            low=0,
            high=n_queries,
            size=n_queries,
        )

        estimates[bootstrap_index] = differences[
            sampled_indices
        ].mean()

    lower, upper = np.percentile(
        estimates,
        [2.5, 97.5],
    )

    return float(lower), float(upper)


records: list[dict[str, float | int | str]] = []

for model in MODELS:
    for protocol in PROTOCOLS:
        for seed in SEEDS:
            calibrated_dir = (
                Path("results/degree_calibrated")
                / model
                / protocol
                / f"seed_{seed}"
            )

            calibrated_ranks = torch.load(
                calibrated_dir
                / "degree_calibrated_test_ranks.pt",
                map_location="cpu",
                weights_only=False,
            ).float()

            baseline_ranks = torch.load(
                Path("results/baselines")
                / "gene_degree"
                / protocol
                / f"seed_{seed}"
                / "test_ranks.pt",
                map_location="cpu",
                weights_only=False,
            ).float()

            with (
                calibrated_dir
                / "degree_calibrated_metrics.json"
            ).open() as handle:
                calibration = json.load(handle)

            if calibrated_ranks.shape != baseline_ranks.shape:
                raise ValueError(
                    f"Rank shape mismatch for {model}, "
                    f"{protocol}, seed {seed}."
                )

            calibrated_rr = (
                1.0 / calibrated_ranks
            ).numpy()

            baseline_rr = (
                1.0 / baseline_ranks
            ).numpy()

            rr_difference = (
                calibrated_rr - baseline_rr
            )

            ci_lower, ci_upper = bootstrap_mean_ci(
                rr_difference,
                seed=seed,
            )

            calibrated_hits10 = (
                calibrated_ranks <= 10
            ).float().numpy()

            baseline_hits10 = (
                baseline_ranks <= 10
            ).float().numpy()

            records.append(
                {
                    "model": model,
                    "protocol": protocol,
                    "seed": seed,
                    "selected_alpha": float(
                        calibration["selected_alpha"]
                    ),
                    "n_queries": int(
                        calibrated_ranks.numel()
                    ),
                    "calibrated_mrr": float(
                        calibrated_rr.mean()
                    ),
                    "gene_degree_mrr": float(
                        baseline_rr.mean()
                    ),
                    "mrr_difference": float(
                        rr_difference.mean()
                    ),
                    "mrr_difference_ci_lower": ci_lower,
                    "mrr_difference_ci_upper": ci_upper,
                    "calibrated_hits_at_10": float(
                        calibrated_hits10.mean()
                    ),
                    "gene_degree_hits_at_10": float(
                        baseline_hits10.mean()
                    ),
                    "hits_at_10_difference": float(
                        (
                            calibrated_hits10
                            - baseline_hits10
                        ).mean()
                    ),
                    "calibrated_better_queries": int(
                        (
                            calibrated_ranks
                            < baseline_ranks
                        ).sum().item()
                    ),
                    "equal_rank_queries": int(
                        (
                            calibrated_ranks
                            == baseline_ranks
                        ).sum().item()
                    ),
                    "gene_degree_better_queries": int(
                        (
                            calibrated_ranks
                            > baseline_ranks
                        ).sum().item()
                    ),
                }
            )


results = pd.DataFrame(records)

results.to_csv(
    "results/degree_calibrated_vs_gene_degree_paired.csv",
    index=False,
)

with Path(
    "results/degree_calibrated_vs_gene_degree_paired.json"
).open("w") as handle:
    json.dump(
        records,
        handle,
        indent=2,
    )


print(
    f"{'Model':<6} "
    f"{'Protocol':<12} "
    f"{'Seed':<6} "
    f"{'Alpha':<7} "
    f"{'Delta MRR':<11} "
    f"{'95% CI':<25} "
    f"{'Wins':<7} "
    f"{'Ties':<7} "
    f"{'Losses':<7}"
)
print("-" * 96)

for record in records:
    ci_text = (
        f"[{record['mrr_difference_ci_lower']:.6f}, "
        f"{record['mrr_difference_ci_upper']:.6f}]"
    )

    print(
        f"{record['model']:<6} "
        f"{record['protocol']:<12} "
        f"{record['seed']:<6} "
        f"{record['selected_alpha']:<7.2f} "
        f"{record['mrr_difference']:<11.6f} "
        f"{ci_text:<25} "
        f"{record['calibrated_better_queries']:<7} "
        f"{record['equal_rank_queries']:<7} "
        f"{record['gene_degree_better_queries']:<7}"
    )

print("\nSaved:")
print("results/degree_calibrated_vs_gene_degree_paired.csv")
print("results/degree_calibrated_vs_gene_degree_paired.json")
