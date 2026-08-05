#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import torch


MODELS = ["gcn", "rgcn", "rgat"]
PROTOCOLS = ["dg_context", "strict_cold"]
SEEDS = [42, 123, 2026]
BOOTSTRAPS = 10000


def bootstrap_ci(
    differences: np.ndarray,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)

    n = differences.shape[0]
    estimates = np.empty(BOOTSTRAPS, dtype=np.float64)

    for index in range(BOOTSTRAPS):
        sample_indices = rng.integers(
            low=0,
            high=n,
            size=n,
        )
        estimates[index] = differences[
            sample_indices
        ].mean()

    lower, upper = np.percentile(
        estimates,
        [2.5, 97.5],
    )

    return float(lower), float(upper)


records = []

for model in MODELS:
    for protocol in PROTOCOLS:
        for seed in SEEDS:
            model_path = (
                Path("results/models")
                / model
                / protocol
                / f"seed_{seed}"
                / "test_ranks.pt"
            )

            baseline_path = (
                Path("results/baselines")
                / "gene_degree"
                / protocol
                / f"seed_{seed}"
                / "test_ranks.pt"
            )

            model_ranks = torch.load(
                model_path,
                map_location="cpu",
                weights_only=False,
            ).float()

            baseline_ranks = torch.load(
                baseline_path,
                map_location="cpu",
                weights_only=False,
            ).float()

            if model_ranks.shape != baseline_ranks.shape:
                raise ValueError(
                    f"Rank shape mismatch for "
                    f"{model}, {protocol}, seed {seed}: "
                    f"{model_ranks.shape} versus "
                    f"{baseline_ranks.shape}"
                )

            model_rr = (
                1.0 / model_ranks
            ).numpy()

            baseline_rr = (
                1.0 / baseline_ranks
            ).numpy()

            rr_difference = model_rr - baseline_rr

            lower, upper = bootstrap_ci(
                rr_difference,
                seed=seed,
            )

            model_hits10 = (
                model_ranks <= 10
            ).float().numpy()

            baseline_hits10 = (
                baseline_ranks <= 10
            ).float().numpy()

            records.append(
                {
                    "model": model,
                    "protocol": protocol,
                    "seed": seed,
                    "n_queries": int(
                        model_ranks.numel()
                    ),
                    "model_mrr": float(
                        model_rr.mean()
                    ),
                    "gene_degree_mrr": float(
                        baseline_rr.mean()
                    ),
                    "mrr_difference": float(
                        rr_difference.mean()
                    ),
                    "mrr_difference_ci_lower": lower,
                    "mrr_difference_ci_upper": upper,
                    "model_hits_at_10": float(
                        model_hits10.mean()
                    ),
                    "gene_degree_hits_at_10": float(
                        baseline_hits10.mean()
                    ),
                    "hits_at_10_difference": float(
                        (
                            model_hits10
                            - baseline_hits10
                        ).mean()
                    ),
                    "model_better_queries": int(
                        (
                            model_ranks
                            < baseline_ranks
                        ).sum().item()
                    ),
                    "equal_rank_queries": int(
                        (
                            model_ranks
                            == baseline_ranks
                        ).sum().item()
                    ),
                    "gene_degree_better_queries": int(
                        (
                            model_ranks
                            > baseline_ranks
                        ).sum().item()
                    ),
                }
            )


output_path = Path(
    "results/model_vs_gene_degree_paired.json"
)

with output_path.open("w") as handle:
    json.dump(
        records,
        handle,
        indent=2,
    )


print(
    f"{'Model':<6} "
    f"{'Protocol':<12} "
    f"{'Seed':<6} "
    f"{'ΔMRR':<11} "
    f"{'95% CI':<25} "
    f"{'Wins':<7} "
    f"{'Ties':<7} "
    f"{'Losses':<7}"
)
print("-" * 90)

for record in records:
    ci_text = (
        f"[{record['mrr_difference_ci_lower']:.6f}, "
        f"{record['mrr_difference_ci_upper']:.6f}]"
    )

    print(
        f"{record['model']:<6} "
        f"{record['protocol']:<12} "
        f"{record['seed']:<6} "
        f"{record['mrr_difference']:<11.6f} "
        f"{ci_text:<25} "
        f"{record['model_better_queries']:<7} "
        f"{record['equal_rank_queries']:<7} "
        f"{record['gene_degree_better_queries']:<7}"
    )

print(f"\nSaved: {output_path}")
