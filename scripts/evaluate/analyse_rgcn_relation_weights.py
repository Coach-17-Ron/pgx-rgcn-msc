#!/usr/bin/env python
"""Reconstruct and compare effective RGCN relation matrices across seeds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(
    "results/models/molecular/rgcn/"
    "strict_cold"
)

OUTPUT_ROOT = Path(
    "results/consensus_inference/molecular/"
    "strict_cold/summary"
)

LONG_OUTPUT = (
    OUTPUT_ROOT
    / "rgcn_relation_weight_importance_by_seed.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_ROOT
    / "rgcn_relation_weight_importance_summary.csv"
)

SEEDS = [42, 123, 2026]


def effective_relation_matrices(
    bases: torch.Tensor,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    """Return one effective transformation matrix per relation.

    bases:
        [num_bases, channels, channels]

    coefficients:
        [num_relations, num_bases]
    """
    return torch.einsum(
        "rb,bij->rij",
        coefficients,
        bases,
    )


def matrix_statistics(
    matrices: torch.Tensor,
) -> dict[str, np.ndarray]:
    flat = matrices.reshape(
        matrices.shape[0],
        -1,
    )

    frobenius = torch.linalg.vector_norm(
        flat,
        dim=1,
    )

    spectral = torch.linalg.matrix_norm(
        matrices,
        ord=2,
    )

    mean_absolute = matrices.abs().mean(
        dim=(1, 2)
    )

    return {
        "frobenius_norm": (
            frobenius.cpu().numpy()
        ),
        "spectral_norm": (
            spectral.cpu().numpy()
        ),
        "mean_absolute_weight": (
            mean_absolute.cpu().numpy()
        ),
    }


def main() -> None:
    rows: list[dict[str, object]] = []

    for seed in SEEDS:
        checkpoint_path = (
            ROOT
            / f"seed_{seed}"
            / "best_checkpoint.pt"
        )

        print(
            f"Loading seed {seed}: "
            f"{checkpoint_path}"
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        state = checkpoint["model_state_dict"]
        relation_to_index = checkpoint[
            "relation_to_index"
        ]

        index_to_relation = {
            int(index): relation
            for relation, index
            in relation_to_index.items()
        }

        for layer in [1, 2]:
            bases = state[
                f"encoder.conv{layer}.weight"
            ].float()

            coefficients = state[
                f"encoder.conv{layer}.comp"
            ].float()

            matrices = effective_relation_matrices(
                bases,
                coefficients,
            )

            statistics = matrix_statistics(
                matrices
            )

            coefficient_norm = (
                torch.linalg.vector_norm(
                    coefficients,
                    dim=1,
                )
                .cpu()
                .numpy()
            )

            for relation_index in range(
                matrices.shape[0]
            ):
                rows.append(
                    {
                        "seed": seed,
                        "layer": layer,
                        "relation_index": relation_index,
                        "relation_name": (
                            index_to_relation[
                                relation_index
                            ]
                        ),
                        "coefficient_norm": float(
                            coefficient_norm[
                                relation_index
                            ]
                        ),
                        "frobenius_norm": float(
                            statistics[
                                "frobenius_norm"
                            ][relation_index]
                        ),
                        "spectral_norm": float(
                            statistics[
                                "spectral_norm"
                            ][relation_index]
                        ),
                        "mean_absolute_weight": float(
                            statistics[
                                "mean_absolute_weight"
                            ][relation_index]
                        ),
                    }
                )

    long = pd.DataFrame(rows)

    long["frobenius_rank_within_seed_layer"] = (
        long.groupby(
            ["seed", "layer"]
        )["frobenius_norm"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    long.to_csv(
        LONG_OUTPUT,
        index=False,
    )

    summary = (
        long.groupby(
            [
                "layer",
                "relation_index",
                "relation_name",
            ],
            as_index=False,
        )
        .agg(
            mean_frobenius_norm=(
                "frobenius_norm",
                "mean",
            ),
            std_frobenius_norm=(
                "frobenius_norm",
                "std",
            ),
            mean_spectral_norm=(
                "spectral_norm",
                "mean",
            ),
            mean_absolute_weight=(
                "mean_absolute_weight",
                "mean",
            ),
            mean_coefficient_norm=(
                "coefficient_norm",
                "mean",
            ),
            mean_rank=(
                "frobenius_rank_within_seed_layer",
                "mean",
            ),
            best_rank=(
                "frobenius_rank_within_seed_layer",
                "min",
            ),
            worst_rank=(
                "frobenius_rank_within_seed_layer",
                "max",
            ),
        )
    )

    summary[
        "stable_top10_all_seeds"
    ] = (
        summary["worst_rank"] <= 10
    )

    summary = summary.sort_values(
        [
            "layer",
            "mean_rank",
            "mean_frobenius_norm",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print("\nRGCN RELATION-WEIGHT ANALYSIS COMPLETE")
    print("Seeds:", len(SEEDS))
    print(
        "Relations:",
        long["relation_name"].nunique(),
    )
    print(
        "Rows:",
        len(long),
    )

    for layer in [1, 2]:
        print(
            f"\nTOP 15 RELATIONS — LAYER {layer}"
        )

        view = (
            summary.loc[
                summary["layer"].eq(layer),
                [
                    "relation_name",
                    "mean_frobenius_norm",
                    "std_frobenius_norm",
                    "mean_rank",
                    "best_rank",
                    "worst_rank",
                    "stable_top10_all_seeds",
                ],
            ]
            .head(15)
        )

        print(view.to_string(index=False))

    print("\nSaved:")
    print(LONG_OUTPUT)
    print(SUMMARY_OUTPUT)


if __name__ == "__main__":
    main()
