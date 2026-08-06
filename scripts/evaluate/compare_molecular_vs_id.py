#!/usr/bin/env python
"""Compare molecular and ID-feature graph models on matched test subsets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pgx_rgcn.evaluation import metrics_from_ranks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("results/models"),
    )
    parser.add_argument(
        "--molecular-root",
        type=Path,
        default=Path("results/models/molecular"),
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=Path("artifacts/splits"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(
            "artifacts/molecular_features/"
            "morgan_radius2_2048_metadata.parquet"
        ),
    )
    parser.add_argument(
        "--duplicate-report",
        type=Path,
        default=Path(
            "results/cross_split_duplicate_structures.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/molecular_summary"),
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def load_ranks(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(path)

    ranks = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    return torch.as_tensor(ranks).long()


def main() -> None:
    args = parse_args()

    metadata = pd.read_parquet(args.metadata)

    has_smiles_lookup = (
        metadata.set_index("node_index")["has_fingerprint"]
        .astype(bool)
        .to_dict()
    )

    drug_id_lookup = (
        metadata.set_index("node_index")["node_id"]
        .to_dict()
    )

    duplicates = pd.read_csv(args.duplicate_report)

    rng = np.random.default_rng(args.bootstrap_seed)
    rows: list[dict[str, object]] = []

    for model in ("gcn", "rgcn", "rgat"):
        for protocol in ("dg_context", "strict_cold"):
            for seed in (42, 123, 2026):
                pairs = pd.read_parquet(
                    args.split_root
                    / protocol
                    / f"seed_{seed}"
                    / "test_pairs.parquet"
                ).reset_index(drop=True)

                baseline_ranks = load_ranks(
                    args.baseline_root
                    / model
                    / protocol
                    / f"seed_{seed}"
                    / "test_ranks.pt"
                )

                molecular_ranks = load_ranks(
                    args.molecular_root
                    / model
                    / protocol
                    / f"seed_{seed}"
                    / "test_ranks.pt"
                )

                if not (
                    len(pairs)
                    == len(baseline_ranks)
                    == len(molecular_ranks)
                ):
                    raise AssertionError(
                        f"Length mismatch for "
                        f"{model} {protocol} seed={seed}"
                    )

                pairs["has_smiles"] = (
                    pairs["drug_idx"]
                    .map(has_smiles_lookup)
                    .fillna(False)
                    .astype(bool)
                )

                pairs["drug_id"] = pairs["drug_idx"].map(
                    drug_id_lookup
                )

                overlap_ids = set(
                    duplicates.loc[
                        (duplicates["seed"] == seed)
                        & (duplicates["split"] == "test")
                        & duplicates[
                            "structure_splits"
                        ].str.contains(
                            "train|training_context_only",
                            regex=True,
                            na=False,
                        ),
                        "drug_id",
                    ]
                )

                pairs["structure_clean"] = ~pairs[
                    "drug_id"
                ].isin(overlap_ids)

                subsets = {
                    "all": np.ones(
                        len(pairs),
                        dtype=bool,
                    ),
                    "with_smiles": pairs[
                        "has_smiles"
                    ].to_numpy(),
                    "without_smiles": (
                        ~pairs["has_smiles"]
                    ).to_numpy(),
                    "with_smiles_structure_clean": (
                        pairs["has_smiles"]
                        & pairs["structure_clean"]
                    ).to_numpy(),
                }

                for subset_name, mask in subsets.items():
                    mask_tensor = torch.from_numpy(
                        np.asarray(
                            mask,
                            dtype=np.bool_,
                        ).copy()
                    )

                    baseline_subset = baseline_ranks[
                        mask_tensor
                    ]
                    molecular_subset = molecular_ranks[
                        mask_tensor
                    ]

                    baseline_metrics = metrics_from_ranks(
                        baseline_subset
                    )
                    molecular_metrics = metrics_from_ranks(
                        molecular_subset
                    )

                    paired_delta = (
                        1.0
                        / molecular_subset.float().numpy()
                        - 1.0
                        / baseline_subset.float().numpy()
                    )

                    bootstrap_delta = np.empty(
                        args.bootstrap_samples,
                        dtype=float,
                    )

                    for index in range(
                        args.bootstrap_samples
                    ):
                        sampled = rng.integers(
                            0,
                            len(paired_delta),
                            size=len(paired_delta),
                        )

                        bootstrap_delta[index] = (
                            paired_delta[sampled].mean()
                        )

                    ci_low, ci_high = np.quantile(
                        bootstrap_delta,
                        [0.025, 0.975],
                    )

                    rows.append(
                        {
                            "model": model,
                            "protocol": protocol,
                            "seed": seed,
                            "subset": subset_name,
                            "n_queries": len(baseline_subset),
                            "baseline_mrr": baseline_metrics[
                                "mrr"
                            ],
                            "molecular_mrr": molecular_metrics[
                                "mrr"
                            ],
                            "delta_mrr": (
                                molecular_metrics["mrr"]
                                - baseline_metrics["mrr"]
                            ),
                            "bootstrap_ci_low": ci_low,
                            "bootstrap_ci_high": ci_high,
                        }
                    )

    results = pd.DataFrame(rows)

    focused = results.loc[
        (results["protocol"] == "strict_cold")
        & (
            results["subset"]
            == "with_smiles_structure_clean"
        )
    ].copy()

    summary = (
        focused.groupby("model")
        .agg(
            seeds=("seed", "count"),
            mean_baseline_mrr=("baseline_mrr", "mean"),
            sd_baseline_mrr=("baseline_mrr", "std"),
            mean_molecular_mrr=("molecular_mrr", "mean"),
            sd_molecular_mrr=("molecular_mrr", "std"),
            mean_delta_mrr=("delta_mrr", "mean"),
            sd_delta_mrr=("delta_mrr", "std"),
            positive_seeds=(
                "delta_mrr",
                lambda values: int((values > 0).sum()),
            ),
            significant_positive_seeds=(
                "bootstrap_ci_low",
                lambda values: int((values > 0).sum()),
            ),
            significant_negative_seeds=(
                "bootstrap_ci_high",
                lambda values: int((values < 0).sum()),
            ),
        )
        .reset_index()
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    detailed_path = (
        args.output_dir
        / "paired_molecular_vs_id_by_subset.csv"
    )

    summary_path = (
        args.output_dir
        / "strict_cold_smiles_clean_seed_summary.csv"
    )

    results.to_csv(
        detailed_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print(
        focused.round(6).to_string(index=False)
    )

    print("\nMODEL SUMMARY")
    print(
        summary.round(6).to_string(index=False)
    )

    print("\nSaved:", detailed_path)
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()
