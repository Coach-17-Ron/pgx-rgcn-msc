#!/usr/bin/env python
"""Compare relation-family ablation with size-matched random edge deletion."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pgx_rgcn.models.graph_models import PGxGraphModel


SEEDS = [42, 123, 2026]
N_RANDOM_REPEATS = 20
RANDOM_SEED = 42

SUMMARY_ROOT = Path(
    "results/consensus_inference/molecular/"
    "strict_cold/summary"
)

COVERAGE_PATH = (
    SUMMARY_ROOT
    / "interpretability_candidate_seed_coverage.csv"
)

ABLATION_PATH = (
    SUMMARY_ROOT
    / "rgcn_relation_family_ablation_by_pair_seed.csv"
)

GLOBAL_PATH = (
    SUMMARY_ROOT
    / "rgcn_relation_family_ablation_global_summary.csv"
)

TENSOR_ROOT = Path(
    "artifacts/tensors_molecular/strict_cold"
)

MODEL_ROOT = Path(
    "results/models/molecular/rgcn/strict_cold"
)

OUTPUT_PATH = (
    SUMMARY_ROOT
    / "rgcn_relation_ablation_random_control.csv"
)

SUMMARY_OUTPUT = (
    SUMMARY_ROOT
    / "rgcn_relation_ablation_random_control_summary.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    parser.add_argument(
        "--top-families",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=N_RANDOM_REPEATS,
    )

    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if (
        requested == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable."
        )

    return torch.device(requested)


def load_model(
    checkpoint: dict[str, object],
    device: torch.device,
) -> PGxGraphModel:
    model = PGxGraphModel(
        model_name=str(checkpoint["model_name"]),
        num_nodes=int(checkpoint["num_nodes"]),
        num_node_types=int(
            checkpoint["num_node_types"]
        ),
        num_relations=int(
            checkpoint["num_relations"]
        ),
        embedding_dim=int(
            checkpoint["embedding_dim"]
        ),
        num_bases=int(checkpoint["num_bases"]),
        dropout=float(checkpoint["dropout"]),
        feature_mode=str(
            checkpoint.get("feature_mode", "id")
        ),
        molecular_feature_dim=checkpoint.get(
            "molecular_feature_dim"
        ),
        drug_type_index=int(
            checkpoint["node_type_to_index"]["drug"]
        ),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model


def encode(
    model: PGxGraphModel,
    bundle: dict[str, object],
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    with torch.no_grad():
        return model.encode(
            node_type=bundle["node_type"].to(device),
            edge_index=edge_index,
            edge_type=edge_type,
            molecular_features=(
                bundle["molecular_features"].to(device)
            ),
            has_molecular_features=(
                bundle["has_molecular_features"].to(device)
            ),
        )


def score_pairs(
    model: PGxGraphModel,
    embeddings: torch.Tensor,
    drug_indices: torch.Tensor,
    gene_indices: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        return (
            embeddings[drug_indices]
            * model.decoder.relation
            * embeddings[gene_indices]
        ).sum(dim=-1)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    print("Device:", device)

    coverage = pd.read_csv(
        COVERAGE_PATH,
        low_memory=False,
    )

    coverage = coverage.loc[
        coverage[
            "eligible_for_strict_cold_explanation"
        ].fillna(False)
    ].copy()

    observed = pd.read_csv(
        ABLATION_PATH,
        low_memory=False,
    )

    global_summary = pd.read_csv(
        GLOBAL_PATH,
        low_memory=False,
    )

    selected_families = (
        global_summary.sort_values(
            "mean_score_drop",
            ascending=False,
        )
        .head(args.top_families)
        ["relation_family"]
        .tolist()
    )

    print("Selected families:")
    for family in selected_families:
        print(" -", family)

    rng = np.random.default_rng(RANDOM_SEED)
    output_rows: list[dict[str, object]] = []

    for seed in SEEDS:
        seed_pairs = (
            coverage.loc[
                coverage["seed"].eq(seed)
            ]
            .sort_values(
                "external_validation_rank"
            )
            .reset_index(drop=True)
        )

        if seed_pairs.empty:
            continue

        print(
            f"\nSeed {seed}: "
            f"{len(seed_pairs)} eligible pairs"
        )

        bundle = torch.load(
            TENSOR_ROOT
            / f"seed_{seed}"
            / "tensor_bundle.pt",
            map_location="cpu",
            weights_only=False,
        )

        checkpoint = torch.load(
            MODEL_ROOT
            / f"seed_{seed}"
            / "best_checkpoint.pt",
            map_location="cpu",
            weights_only=False,
        )

        model = load_model(
            checkpoint,
            device,
        )

        edge_index = bundle["edge_index"].to(device)
        edge_type = bundle["edge_type"].to(device)

        relation_to_index = {
            str(name): int(index)
            for name, index in checkpoint[
                "relation_to_index"
            ].items()
        }

        drug_indices = torch.tensor(
            seed_pairs["drug_idx"].to_numpy(),
            dtype=torch.long,
            device=device,
        )

        gene_indices = torch.tensor(
            seed_pairs["gene_idx"].to_numpy(),
            dtype=torch.long,
            device=device,
        )

        baseline_embeddings = encode(
            model,
            bundle,
            edge_index,
            edge_type,
            device,
        )

        baseline_scores = score_pairs(
            model,
            baseline_embeddings,
            drug_indices,
            gene_indices,
        ).detach().cpu().numpy()

        del baseline_embeddings

        for family in selected_families:
            family_rows = observed.loc[
                observed["seed"].eq(seed)
                & observed[
                    "relation_family"
                ].eq(family)
            ]

            if family_rows.empty:
                continue

            relation_members = (
                str(
                    family_rows[
                        "relation_members"
                    ].iloc[0]
                )
                .split("|")
            )

            member_indices = torch.tensor(
                [
                    relation_to_index[name]
                    for name in relation_members
                ],
                dtype=edge_type.dtype,
                device=device,
            )

            target_mask = torch.isin(
                edge_type,
                member_indices,
            )

            removed_count = int(
                target_mask.sum().item()
            )

            eligible_random_positions = (
                torch.where(~target_mask)[0]
                .detach()
                .cpu()
                .numpy()
            )

            if removed_count > len(
                eligible_random_positions
            ):
                raise ValueError(
                    f"Not enough non-family edges for {family}"
                )

            random_score_drops = np.zeros(
                (
                    args.repeats,
                    len(seed_pairs),
                ),
                dtype=np.float64,
            )

            for repeat in range(args.repeats):
                sampled_positions = rng.choice(
                    eligible_random_positions,
                    size=removed_count,
                    replace=False,
                )

                keep_mask = torch.ones(
                    edge_type.shape[0],
                    dtype=torch.bool,
                    device=device,
                )

                keep_mask[
                    torch.tensor(
                        sampled_positions,
                        dtype=torch.long,
                        device=device,
                    )
                ] = False

                random_embeddings = encode(
                    model,
                    bundle,
                    edge_index[:, keep_mask],
                    edge_type[keep_mask],
                    device,
                )

                random_scores = score_pairs(
                    model,
                    random_embeddings,
                    drug_indices,
                    gene_indices,
                ).detach().cpu().numpy()

                random_score_drops[
                    repeat
                ] = (
                    baseline_scores
                    - random_scores
                )

                del random_embeddings

                if device.type == "cuda":
                    torch.cuda.empty_cache()

            observed_lookup = (
                family_rows.set_index(
                    "external_validation_rank"
                )["score_drop"]
                .to_dict()
            )

            for pair_position, pair in enumerate(
                seed_pairs.itertuples(index=False)
            ):
                observed_drop = float(
                    observed_lookup[
                        int(
                            pair.external_validation_rank
                        )
                    ]
                )

                null_values = random_score_drops[
                    :,
                    pair_position,
                ]

                null_mean = float(
                    null_values.mean()
                )

                null_std = float(
                    null_values.std(ddof=1)
                )

                empirical_p = float(
                    (
                        np.sum(
                            null_values
                            >= observed_drop
                        )
                        + 1
                    )
                    / (
                        len(null_values) + 1
                    )
                )

                z_score = (
                    (observed_drop - null_mean)
                    / null_std
                    if null_std > 0
                    else np.nan
                )

                output_rows.append(
                    {
                        "seed": seed,
                        "external_validation_rank": int(
                            pair.external_validation_rank
                        ),
                        "drug_name": pair.drug_name,
                        "gene_name": pair.gene_name,
                        "relation_family": family,
                        "removed_edge_count": removed_count,
                        "observed_score_drop": (
                            observed_drop
                        ),
                        "random_mean_score_drop": (
                            null_mean
                        ),
                        "random_std_score_drop": (
                            null_std
                        ),
                        "excess_score_drop_over_random": (
                            observed_drop - null_mean
                        ),
                        "random_control_z_score": z_score,
                        "empirical_one_sided_pvalue": (
                            empirical_p
                        ),
                        "random_repeats": args.repeats,
                    }
                )

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = pd.DataFrame(output_rows)
    result.to_csv(OUTPUT_PATH, index=False)

    summary = (
        result.groupby(
            "relation_family",
            as_index=False,
        )
        .agg(
            pair_seed_evaluations=(
                "observed_score_drop",
                "size",
            ),
            mean_observed_score_drop=(
                "observed_score_drop",
                "mean",
            ),
            mean_random_score_drop=(
                "random_mean_score_drop",
                "mean",
            ),
            mean_excess_drop_over_random=(
                "excess_score_drop_over_random",
                "mean",
            ),
            median_excess_drop_over_random=(
                "excess_score_drop_over_random",
                "median",
            ),
            fraction_exceeding_random_mean=(
                "excess_score_drop_over_random",
                lambda values: float(
                    (values > 0).mean()
                ),
            ),
            fraction_empirical_p_le_0_05=(
                "empirical_one_sided_pvalue",
                lambda values: float(
                    (values <= 0.05).mean()
                ),
            ),
        )
        .sort_values(
            "mean_excess_drop_over_random",
            ascending=False,
        )
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print(
        "\nRGCN RANDOM-EDGE CONTROL COMPLETE"
    )

    print("\nRELATION-FAMILY SUMMARY")
    print(summary.to_string(index=False))

    print("\nSaved:")
    print(OUTPUT_PATH)
    print(SUMMARY_OUTPUT)


if __name__ == "__main__":
    main()
