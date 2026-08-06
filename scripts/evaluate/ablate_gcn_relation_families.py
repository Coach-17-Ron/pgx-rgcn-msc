#!/usr/bin/env python
"""Pair-level reciprocal relation-family ablation for molecular GCN."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pgx_rgcn.models.graph_models import PGxGraphModel


SEEDS = [42, 123, 2026]
NODE_TYPES = {"drug", "gene", "variant", "phenotype"}

SUMMARY_ROOT = Path(
    "results/consensus_inference/molecular/"
    "strict_cold/summary"
)

COVERAGE_PATH = (
    SUMMARY_ROOT
    / "interpretability_candidate_seed_coverage.csv"
)

FINAL_EVIDENCE_PATH = (
    SUMMARY_ROOT
    / "final_external_validation_evidence_table.csv"
)

TENSOR_ROOT = Path(
    "artifacts/tensors_molecular/strict_cold"
)

MODEL_ROOT = Path(
    "results/models/molecular/gcn/strict_cold"
)

LONG_OUTPUT = (
    SUMMARY_ROOT
    / "gcn_relation_family_ablation_by_pair_seed.csv"
)

PAIR_OUTPUT = (
    SUMMARY_ROOT
    / "gcn_relation_family_ablation_pair_summary.csv"
)

GLOBAL_OUTPUT = (
    SUMMARY_ROOT
    / "gcn_relation_family_ablation_global_summary.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
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
            "CUDA was requested but is unavailable."
        )

    return torch.device(requested)


def relation_family_name(
    relation_name: str,
) -> str:
    """Collapse reciprocal directed relations into one family."""

    parts = relation_name.split("_")

    if (
        len(parts) < 3
        or parts[0] not in NODE_TYPES
        or parts[1] not in NODE_TYPES
    ):
        return relation_name

    source_type = parts[0]
    target_type = parts[1]
    relation_suffix = "_".join(parts[2:])

    endpoint_pair = sorted(
        [source_type, target_type]
    )

    return (
        f"{endpoint_pair[0]}_{endpoint_pair[1]}_"
        f"{relation_suffix}"
    )


def build_relation_families(
    relation_to_index: dict[str, int],
) -> dict[str, list[tuple[str, int]]]:
    families: dict[
        str,
        list[tuple[str, int]],
    ] = defaultdict(list)

    for relation_name, relation_index in (
        relation_to_index.items()
    ):
        family = relation_family_name(
            str(relation_name)
        )

        families[family].append(
            (
                str(relation_name),
                int(relation_index),
            )
        )

    return {
        family: sorted(
            members,
            key=lambda item: item[1],
        )
        for family, members in sorted(
            families.items()
        )
    }


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
            checkpoint.get(
                "feature_mode",
                "id",
            )
        ),
        molecular_feature_dim=checkpoint.get(
            "molecular_feature_dim"
        ),
        drug_type_index=int(
            checkpoint["node_type_to_index"][
                "drug"
            ]
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
    node_type: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    molecular_features: torch.Tensor,
    has_molecular_features: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        return model.encode(
            node_type=node_type,
            edge_index=edge_index,
            edge_type=edge_type,
            molecular_features=molecular_features,
            has_molecular_features=(
                has_molecular_features
            ),
        )


def score_and_rank_pairs(
    model: PGxGraphModel,
    embeddings: torch.Tensor,
    pair_drug_indices: torch.Tensor,
    pair_gene_indices: torch.Tensor,
    all_gene_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return target score and rank for each candidate pair."""

    with torch.no_grad():
        drug_embeddings = embeddings[
            pair_drug_indices
        ]

        candidate_gene_embeddings = embeddings[
            all_gene_indices
        ]

        relation = model.decoder.relation

        all_scores = (
            drug_embeddings[:, None, :]
            * relation[None, None, :]
            * candidate_gene_embeddings[None, :, :]
        ).sum(dim=-1)

        gene_position_lookup = {
            int(gene_idx): position
            for position, gene_idx in enumerate(
                all_gene_indices
                .detach()
                .cpu()
                .tolist()
            )
        }

        target_positions = torch.tensor(
            [
                gene_position_lookup[
                    int(gene_idx)
                ]
                for gene_idx in (
                    pair_gene_indices
                    .detach()
                    .cpu()
                    .tolist()
                )
            ],
            dtype=torch.long,
            device=all_scores.device,
        )

        row_indices = torch.arange(
            all_scores.shape[0],
            device=all_scores.device,
        )

        target_scores = all_scores[
            row_indices,
            target_positions,
        ]

        # Rank 1 is best. This matches descending score order.
        target_ranks = (
            (all_scores > target_scores[:, None])
            .sum(dim=1)
            + 1
        )

    return target_scores, target_ranks


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

    evidence = pd.read_csv(
        FINAL_EVIDENCE_PATH,
        low_memory=False,
    )[
        [
            "external_validation_rank",
            "drug_name",
            "gene_name",
            "pubmed_pair_evidence_class",
            "has_positive_pubmed_support",
        ]
    ].drop_duplicates(
        [
            "external_validation_rank",
            "drug_name",
            "gene_name",
        ]
    )

    coverage = coverage.merge(
        evidence,
        on=[
            "external_validation_rank",
            "drug_name",
            "gene_name",
        ],
        how="left",
        validate="many_to_one",
    )

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

        tensor_path = (
            TENSOR_ROOT
            / f"seed_{seed}"
            / "tensor_bundle.pt"
        )

        checkpoint_path = (
            MODEL_ROOT
            / f"seed_{seed}"
            / "best_checkpoint.pt"
        )

        bundle = torch.load(
            tensor_path,
            map_location="cpu",
            weights_only=False,
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        model = load_model(
            checkpoint,
            device,
        )

        relation_to_index = {
            str(name): int(index)
            for name, index in checkpoint[
                "relation_to_index"
            ].items()
        }

        families = build_relation_families(
            relation_to_index
        )

        print(
            "Relation families:",
            len(families),
        )

        node_type = bundle["node_type"].to(
            device
        )

        edge_index = bundle["edge_index"].to(
            device
        )

        edge_type = bundle["edge_type"].to(
            device
        )

        molecular_features = bundle[
            "molecular_features"
        ].to(device)

        has_molecular_features = bundle[
            "has_molecular_features"
        ].to(device)

        gene_type_index = int(
            bundle["node_type_to_index"]["gene"]
        )

        all_gene_indices = torch.where(
            bundle["node_type"]
            == gene_type_index
        )[0].long().to(device)

        pair_drug_indices = torch.tensor(
            seed_pairs["drug_idx"].to_numpy(),
            dtype=torch.long,
            device=device,
        )

        pair_gene_indices = torch.tensor(
            seed_pairs["gene_idx"].to_numpy(),
            dtype=torch.long,
            device=device,
        )

        baseline_embeddings = encode(
            model=model,
            node_type=node_type,
            edge_index=edge_index,
            edge_type=edge_type,
            molecular_features=molecular_features,
            has_molecular_features=(
                has_molecular_features
            ),
        )

        (
            baseline_scores,
            baseline_ranks,
        ) = score_and_rank_pairs(
            model=model,
            embeddings=baseline_embeddings,
            pair_drug_indices=pair_drug_indices,
            pair_gene_indices=pair_gene_indices,
            all_gene_indices=all_gene_indices,
        )

        baseline_scores_cpu = (
            baseline_scores.detach().cpu()
        )

        baseline_ranks_cpu = (
            baseline_ranks.detach().cpu()
        )

        del baseline_embeddings

        for family_number, (
            family_name,
            members,
        ) in enumerate(
            families.items(),
            start=1,
        ):
            member_indices = torch.tensor(
                [
                    relation_index
                    for _, relation_index in members
                ],
                dtype=edge_type.dtype,
                device=device,
            )

            keep_mask = ~torch.isin(
                edge_type,
                member_indices,
            )

            retained_edge_index = edge_index[
                :,
                keep_mask,
            ]

            retained_edge_type = edge_type[
                keep_mask
            ]

            removed_edge_count = int(
                (~keep_mask).sum().item()
            )

            print(
                f"  [{family_number:02d}/"
                f"{len(families):02d}] "
                f"{family_name}: "
                f"removed {removed_edge_count} edges"
            )

            ablated_embeddings = encode(
                model=model,
                node_type=node_type,
                edge_index=retained_edge_index,
                edge_type=retained_edge_type,
                molecular_features=(
                    molecular_features
                ),
                has_molecular_features=(
                    has_molecular_features
                ),
            )

            (
                ablated_scores,
                ablated_ranks,
            ) = score_and_rank_pairs(
                model=model,
                embeddings=ablated_embeddings,
                pair_drug_indices=(
                    pair_drug_indices
                ),
                pair_gene_indices=(
                    pair_gene_indices
                ),
                all_gene_indices=all_gene_indices,
            )

            ablated_scores_cpu = (
                ablated_scores.detach().cpu()
            )

            ablated_ranks_cpu = (
                ablated_ranks.detach().cpu()
            )

            member_names = "|".join(
                name for name, _ in members
            )

            for pair_position, pair in (
                enumerate(
                    seed_pairs.itertuples(
                        index=False
                    )
                )
            ):
                baseline_score = float(
                    baseline_scores_cpu[
                        pair_position
                    ].item()
                )

                ablated_score = float(
                    ablated_scores_cpu[
                        pair_position
                    ].item()
                )

                baseline_rank = int(
                    baseline_ranks_cpu[
                        pair_position
                    ].item()
                )

                ablated_rank = int(
                    ablated_ranks_cpu[
                        pair_position
                    ].item()
                )

                output_rows.append(
                    {
                        "seed": seed,
                        "external_validation_rank": (
                            int(
                                pair.external_validation_rank
                            )
                        ),
                        "drug_idx": int(
                            pair.drug_idx
                        ),
                        "drug_name": pair.drug_name,
                        "gene_idx": int(
                            pair.gene_idx
                        ),
                        "gene_name": pair.gene_name,
                        "pubmed_pair_evidence_class": (
                            pair.pubmed_pair_evidence_class
                        ),
                        "has_positive_pubmed_support": (
                            pair.has_positive_pubmed_support
                        ),
                        "relation_family": (
                            family_name
                        ),
                        "relation_members": (
                            member_names
                        ),
                        "removed_edge_count": (
                            removed_edge_count
                        ),
                        "baseline_score": (
                            baseline_score
                        ),
                        "ablated_score": (
                            ablated_score
                        ),
                        "score_drop": (
                            baseline_score
                            - ablated_score
                        ),
                        "absolute_score_change": abs(
                            baseline_score
                            - ablated_score
                        ),
                        "baseline_rank": (
                            baseline_rank
                        ),
                        "ablated_rank": (
                            ablated_rank
                        ),
                        "rank_worsening": (
                            ablated_rank
                            - baseline_rank
                        ),
                        "absolute_rank_change": abs(
                            ablated_rank
                            - baseline_rank
                        ),
                    }
                )

            del ablated_embeddings

            if device.type == "cuda":
                torch.cuda.empty_cache()

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    results = pd.DataFrame(output_rows)

    results["supportive_score_effect"] = (
        results["score_drop"] > 0
    )

    results["supportive_rank_effect"] = (
        results["rank_worsening"] > 0
    )

    results["score_drop_rank_within_pair_seed"] = (
        results.groupby(
            [
                "seed",
                "external_validation_rank",
            ]
        )["score_drop"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    results.to_csv(
        LONG_OUTPUT,
        index=False,
    )

    pair_summary = (
        results.groupby(
            [
                "external_validation_rank",
                "drug_name",
                "gene_name",
                "pubmed_pair_evidence_class",
                "relation_family",
                "relation_members",
            ],
            as_index=False,
        )
        .agg(
            eligible_seed_count=(
                "seed",
                "nunique",
            ),
            mean_score_drop=(
                "score_drop",
                "mean",
            ),
            minimum_score_drop=(
                "score_drop",
                "min",
            ),
            maximum_score_drop=(
                "score_drop",
                "max",
            ),
            mean_rank_worsening=(
                "rank_worsening",
                "mean",
            ),
            maximum_rank_worsening=(
                "rank_worsening",
                "max",
            ),
            supportive_score_seed_fraction=(
                "supportive_score_effect",
                "mean",
            ),
            supportive_rank_seed_fraction=(
                "supportive_rank_effect",
                "mean",
            ),
            mean_within_pair_seed_rank=(
                "score_drop_rank_within_pair_seed",
                "mean",
            ),
        )
    )

    pair_summary[
        "relation_rank_within_pair"
    ] = (
        pair_summary.groupby(
            "external_validation_rank"
        )["mean_score_drop"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    pair_summary = pair_summary.sort_values(
        [
            "external_validation_rank",
            "relation_rank_within_pair",
        ]
    )

    pair_summary.to_csv(
        PAIR_OUTPUT,
        index=False,
    )

    global_summary = (
        results.groupby(
            [
                "relation_family",
                "relation_members",
            ],
            as_index=False,
        )
        .agg(
            pair_seed_evaluations=(
                "score_drop",
                "size",
            ),
            candidate_pairs=(
                "external_validation_rank",
                "nunique",
            ),
            mean_score_drop=(
                "score_drop",
                "mean",
            ),
            median_score_drop=(
                "score_drop",
                "median",
            ),
            mean_absolute_score_change=(
                "absolute_score_change",
                "mean",
            ),
            mean_rank_worsening=(
                "rank_worsening",
                "mean",
            ),
            median_rank_worsening=(
                "rank_worsening",
                "median",
            ),
            supportive_score_fraction=(
                "supportive_score_effect",
                "mean",
            ),
            supportive_rank_fraction=(
                "supportive_rank_effect",
                "mean",
            ),
            mean_removed_edge_count=(
                "removed_edge_count",
                "mean",
            ),
        )
    )

    global_summary = global_summary.sort_values(
        [
            "mean_score_drop",
            "mean_rank_worsening",
        ],
        ascending=[
            False,
            False,
        ],
    )

    global_summary.to_csv(
        GLOBAL_OUTPUT,
        index=False,
    )

    print("\nGCN RELATION-FAMILY ABLATION COMPLETE")
    print(
        "Pair-seed evaluations:",
        results[
            [
                "seed",
                "external_validation_rank",
            ]
        ].drop_duplicates().shape[0],
    )
    print(
        "Relation families:",
        results["relation_family"].nunique(),
    )
    print("Rows:", len(results))

    print("\nTOP GLOBAL SUPPORTIVE RELATION FAMILIES")
    print(
        global_summary[
            [
                "relation_family",
                "mean_score_drop",
                "median_score_drop",
                "mean_rank_worsening",
                "supportive_score_fraction",
                "supportive_rank_fraction",
                "mean_removed_edge_count",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    print("\nTOP THREE RELATIONS PER PAIR")
    print(
        pair_summary.loc[
            pair_summary[
                "relation_rank_within_pair"
            ].le(3),
            [
                "external_validation_rank",
                "drug_name",
                "gene_name",
                "relation_rank_within_pair",
                "relation_family",
                "mean_score_drop",
                "mean_rank_worsening",
                "eligible_seed_count",
            ],
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(LONG_OUTPUT)
    print(PAIR_OUTPUT)
    print(GLOBAL_OUTPUT)


if __name__ == "__main__":
    main()
