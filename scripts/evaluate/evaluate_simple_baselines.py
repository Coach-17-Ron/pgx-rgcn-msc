#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate random-ranking and training gene-degree baselines "
            "using filtered drug-to-gene ranking."
        )
    )

    parser.add_argument(
        "--protocol",
        required=True,
        choices=["dg_context", "strict_cold"],
    )
    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        choices=[42, 123, 2026],
    )
    parser.add_argument(
        "--tensor-root",
        type=Path,
        default=Path("artifacts/tensors"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/baselines"),
    )

    return parser.parse_args()


def build_positive_lookup(
    known_positive_pairs: torch.Tensor,
) -> dict[int, set[int]]:
    lookup: dict[int, set[int]] = {}

    for drug_index, gene_index in known_positive_pairs.tolist():
        lookup.setdefault(int(drug_index), set()).add(
            int(gene_index)
        )

    return lookup


def metrics_from_ranks(
    ranks: torch.Tensor,
) -> dict[str, float | int]:
    ranks = ranks.float()

    return {
        "mrr": float((1.0 / ranks).mean().item()),
        "hits_at_1": float((ranks <= 1).float().mean().item()),
        "hits_at_3": float((ranks <= 3).float().mean().item()),
        "hits_at_10": float((ranks <= 10).float().mean().item()),
        "mean_rank": float(ranks.mean().item()),
        "median_rank": float(ranks.median().item()),
        "n_queries": int(ranks.numel()),
    }


def filtered_ranks_from_gene_scores(
    *,
    evaluation_pairs: torch.Tensor,
    gene_indices: torch.Tensor,
    known_positive_pairs: torch.Tensor,
    gene_scores: torch.Tensor,
) -> torch.Tensor:
    positive_lookup = build_positive_lookup(
        known_positive_pairs
    )

    gene_to_position = {
        int(gene_index): position
        for position, gene_index in enumerate(
            gene_indices.tolist()
        )
    }

    ranks: list[int] = []

    for drug_index, true_gene_index in evaluation_pairs.tolist():
        drug_index = int(drug_index)
        true_gene_index = int(true_gene_index)

        true_position = gene_to_position[true_gene_index]
        true_score = gene_scores[true_position]

        candidate_scores = gene_scores.clone()

        for known_gene_index in positive_lookup.get(
            drug_index,
            set(),
        ):
            if known_gene_index == true_gene_index:
                continue

            position = gene_to_position.get(known_gene_index)

            if position is not None:
                candidate_scores[position] = -torch.inf

        rank = 1 + int(
            (candidate_scores > true_score).sum().item()
        )

        ranks.append(rank)

    return torch.tensor(ranks, dtype=torch.long)


def evaluate_random_baseline(
    *,
    evaluation_pairs: torch.Tensor,
    gene_indices: torch.Tensor,
    known_positive_pairs: torch.Tensor,
    seed: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(
        seed
    )

    gene_scores = torch.rand(
        gene_indices.numel(),
        generator=generator,
    )

    ranks = filtered_ranks_from_gene_scores(
        evaluation_pairs=evaluation_pairs,
        gene_indices=gene_indices,
        known_positive_pairs=known_positive_pairs,
        gene_scores=gene_scores,
    )

    return metrics_from_ranks(ranks), ranks


def evaluate_gene_degree_baseline(
    *,
    train_positive_pairs: torch.Tensor,
    evaluation_pairs: torch.Tensor,
    gene_indices: torch.Tensor,
    known_positive_pairs: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor]:
    gene_to_position = {
        int(gene_index): position
        for position, gene_index in enumerate(
            gene_indices.tolist()
        )
    }

    gene_scores = torch.zeros(
        gene_indices.numel(),
        dtype=torch.float,
    )

    for gene_index in train_positive_pairs[:, 1].tolist():
        position = gene_to_position.get(int(gene_index))

        if position is not None:
            gene_scores[position] += 1.0

    ranks = filtered_ranks_from_gene_scores(
        evaluation_pairs=evaluation_pairs,
        gene_indices=gene_indices,
        known_positive_pairs=known_positive_pairs,
        gene_scores=gene_scores,
    )

    return metrics_from_ranks(ranks), ranks


def save_result(
    *,
    baseline: str,
    protocol: str,
    seed: int,
    metrics: dict[str, Any],
    ranks: torch.Tensor,
    output_root: Path,
) -> None:
    output_dir = (
        output_root
        / baseline
        / protocol
        / f"seed_{seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "model": baseline,
        "protocol": protocol,
        "seed": seed,
        **metrics,
    }

    metrics_path = output_dir / "test_metrics.json"
    ranks_path = output_dir / "test_ranks.pt"

    with metrics_path.open("w") as handle:
        json.dump(
            result,
            handle,
            indent=2,
            sort_keys=True,
        )

    torch.save(ranks.cpu(), ranks_path)

    print(
        f"{baseline:<12} "
        f"{protocol:<12} "
        f"seed={seed:<4} "
        f"MRR={metrics['mrr']:.6f} "
        f"H@10={metrics['hits_at_10']:.6f}"
    )


def main() -> None:
    args = parse_args()

    tensor_path = (
        args.tensor_root
        / args.protocol
        / f"seed_{args.seed}"
        / "tensor_bundle.pt"
    )

    bundle = torch.load(
        tensor_path,
        map_location="cpu",
        weights_only=False,
    )

    node_type = bundle["node_type"]
    gene_type_index = bundle[
        "node_type_to_index"
    ]["gene"]

    gene_indices = torch.where(
        node_type == gene_type_index
    )[0].long()

    evaluation_pairs = bundle[
        "test_positive_pairs"
    ].cpu()

    known_positive_pairs = bundle[
        "all_known_positive_pairs"
    ].cpu()

    train_positive_pairs = bundle[
        "train_positive_pairs"
    ].cpu()

    random_metrics, random_ranks = (
        evaluate_random_baseline(
            evaluation_pairs=evaluation_pairs,
            gene_indices=gene_indices,
            known_positive_pairs=known_positive_pairs,
            seed=args.seed,
        )
    )

    degree_metrics, degree_ranks = (
        evaluate_gene_degree_baseline(
            train_positive_pairs=train_positive_pairs,
            evaluation_pairs=evaluation_pairs,
            gene_indices=gene_indices,
            known_positive_pairs=known_positive_pairs,
        )
    )

    save_result(
        baseline="random",
        protocol=args.protocol,
        seed=args.seed,
        metrics=random_metrics,
        ranks=random_ranks,
        output_root=args.output_root,
    )

    save_result(
        baseline="gene_degree",
        protocol=args.protocol,
        seed=args.seed,
        metrics=degree_metrics,
        ranks=degree_ranks,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
