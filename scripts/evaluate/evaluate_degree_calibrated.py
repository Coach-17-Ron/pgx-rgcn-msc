#!/usr/bin/env python3
"""Validation-tuned gene-degree calibration for saved graph models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from pgx_rgcn.evaluation import (
    build_positive_lookup,
    metrics_from_ranks,
)
from pgx_rgcn.models.graph_models import PGxGraphModel


DEFAULT_ALPHAS = [
    -2.0,
    -1.0,
    -0.5,
    -0.25,
    -0.10,
    -0.05,
    0.0,
    0.05,
    0.10,
    0.25,
    0.5,
    1.0,
    2.0,
    3.0,
    4.0,
    6.0,
    8.0,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune a model-plus-gene-degree calibration weight on "
            "validation data, then evaluate once on the test set."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=["gcn", "rgcn", "rgat"],
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
        "--model-root",
        type=Path,
        default=Path("results/models"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/degree_calibrated"),
    )
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=64,
    )

    return parser.parse_args()


def standardise(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    standard_deviation = values.std(unbiased=False)

    if float(standard_deviation.item()) == 0.0:
        return torch.zeros_like(values)

    return (
        values - values.mean()
    ) / standard_deviation.clamp_min(1e-12)


def build_gene_degree_scores(
    *,
    train_positive_pairs: torch.Tensor,
    gene_indices: torch.Tensor,
) -> torch.Tensor:
    gene_to_position = {
        int(gene_index): position
        for position, gene_index in enumerate(
            gene_indices.tolist()
        )
    }

    degrees = torch.zeros(
        gene_indices.numel(),
        dtype=torch.float,
    )

    for gene_index in train_positive_pairs[:, 1].tolist():
        position = gene_to_position.get(int(gene_index))

        if position is not None:
            degrees[position] += 1.0

    return standardise(torch.log1p(degrees))


@torch.no_grad()
def calibrated_ranks(
    *,
    model: PGxGraphModel,
    node_type: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    evaluation_pairs: torch.Tensor,
    gene_indices: torch.Tensor,
    known_positive_pairs: torch.Tensor,
    degree_scores: torch.Tensor,
    alpha: float,
    query_batch_size: int,
) -> torch.Tensor:
    model.eval()

    embeddings = model.encode(
        node_type=node_type,
        edge_index=edge_index,
        edge_type=edge_type,
    )

    candidate_genes = gene_indices.to(
        embeddings.device
    )
    candidate_gene_embeddings = embeddings[
        candidate_genes
    ]

    degree_scores = degree_scores.to(
        embeddings.device
    )

    relation = model.decoder.relation

    positive_lookup = build_positive_lookup(
        known_positive_pairs
    )

    gene_to_position = {
        int(gene_index): position
        for position, gene_index in enumerate(
            candidate_genes.cpu().tolist()
        )
    }

    ranks: list[int] = []

    for start in range(
        0,
        evaluation_pairs.shape[0],
        query_batch_size,
    ):
        batch = evaluation_pairs[
            start:start + query_batch_size
        ].to(embeddings.device)

        for drug_index, true_gene_index in batch.tolist():
            drug_embedding = embeddings[int(drug_index)]

            model_scores = (
                drug_embedding.unsqueeze(0)
                * relation.unsqueeze(0)
                * candidate_gene_embeddings
            ).sum(dim=-1)

            model_scores = standardise(model_scores)

            scores = (
                model_scores
                + float(alpha) * degree_scores
            )

            true_position = gene_to_position.get(
                int(true_gene_index)
            )

            if true_position is None:
                raise AssertionError(
                    f"True gene {true_gene_index} is absent "
                    "from the candidate set."
                )

            true_score = scores[true_position].clone()

            for known_gene_index in positive_lookup.get(
                int(drug_index),
                set(),
            ):
                if known_gene_index == int(true_gene_index):
                    continue

                known_position = gene_to_position.get(
                    int(known_gene_index)
                )

                if known_position is not None:
                    scores[known_position] = -torch.inf

            rank = 1 + int(
                (scores > true_score).sum().item()
            )

            ranks.append(rank)

    return torch.tensor(
        ranks,
        dtype=torch.long,
    )


def load_model(
    *,
    checkpoint_path: Path,
) -> tuple[PGxGraphModel, dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model = PGxGraphModel(
        model_name=checkpoint["model_name"],
        num_nodes=int(checkpoint["num_nodes"]),
        num_node_types=int(
            checkpoint["num_node_types"]
        ),
        num_relations=int(checkpoint["num_relations"]),
        embedding_dim=int(checkpoint["embedding_dim"]),
        num_bases=int(checkpoint["num_bases"]),
        dropout=float(checkpoint["dropout"]),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    return model, checkpoint


def main() -> None:
    args = parse_args()

    tensor_path = (
        args.tensor_root
        / args.protocol
        / f"seed_{args.seed}"
        / "tensor_bundle.pt"
    )

    checkpoint_path = (
        args.model_root
        / args.model
        / args.protocol
        / f"seed_{args.seed}"
        / "best_checkpoint.pt"
    )

    bundle = torch.load(
        tensor_path,
        map_location="cpu",
        weights_only=False,
    )

    model, checkpoint = load_model(
        checkpoint_path=checkpoint_path
    )

    node_type = bundle["node_type"].cpu()
    edge_index = bundle["edge_index"].cpu()
    edge_type = bundle["edge_type"].cpu()

    validation_pairs = bundle[
        "validation_positive_pairs"
    ].cpu()

    test_pairs = bundle[
        "test_positive_pairs"
    ].cpu()

    known_positive_pairs = bundle[
        "all_known_positive_pairs"
    ].cpu()

    train_positive_pairs = bundle[
        "train_positive_pairs"
    ].cpu()

    gene_type_index = bundle[
        "node_type_to_index"
    ]["gene"]

    gene_indices = torch.where(
        node_type == gene_type_index
    )[0].long()

    degree_scores = build_gene_degree_scores(
        train_positive_pairs=train_positive_pairs,
        gene_indices=gene_indices,
    )

    validation_results: list[dict[str, Any]] = []

    for alpha in DEFAULT_ALPHAS:
        ranks = calibrated_ranks(
            model=model,
            node_type=node_type,
            edge_index=edge_index,
            edge_type=edge_type,
            evaluation_pairs=validation_pairs,
            gene_indices=gene_indices,
            known_positive_pairs=known_positive_pairs,
            degree_scores=degree_scores,
            alpha=alpha,
            query_batch_size=args.query_batch_size,
        )

        metrics = metrics_from_ranks(ranks)

        validation_results.append(
            {
                "alpha": alpha,
                **metrics,
            }
        )

        print(
            f"alpha={alpha:>5.2f} "
            f"validation_mrr={metrics['mrr']:.6f} "
            f"hits_at_10={metrics['hits_at_10']:.6f}"
        )

    best_validation = max(
        validation_results,
        key=lambda result: (
            result["mrr"],
            -abs(result["alpha"]),
        ),
    )

    selected_alpha = float(
        best_validation["alpha"]
    )

    test_ranks = calibrated_ranks(
        model=model,
        node_type=node_type,
        edge_index=edge_index,
        edge_type=edge_type,
        evaluation_pairs=test_pairs,
        gene_indices=gene_indices,
        known_positive_pairs=known_positive_pairs,
        degree_scores=degree_scores,
        alpha=selected_alpha,
        query_batch_size=args.query_batch_size,
    )

    test_metrics = metrics_from_ranks(test_ranks)

    output_dir = (
        args.output_root
        / args.model
        / args.protocol
        / f"seed_{args.seed}"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "model": args.model,
        "protocol": args.protocol,
        "seed": args.seed,
        "checkpoint": str(checkpoint_path),
        "best_epoch": int(checkpoint["best_epoch"]),
        "selected_alpha": selected_alpha,
        "validation_selection_metric": "mrr",
        "best_calibrated_validation_metrics": (
            best_validation
        ),
        "validation_grid": validation_results,
        "test_metrics": test_metrics,
    }

    metrics_path = (
        output_dir / "degree_calibrated_metrics.json"
    )
    ranks_path = (
        output_dir / "degree_calibrated_test_ranks.pt"
    )

    with metrics_path.open("w") as handle:
        json.dump(
            output,
            handle,
            indent=2,
            sort_keys=True,
        )

    torch.save(test_ranks, ranks_path)

    print("\nDegree-calibrated evaluation complete")
    print(f"Model: {args.model}")
    print(f"Protocol: {args.protocol}")
    print(f"Seed: {args.seed}")
    print(f"Selected alpha: {selected_alpha:.2f}")
    print(
        "Validation MRR: "
        f"{best_validation['mrr']:.6f}"
    )
    print(f"Test MRR: {test_metrics['mrr']:.6f}")
    print(
        "Test Hits@10: "
        f"{test_metrics['hits_at_10']:.6f}"
    )
    print(f"Saved: {metrics_path}")
    print(f"Saved: {ranks_path}")


if __name__ == "__main__":
    main()
