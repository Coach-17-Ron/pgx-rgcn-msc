#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from pgx_rgcn.evaluation import evaluate_filtered_ranking
from pgx_rgcn.models.graph_models import PGxGraphModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved graph model on the untouched test set."
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
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )

    parser.add_argument(
        "--feature-mode",
        choices=["id", "molecular"],
        default="id",
    )
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=64,
    )

    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but no GPU is available."
        )

    return torch.device(requested)


def make_json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]

    return value


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    tensor_path = (
        args.tensor_root
        / args.protocol
        / f"seed_{args.seed}"
        / "tensor_bundle.pt"
    )

    run_dir = (
        args.model_root
        / args.feature_mode
        / args.model
        / args.protocol
        / f"seed_{args.seed}"
    )

    checkpoint_path = run_dir / "best_checkpoint.pt"

    if not tensor_path.exists():
        raise FileNotFoundError(
            f"Tensor bundle not found: {tensor_path}"
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
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

    checkpoint_feature_mode = checkpoint.get(
        "feature_mode",
        "id",
    )

    if checkpoint_feature_mode != args.feature_mode:
        raise ValueError(
            "Requested feature mode does not match checkpoint: "
            f"{args.feature_mode!r} versus "
            f"{checkpoint_feature_mode!r}."
        )

    model = PGxGraphModel(
        model_name=checkpoint["model_name"],
        num_nodes=int(checkpoint["num_nodes"]),
        num_node_types=int(checkpoint["num_node_types"]),
        num_relations=int(checkpoint["num_relations"]),
        embedding_dim=int(checkpoint["embedding_dim"]),
        num_bases=int(checkpoint["num_bases"]),
        dropout=float(checkpoint["dropout"]),
        feature_mode=checkpoint_feature_mode,
        molecular_feature_dim=checkpoint.get(
            "molecular_feature_dim"
        ),
        drug_type_index=int(
            checkpoint["node_type_to_index"]["drug"]
        ),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    node_type = bundle["node_type"].to(device)
    edge_index = bundle["edge_index"].to(device)
    edge_type = bundle["edge_type"].to(device)

    molecular_features = None
    has_molecular_features = None

    if args.feature_mode == "molecular":
        molecular_features = bundle.get(
            "molecular_features"
        )
        has_molecular_features = bundle.get(
            "has_molecular_features"
        )

        if (
            molecular_features is None
            or has_molecular_features is None
        ):
            raise ValueError(
                "Molecular tensors are missing from the bundle."
            )

        molecular_features = molecular_features.to(device)
        has_molecular_features = (
            has_molecular_features.to(device)
        )

    test_pairs = bundle[
        "test_positive_pairs"
    ].to(device)

    known_positive_pairs = bundle[
        "all_known_positive_pairs"
    ].cpu()

    gene_type_index = bundle[
        "node_type_to_index"
    ]["gene"]

    gene_indices = torch.where(
        bundle["node_type"] == gene_type_index
    )[0].long()

    with torch.no_grad():
        metrics, ranks = evaluate_filtered_ranking(
            model=model,
            node_type=node_type,
            edge_index=edge_index,
            edge_type=edge_type,
            evaluation_pairs=test_pairs,
            gene_indices=gene_indices,
            known_positive_pairs=known_positive_pairs,
            query_batch_size=args.query_batch_size,
            molecular_features=molecular_features,
            has_molecular_features=has_molecular_features,
        )

    output = {
        "model": args.model,
        "feature_mode": args.feature_mode,
        "protocol": args.protocol,
        "seed": args.seed,
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_validation_mrr": float(
            checkpoint["best_validation_mrr"]
        ),
        "test_positive_pairs": int(test_pairs.shape[0]),
        **metrics,
    }

    metrics_path = run_dir / "test_metrics.json"
    ranks_path = run_dir / "test_ranks.pt"

    with metrics_path.open("w") as handle:
        json.dump(
            make_json_safe(output),
            handle,
            indent=2,
            sort_keys=True,
        )

    torch.save(ranks.detach().cpu(), ranks_path)

    print("Test evaluation complete")
    print(f"Model: {args.model}")
    print(f"Feature mode: {args.feature_mode}")
    print(f"Protocol: {args.protocol}")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")
    print(f"Test pairs: {test_pairs.shape[0]}")

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    print(f"Metrics saved: {metrics_path}")
    print(f"Ranks saved: {ranks_path}")


if __name__ == "__main__":
    main()
