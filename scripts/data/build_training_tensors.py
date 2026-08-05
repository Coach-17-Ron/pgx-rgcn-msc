#!/usr/bin/env python3
"""Build PyTorch tensor bundles for one cold-drug split."""

from __future__ import annotations

import argparse
from pathlib import Path

from pgx_rgcn.tensor_builder import build_tensor_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--protocol",
        choices=["dg_context", "strict_cold"],
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=Path("artifacts/graph"),
    )

    parser.add_argument(
        "--split-root",
        type=Path,
        default=Path("artifacts/splits"),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/tensors"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    split_dir = (
        args.split_root
        / args.protocol
        / f"seed_{args.seed}"
    )

    output_path = (
        args.output_root
        / args.protocol
        / f"seed_{args.seed}"
        / "tensor_bundle.pt"
    )

    bundle = build_tensor_bundle(
        graph_dir=args.graph_dir,
        split_dir=split_dir,
        output_path=output_path,
    )

    print("Saved:", output_path)
    print("Nodes:", bundle["num_nodes"])
    print("Relations:", bundle["num_relations"])
    print("Edges:", bundle["edge_index"].shape[1])
    print(
        "Train positives:",
        bundle["train_positive_pairs"].shape[0],
    )
    print(
        "Validation positives:",
        bundle["validation_positive_pairs"].shape[0],
    )
    print(
        "Test positives:",
        bundle["test_positive_pairs"].shape[0],
    )


if __name__ == "__main__":
    main()
