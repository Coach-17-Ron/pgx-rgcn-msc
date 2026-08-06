#!/usr/bin/env python3
"""Train one PGx graph-model baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from pgx_rgcn.training import train_one_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["gcn", "rgcn", "rgat"],
        required=True,
    )

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
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    parser.add_argument(
        "--feature-mode",
        choices=["id", "molecular"],
        default="id",
        help=(
            "Use learned node-ID features or molecular "
            "fingerprints for drug nodes."
        ),
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Optional limit for smoke tests.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/graph_model_baseline.json"
        ),
    )

    parser.add_argument(
        "--tensor-root",
        type=Path,
        default=Path("artifacts/tensors"),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/models"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tensor_path = (
        args.tensor_root
        / args.protocol
        / f"seed_{args.seed}"
        / "tensor_bundle.pt"
    )

    output_dir = (
        args.output_root
        / args.feature_mode
        / args.model
        / args.protocol
        / f"seed_{args.seed}"
    )

    train_one_run(
        model_name=args.model,
        protocol=args.protocol,
        seed=args.seed,
        tensor_path=tensor_path,
        config_path=args.config,
        output_dir=output_dir,
        device_name=args.device,
        maximum_epochs=args.max_epochs,
        feature_mode=args.feature_mode,
    )


if __name__ == "__main__":
    main()
