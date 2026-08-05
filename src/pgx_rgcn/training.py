"""Shared training loop for GCN, R-GCN and RGAT baselines."""

from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from pgx_rgcn.evaluation import evaluate_filtered_ranking
from pgx_rgcn.models import PGxGraphModel, count_parameters
from pgx_rgcn.sampling import sample_negative_genes


def set_reproducible_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(
    value: dict[str, Any],
    path: Path,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            sort_keys=True,
        )


def choose_device(
    requested: str,
) -> torch.device:
    """Resolve CPU or CUDA device."""
    requested = requested.lower()

    if requested == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but no GPU is available in this session."
        )

    if requested not in {"cpu", "cuda"}:
        raise ValueError(
            "device must be auto, cpu or cuda."
        )

    return torch.device(requested)


def train_one_run(
    *,
    model_name: str,
    protocol: str,
    seed: int,
    tensor_path: Path,
    config_path: Path,
    output_dir: Path,
    device_name: str = "auto",
    maximum_epochs: int | None = None,
) -> dict[str, Any]:
    """Train one model on one protocol and seed."""
    config = load_json(config_path)

    if model_name not in config["models"]:
        raise ValueError(
            f"Model {model_name!r} is not listed in the configuration."
        )

    if protocol not in config["protocols"]:
        raise ValueError(
            f"Protocol {protocol!r} is not listed in the configuration."
        )

    set_reproducible_seed(seed)
    device = choose_device(device_name)

    bundle = torch.load(
        tensor_path,
        map_location="cpu",
        weights_only=False,
    )

    node_type = bundle["node_type"].to(device)
    edge_index = bundle["edge_index"].to(device)
    edge_type = bundle["edge_type"].to(device)

    train_positive_pairs_cpu = bundle[
        "train_positive_pairs"
    ].cpu()

    validation_pairs = bundle[
        "validation_positive_pairs"
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

    model = PGxGraphModel(
        model_name=model_name,
        num_nodes=bundle["num_nodes"],
        num_node_types=bundle["num_node_types"],
        num_relations=bundle["num_relations"],
        embedding_dim=int(config["embedding_dim"]),
        num_bases=int(config["num_bases"]),
        dropout=float(config["dropout"]),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    loss_function = nn.BCEWithLogitsLoss()

    configured_epochs = int(config["epochs"])
    epochs = (
        min(configured_epochs, int(maximum_epochs))
        if maximum_epochs is not None
        else configured_epochs
    )

    validation_interval = int(
        config["validation_interval"]
    )
    patience = int(config["patience"])
    negatives_per_positive = int(
        config["negatives_per_positive"]
    )

    negative_generator = torch.Generator(
        device="cpu"
    ).manual_seed(seed)

    best_validation_mrr = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    evaluations_without_improvement = 0

    history: list[dict[str, Any]] = []

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()

        negative_pairs_cpu = sample_negative_genes(
            positive_pairs=train_positive_pairs_cpu,
            gene_indices=gene_indices,
            known_positive_pairs=known_positive_pairs,
            generator=negative_generator,
            negatives_per_positive=negatives_per_positive,
        )

        positive_pairs = train_positive_pairs_cpu.to(device)
        negative_pairs = negative_pairs_cpu.to(device)

        optimizer.zero_grad(set_to_none=True)

        embeddings = model.encode(
            node_type=node_type,
            edge_index=edge_index,
            edge_type=edge_type,
        )

        positive_scores = model.score_pairs(
            embeddings=embeddings,
            pairs=positive_pairs,
        )

        negative_scores = model.score_pairs(
            embeddings=embeddings,
            pairs=negative_pairs,
        )

        scores = torch.cat(
            [positive_scores, negative_scores],
            dim=0,
        )

        labels = torch.cat(
            [
                torch.ones_like(positive_scores),
                torch.zeros_like(negative_scores),
            ],
            dim=0,
        )

        loss = loss_function(scores, labels)
        loss.backward()

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float(config["gradient_clip_norm"]),
        )

        optimizer.step()

        record: dict[str, Any] = {
            "epoch": epoch,
            "training_loss": float(loss.item()),
            "gradient_norm": float(gradient_norm),
            "validation_mrr": None,
            "validation_hits_at_1": None,
            "validation_hits_at_3": None,
            "validation_hits_at_10": None,
        }

        should_validate = (
            epoch == 1
            or epoch % validation_interval == 0
            or epoch == epochs
        )

        if should_validate:
            validation_metrics, _ = evaluate_filtered_ranking(
                model=model,
                node_type=node_type,
                edge_index=edge_index,
                edge_type=edge_type,
                evaluation_pairs=validation_pairs,
                gene_indices=gene_indices.to(device),
                known_positive_pairs=known_positive_pairs,
                query_batch_size=int(
                    config["query_batch_size"]
                ),
            )

            validation_mrr = float(
                validation_metrics["mrr"]
            )

            record.update(
                {
                    "validation_mrr": validation_mrr,
                    "validation_hits_at_1": float(
                        validation_metrics["hits_at_1"]
                    ),
                    "validation_hits_at_3": float(
                        validation_metrics["hits_at_3"]
                    ),
                    "validation_hits_at_10": float(
                        validation_metrics["hits_at_10"]
                    ),
                }
            )

            improved = (
                validation_mrr > best_validation_mrr
            )

            if improved:
                best_validation_mrr = validation_mrr
                best_epoch = epoch
                best_state = copy.deepcopy(
                    model.state_dict()
                )
                evaluations_without_improvement = 0
            else:
                evaluations_without_improvement += 1

            print(
                f"epoch={epoch:03d} "
                f"loss={loss.item():.6f} "
                f"val_mrr={validation_mrr:.6f} "
                f"best={best_validation_mrr:.6f} "
                f"patience={evaluations_without_improvement}/"
                f"{patience}"
            )

            if evaluations_without_improvement >= patience:
                print(
                    f"Early stopping at epoch {epoch}."
                )
                history.append(record)
                break

        elif epoch % 10 == 0:
            print(
                f"epoch={epoch:03d} "
                f"loss={loss.item():.6f}"
            )

        history.append(record)

    if best_state is None:
        raise RuntimeError(
            "Training completed without producing a checkpoint."
        )

    model.load_state_dict(best_state)

    elapsed_seconds = time.time() - start_time

    checkpoint = {
        "model_name": model_name,
        "protocol": protocol,
        "seed": int(seed),
        "model_state_dict": best_state,
        "num_nodes": int(bundle["num_nodes"]),
        "num_node_types": int(
            bundle["num_node_types"]
        ),
        "num_relations": int(
            bundle["num_relations"]
        ),
        "embedding_dim": int(
            config["embedding_dim"]
        ),
        "num_bases": int(config["num_bases"]),
        "dropout": float(config["dropout"]),
        "best_epoch": int(best_epoch),
        "best_validation_mrr": float(
            best_validation_mrr
        ),
        "relation_to_index": bundle[
            "relation_to_index"
        ],
        "node_type_to_index": bundle[
            "node_type_to_index"
        ],
    }

    torch.save(
        checkpoint,
        output_dir / "best_checkpoint.pt",
    )

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(
        output_dir / "training_history.csv",
        index=False,
    )

    summary = {
        "model": model_name,
        "protocol": protocol,
        "seed": int(seed),
        "device": str(device),
        "trainable_parameters": int(
            count_parameters(model)
        ),
        "epochs_completed": int(
            history_frame["epoch"].max()
        ),
        "best_epoch": int(best_epoch),
        "best_validation_mrr": float(
            best_validation_mrr
        ),
        "elapsed_seconds": float(elapsed_seconds),
        "training_positive_pairs": int(
            len(train_positive_pairs_cpu)
        ),
        "validation_positive_pairs": int(
            len(validation_pairs)
        ),
    }

    save_json(
        summary,
        output_dir / "training_summary.json",
    )

    print("\nTraining complete")
    print("Model:", model_name)
    print("Protocol:", protocol)
    print("Seed:", seed)
    print("Device:", device)
    print("Best epoch:", best_epoch)
    print(
        "Best validation MRR:",
        f"{best_validation_mrr:.6f}",
    )
    print(
        "Elapsed minutes:",
        f"{elapsed_seconds / 60:.2f}",
    )

    return summary
