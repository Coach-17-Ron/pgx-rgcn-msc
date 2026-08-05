"""Build reproducible PyTorch tensors for PGx graph experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch


NODE_TYPE_ORDER = [
    "drug",
    "gene",
    "variant",
    "phenotype",
]


def load_graph_inputs(
    graph_dir: Path,
    split_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_parquet(graph_dir / "nodes.parquet")
    train_edges = pd.read_parquet(split_dir / "train_edges.parquet")
    validation_pairs = pd.read_parquet(
        split_dir / "validation_pairs.parquet"
    )
    test_pairs = pd.read_parquet(
        split_dir / "test_pairs.parquet"
    )

    return nodes, train_edges, validation_pairs, test_pairs


def validate_nodes(nodes: pd.DataFrame) -> None:
    required = {
        "node_index",
        "node_type",
    }

    missing = required.difference(nodes.columns)
    if missing:
        raise ValueError(
            f"Node table missing columns: {sorted(missing)}"
        )

    ordered = nodes.sort_values("node_index")

    expected = list(range(len(ordered)))
    observed = ordered["node_index"].astype(int).tolist()

    if observed != expected:
        raise AssertionError(
            "Node indices must be contiguous from 0 to N-1."
        )

    unknown_types = set(nodes["node_type"]) - set(NODE_TYPE_ORDER)

    if unknown_types:
        raise ValueError(
            f"Unknown node types: {sorted(unknown_types)}"
        )


def validate_edges(
    edges: pd.DataFrame,
    num_nodes: int,
) -> None:
    required = {
        "source_idx",
        "target_idx",
        "relation_type",
    }

    missing = required.difference(edges.columns)
    if missing:
        raise ValueError(
            f"Edge table missing columns: {sorted(missing)}"
        )

    minimum_index = min(
        int(edges["source_idx"].min()),
        int(edges["target_idx"].min()),
    )

    maximum_index = max(
        int(edges["source_idx"].max()),
        int(edges["target_idx"].max()),
    )

    if minimum_index < 0 or maximum_index >= num_nodes:
        raise AssertionError(
            "One or more edge endpoints fall outside the node range."
        )


def build_relation_mapping(
    edges: pd.DataFrame,
) -> dict[str, int]:
    relations = sorted(
        str(value)
        for value in edges["relation_type"].unique()
    )

    return {
        relation: index
        for index, relation in enumerate(relations)
    }


def build_node_type_mapping() -> dict[str, int]:
    return {
        node_type: index
        for index, node_type in enumerate(NODE_TYPE_ORDER)
    }


def pairs_to_tensor(
    pairs: pd.DataFrame,
) -> torch.Tensor:
    if pairs.empty:
        return torch.empty((0, 2), dtype=torch.long)

    return torch.tensor(
        pairs[
            ["drug_idx", "gene_idx"]
        ].astype("int64").to_numpy(),
        dtype=torch.long,
    )


def build_known_positive_pairs(
    graph_dir: Path,
) -> torch.Tensor:
    edges = pd.read_parquet(graph_dir / "edges.parquet")

    drug_gene = edges[
        (
            (edges["source_type"] == "drug")
            & (edges["target_type"] == "gene")
        )
        |
        (
            (edges["source_type"] == "gene")
            & (edges["target_type"] == "drug")
        )
    ].copy()

    drug_gene["drug_idx"] = drug_gene["source_idx"].where(
        drug_gene["source_type"] == "drug",
        drug_gene["target_idx"],
    )

    drug_gene["gene_idx"] = drug_gene["target_idx"].where(
        drug_gene["target_type"] == "gene",
        drug_gene["source_idx"],
    )

    positives = (
        drug_gene[
            drug_gene["association"] == "associated"
        ][
            ["drug_idx", "gene_idx"]
        ]
        .drop_duplicates()
        .sort_values(["drug_idx", "gene_idx"])
    )

    return torch.tensor(
        positives.astype("int64").to_numpy(),
        dtype=torch.long,
    )


def build_training_positive_pairs(
    train_edges: pd.DataFrame,
) -> torch.Tensor:
    positives = train_edges[
        train_edges["is_drug_gene"]
        & (train_edges["association"] == "associated")
        & (
            train_edges["source_type"] == "drug"
        )
        & (
            train_edges["target_type"] == "gene"
        )
    ][
        ["source_idx", "target_idx"]
    ].drop_duplicates()

    positives.columns = [
        "drug_idx",
        "gene_idx",
    ]

    positives = positives.sort_values(
        ["drug_idx", "gene_idx"]
    )

    return torch.tensor(
        positives.astype("int64").to_numpy(),
        dtype=torch.long,
    )


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


def build_tensor_bundle(
    graph_dir: Path,
    split_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    nodes, train_edges, validation_pairs, test_pairs = (
        load_graph_inputs(
            graph_dir=graph_dir,
            split_dir=split_dir,
        )
    )

    validate_nodes(nodes)
    validate_edges(
        train_edges,
        num_nodes=len(nodes),
    )

    relation_to_index = build_relation_mapping(train_edges)
    node_type_to_index = build_node_type_mapping()

    edge_index = torch.tensor(
        train_edges[
            ["source_idx", "target_idx"]
        ].astype("int64").to_numpy().T,
        dtype=torch.long,
    )

    edge_type = torch.tensor(
        train_edges["relation_type"]
        .map(relation_to_index)
        .astype("int64")
        .to_numpy(),
        dtype=torch.long,
    )

    node_type = torch.tensor(
        nodes.sort_values("node_index")["node_type"]
        .map(node_type_to_index)
        .astype("int64")
        .to_numpy(),
        dtype=torch.long,
    )

    bundle = {
        "num_nodes": int(len(nodes)),
        "num_relations": int(len(relation_to_index)),
        "num_node_types": int(len(node_type_to_index)),
        "edge_index": edge_index,
        "edge_type": edge_type,
        "node_type": node_type,
        "train_positive_pairs": build_training_positive_pairs(
            train_edges
        ),
        "validation_positive_pairs": pairs_to_tensor(
            validation_pairs
        ),
        "test_positive_pairs": pairs_to_tensor(
            test_pairs
        ),
        "all_known_positive_pairs": build_known_positive_pairs(
            graph_dir
        ),
        "relation_to_index": relation_to_index,
        "node_type_to_index": node_type_to_index,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(bundle, output_path)

    save_json(
        relation_to_index,
        output_path.parent / "relation_to_index.json",
    )

    save_json(
        node_type_to_index,
        output_path.parent / "node_type_to_index.json",
    )

    return bundle
