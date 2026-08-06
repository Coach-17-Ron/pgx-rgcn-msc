"""Build reproducible PyTorch tensors for PGx graph experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
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


def build_molecular_feature_tensors(
    *,
    nodes: pd.DataFrame,
    fingerprint_path: Path,
    metadata_path: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align drug fingerprints to the full graph node index space."""
    fingerprints = np.load(fingerprint_path)

    metadata = pd.read_parquet(metadata_path).sort_values(
        "fingerprint_row"
    )

    required_metadata_columns = {
        "node_index",
        "node_id",
        "has_fingerprint",
        "fingerprint_row",
    }

    missing_columns = required_metadata_columns.difference(
        metadata.columns
    )

    if missing_columns:
        raise ValueError(
            "Molecular metadata missing columns: "
            f"{sorted(missing_columns)}"
        )

    if len(metadata) != fingerprints.shape[0]:
        raise AssertionError(
            "Fingerprint matrix row count does not match metadata."
        )

    expected_rows = list(range(len(metadata)))
    observed_rows = (
        metadata["fingerprint_row"]
        .astype(int)
        .tolist()
    )

    if observed_rows != expected_rows:
        raise AssertionError(
            "Fingerprint rows must be contiguous and ordered."
        )

    graph_drugs = (
        nodes.loc[
            nodes["node_type"] == "drug",
            ["node_index", "node_id"],
        ]
        .sort_values("node_index")
        .reset_index(drop=True)
    )

    aligned_metadata = (
        metadata[
            [
                "node_index",
                "node_id",
                "has_fingerprint",
            ]
        ]
        .sort_values("node_index")
        .reset_index(drop=True)
    )

    if not graph_drugs.equals(
        aligned_metadata[
            ["node_index", "node_id"]
        ]
    ):
        raise AssertionError(
            "Molecular metadata does not align with graph drug nodes."
        )

    num_nodes = len(nodes)
    feature_dim = int(fingerprints.shape[1])

    full_features = np.zeros(
        (num_nodes, feature_dim),
        dtype=np.uint8,
    )

    full_mask = np.zeros(
        num_nodes,
        dtype=bool,
    )

    drug_node_indices = aligned_metadata[
        "node_index"
    ].astype(int).to_numpy()

    full_features[drug_node_indices] = fingerprints.astype(
        np.uint8,
        copy=False,
    )

    full_mask[drug_node_indices] = aligned_metadata[
        "has_fingerprint"
    ].astype(bool).to_numpy()

    return (
        torch.from_numpy(full_features),
        torch.from_numpy(full_mask),
    )


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
    molecular_feature_dir: Path | None = None,
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

    molecular_features = None
    has_molecular_features = None

    if molecular_feature_dir is not None:
        molecular_features, has_molecular_features = (
            build_molecular_feature_tensors(
                nodes=nodes,
                fingerprint_path=(
                    molecular_feature_dir
                    / "morgan_radius2_2048.npy"
                ),
                metadata_path=(
                    molecular_feature_dir
                    / "morgan_radius2_2048_metadata.parquet"
                ),
            )
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
        "molecular_features": molecular_features,
        "has_molecular_features": has_molecular_features,
        "molecular_feature_dim": (
            int(molecular_features.shape[1])
            if molecular_features is not None
            else None
        ),
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
