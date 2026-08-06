#!/usr/bin/env python
"""Extract RGAT edge attention and summarise it across strict-cold seeds."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from pgx_rgcn.models.graph_models import PGxGraphModel


SEEDS = [42, 123, 2026]

GRAPH_ROOT = Path("artifacts/graph")
TENSOR_ROOT = Path(
    "artifacts/tensors_molecular/strict_cold"
)
MODEL_ROOT = Path(
    "results/models/molecular/rgat/strict_cold"
)

SUMMARY_ROOT = Path(
    "results/consensus_inference/molecular/"
    "strict_cold/summary"
)

CANDIDATE_PATH = (
    SUMMARY_ROOT
    / "external_validation_candidate_pairs.csv"
)

EDGE_OUTPUT = (
    SUMMARY_ROOT
    / "rgat_attention_edges_by_seed_layer.csv"
)

RELATION_OUTPUT = (
    SUMMARY_ROOT
    / "rgat_attention_relation_summary.csv"
)

CANDIDATE_OUTPUT = (
    SUMMARY_ROOT
    / "rgat_candidate_incident_attention.csv"
)


def load_model(
    checkpoint: dict[str, object],
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
    model.eval()

    return model


def extract_attention(
    model: PGxGraphModel,
    bundle: dict[str, object],
) -> tuple[torch.Tensor, torch.Tensor]:
    node_type = bundle["node_type"]
    edge_index = bundle["edge_index"]
    edge_type = bundle["edge_type"]

    with torch.no_grad():
        x0 = model.features(
            node_type=node_type,
            molecular_features=(
                bundle["molecular_features"]
            ),
            has_molecular_features=(
                bundle["has_molecular_features"]
            ),
        )

        x1, (_, alpha1) = model.encoder.conv1(
            x0,
            edge_index,
            edge_type,
            return_attention_weights=True,
        )

        x1 = model.encoder.activation(x1)
        x1 = model.encoder.dropout(x1)

        _, (_, alpha2) = model.encoder.conv2(
            x1,
            edge_index,
            edge_type,
            return_attention_weights=True,
        )

    return (
        alpha1.reshape(-1).cpu(),
        alpha2.reshape(-1).cpu(),
    )


def main() -> None:
    edges = pd.read_parquet(
        GRAPH_ROOT / "edges.parquet"
    )

    nodes = pd.read_parquet(
        GRAPH_ROOT / "nodes.parquet"
    )[
        [
            "node_index",
            "node_id",
            "node_name",
            "node_type",
        ]
    ]

    candidates = pd.read_csv(
        CANDIDATE_PATH,
        low_memory=False,
    )[
        [
            "external_validation_rank",
            "drug_idx",
            "drug_name",
            "gene_idx",
            "gene_name",
        ]
    ].drop_duplicates()

    graph_metadata = edges[
        [
            "source_idx",
            "target_idx",
            "rel_idx",
            "relation_type",
            "evidence",
            "association",
            "pmids",
        ]
    ].copy()

    edge_frames: list[pd.DataFrame] = []

    for seed in SEEDS:
        print(f"Extracting RGAT attention: seed {seed}")

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

        model = load_model(checkpoint)

        alpha1, alpha2 = extract_attention(
            model,
            bundle,
        )

        tensor_edges = pd.DataFrame(
            {
                "source_idx": (
                    bundle["edge_index"][0]
                    .cpu()
                    .numpy()
                ),
                "target_idx": (
                    bundle["edge_index"][1]
                    .cpu()
                    .numpy()
                ),
                "rel_idx": (
                    bundle["edge_type"]
                    .cpu()
                    .numpy()
                ),
                "tensor_edge_position": range(
                    bundle["edge_index"].shape[1]
                ),
                "layer1_attention": alpha1.numpy(),
                "layer2_attention": alpha2.numpy(),
            }
        )

        aligned = tensor_edges.merge(
            graph_metadata,
            on=[
                "source_idx",
                "target_idx",
                "rel_idx",
            ],
            how="left",
            validate="one_to_one",
        )

        if aligned["relation_type"].isna().any():
            raise ValueError(
                f"Unmatched tensor edges for seed {seed}"
            )

        source_nodes = nodes.rename(
            columns={
                "node_index": "source_idx",
                "node_id": "source_id",
                "node_name": "source_name",
                "node_type": "source_type",
            }
        )

        target_nodes = nodes.rename(
            columns={
                "node_index": "target_idx",
                "node_id": "target_id",
                "node_name": "target_name",
                "node_type": "target_type",
            }
        )

        aligned = (
            aligned.merge(
                source_nodes,
                on="source_idx",
                how="left",
                validate="many_to_one",
            )
            .merge(
                target_nodes,
                on="target_idx",
                how="left",
                validate="many_to_one",
            )
        )

        long = aligned.melt(
            id_vars=[
                "tensor_edge_position",
                "source_idx",
                "target_idx",
                "rel_idx",
                "relation_type",
                "evidence",
                "association",
                "pmids",
                "source_id",
                "source_name",
                "source_type",
                "target_id",
                "target_name",
                "target_type",
            ],
            value_vars=[
                "layer1_attention",
                "layer2_attention",
            ],
            var_name="layer_name",
            value_name="attention",
        )

        long["seed"] = seed
        long["layer"] = (
            long["layer_name"]
            .map(
                {
                    "layer1_attention": 1,
                    "layer2_attention": 2,
                }
            )
            .astype(int)
        )

        long = long.drop(columns="layer_name")
        edge_frames.append(long)

    attention = pd.concat(
        edge_frames,
        ignore_index=True,
    )

    attention["attention_rank_within_seed_layer"] = (
        attention.groupby(
            ["seed", "layer"]
        )["attention"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    attention.to_csv(
        EDGE_OUTPUT,
        index=False,
    )

    relation_summary = (
        attention.groupby(
            [
                "seed",
                "layer",
                "relation_type",
            ],
            as_index=False,
        )
        .agg(
            edge_count=("attention", "size"),
            mean_attention=("attention", "mean"),
            median_attention=("attention", "median"),
            maximum_attention=("attention", "max"),
            total_attention_mass=("attention", "sum"),
        )
    )

    total_mass = (
        relation_summary.groupby(
            ["seed", "layer"]
        )["total_attention_mass"]
        .transform("sum")
    )

    relation_summary[
        "attention_mass_fraction"
    ] = (
        relation_summary["total_attention_mass"]
        / total_mass
    )

    across_seeds = (
        relation_summary.groupby(
            ["layer", "relation_type"],
            as_index=False,
        )
        .agg(
            mean_edge_count=("edge_count", "mean"),
            mean_attention=(
                "mean_attention",
                "mean",
            ),
            mean_median_attention=(
                "median_attention",
                "mean",
            ),
            mean_maximum_attention=(
                "maximum_attention",
                "mean",
            ),
            mean_attention_mass_fraction=(
                "attention_mass_fraction",
                "mean",
            ),
            std_attention_mass_fraction=(
                "attention_mass_fraction",
                "std",
            ),
        )
    )

    across_seeds[
        "attention_mass_rank_within_layer"
    ] = (
        across_seeds.groupby("layer")[
            "mean_attention_mass_fraction"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    across_seeds = across_seeds.sort_values(
        [
            "layer",
            "attention_mass_rank_within_layer",
        ]
    )

    across_seeds.to_csv(
        RELATION_OUTPUT,
        index=False,
    )

    incident_rows: list[pd.DataFrame] = []

    for pair in candidates.itertuples(index=False):
        incident = attention.loc[
            attention["source_idx"].isin(
                [int(pair.drug_idx), int(pair.gene_idx)]
            )
            | attention["target_idx"].isin(
                [int(pair.drug_idx), int(pair.gene_idx)]
            )
        ].copy()

        if incident.empty:
            continue

        incident[
            "external_validation_rank"
        ] = int(pair.external_validation_rank)

        incident["candidate_drug_name"] = (
            pair.drug_name
        )
        incident["candidate_gene_name"] = (
            pair.gene_name
        )

        incident["incident_to_candidate_drug"] = (
            incident["source_idx"].eq(
                int(pair.drug_idx)
            )
            | incident["target_idx"].eq(
                int(pair.drug_idx)
            )
        )

        incident["incident_to_candidate_gene"] = (
            incident["source_idx"].eq(
                int(pair.gene_idx)
            )
            | incident["target_idx"].eq(
                int(pair.gene_idx)
            )
        )

        incident_rows.append(incident)

    if incident_rows:
        candidate_attention = pd.concat(
            incident_rows,
            ignore_index=True,
        )

        candidate_attention[
            "attention_rank_within_pair_seed_layer"
        ] = (
            candidate_attention.groupby(
                [
                    "external_validation_rank",
                    "seed",
                    "layer",
                ]
            )["attention"]
            .rank(
                method="min",
                ascending=False,
            )
            .astype(int)
        )
    else:
        candidate_attention = pd.DataFrame()

    candidate_attention.to_csv(
        CANDIDATE_OUTPUT,
        index=False,
    )

    print("\nRGAT ATTENTION EXTRACTION COMPLETE")
    print("Edge-attention rows:", len(attention))
    print(
        "Relation summaries:",
        len(across_seeds),
    )
    print(
        "Candidate incident rows:",
        len(candidate_attention),
    )

    for layer in [1, 2]:
        print(
            f"\nTOP 15 RELATIONS BY ATTENTION MASS — "
            f"LAYER {layer}"
        )

        print(
            across_seeds.loc[
                across_seeds["layer"].eq(layer),
                [
                    "relation_type",
                    "mean_edge_count",
                    "mean_attention",
                    "mean_attention_mass_fraction",
                    "attention_mass_rank_within_layer",
                ],
            ]
            .head(15)
            .to_string(index=False)
        )

    print("\nCANDIDATE DRUG INCIDENT-EDGE COVERAGE")

    if candidate_attention.empty:
        print("No candidate incident edges found.")
    else:
        coverage = (
            candidate_attention.groupby(
                "external_validation_rank",
                as_index=False,
            )
            .agg(
                drug_incident_rows=(
                    "incident_to_candidate_drug",
                    "sum",
                ),
                gene_incident_rows=(
                    "incident_to_candidate_gene",
                    "sum",
                ),
            )
        )

        print(coverage.to_string(index=False))

    print("\nSaved:")
    print(EDGE_OUTPUT)
    print(RELATION_OUTPUT)
    print(CANDIDATE_OUTPUT)


if __name__ == "__main__":
    main()
