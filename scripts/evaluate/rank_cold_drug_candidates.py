#!/usr/bin/env python
"""Rank genes for strict-cold drugs using three-model consensus."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pgx_rgcn.models.graph_models import PGxGraphModel


MODELS = ("gcn", "rgcn", "rgat")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank all candidate genes for held-out drugs and "
            "combine GCN, R-GCN and RGAT rankings."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        choices=(42, 123, 2026),
    )
    parser.add_argument(
        "--protocol",
        default="strict_cold",
        choices=("strict_cold", "dg_context"),
    )
    parser.add_argument(
        "--tensor-root",
        type=Path,
        default=Path("artifacts/tensors"),
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=Path("artifacts/splits"),
    )
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=Path("artifacts/graph"),
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("results/models"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/consensus_inference"),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=250,
    )
    parser.add_argument(
        "--rrf-constant",
        type=float,
        default=60.0,
    )
    parser.add_argument(
        "--feature-mode",
        choices=("id", "molecular"),
        default="id",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
    )

    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but no GPU is available."
        )

    return torch.device(requested)


def load_model(
    model_name: str,
    protocol: str,
    seed: int,
    model_root: Path,
    feature_mode: str,
    device: torch.device,
) -> PGxGraphModel:
    checkpoint_root = (
        model_root / "molecular"
        if feature_mode == "molecular"
        else model_root
    )

    checkpoint_path = (
        checkpoint_root
        / model_name
        / protocol
        / f"seed_{seed}"
        / "best_checkpoint.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

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
        feature_mode=checkpoint.get(
            "feature_mode",
            "id",
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

    model = model.to(device)
    model.eval()

    return model


def ranks_from_scores(scores: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(
        scores,
        descending=True,
        stable=True,
    )

    ranks = torch.empty_like(
        order,
        dtype=torch.long,
    )

    ranks[order] = torch.arange(
        1,
        len(order) + 1,
        device=order.device,
    )

    return ranks


def build_pair_label_lookup(
    pair_assignments: pd.DataFrame,
) -> dict[tuple[int, int], dict[str, object]]:
    lookup: dict[
        tuple[int, int],
        dict[str, object],
    ] = {}

    for row in pair_assignments.itertuples(
        index=False
    ):
        lookup[
            (int(row.drug_idx), int(row.gene_idx))
        ] = {
            "association": str(row.association),
            "evidence_categories": "|".join(
                sorted(
                    str(value)
                    for value in row.evidence_categories
                )
            ),
            "original_split": str(row.split),
        }

    return lookup


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    tensor_root = (
        Path("artifacts/tensors_molecular")
        if args.feature_mode == "molecular"
        else args.tensor_root
    )

    tensor_path = (
        tensor_root
        / args.protocol
        / f"seed_{args.seed}"
        / "tensor_bundle.pt"
    )

    split_dir = (
        args.split_root
        / args.protocol
        / f"seed_{args.seed}"
    )

    bundle = torch.load(
        tensor_path,
        map_location="cpu",
        weights_only=False,
    )

    nodes = pd.read_parquet(
        args.graph_dir / "nodes.parquet"
    ).sort_values("node_index")

    assignments = pd.read_parquet(
        split_dir / "drug_assignments.parquet"
    )

    pair_assignments = pd.read_parquet(
        split_dir / "pair_assignments.parquet"
    )

    test_drugs = assignments.loc[
        assignments["split"].eq("test"),
        [
            "drug_idx",
            "drug_id",
            "drug_name",
            "associated_pair_count",
            "all_pair_count",
            "degree_stratum",
        ],
    ].sort_values("drug_idx")

    gene_type_index = bundle[
        "node_type_to_index"
    ]["gene"]

    gene_indices = torch.where(
        bundle["node_type"] == gene_type_index
    )[0].long()

    gene_nodes = (
        nodes.set_index("node_index")
        .loc[
            gene_indices.tolist(),
            ["node_id", "node_name"],
        ]
        .reset_index()
        .rename(
            columns={
                "node_index": "gene_idx",
                "node_id": "gene_id",
                "node_name": "gene_name",
            }
        )
    )

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
                "Molecular feature tensors are missing."
            )

        molecular_features = molecular_features.to(device)
        has_molecular_features = (
            has_molecular_features.to(device)
        )
    candidate_genes = gene_indices.to(device)

    rank_matrices: dict[str, torch.Tensor] = {}
    score_matrices: dict[str, torch.Tensor] = {}

    drug_indices = torch.tensor(
        test_drugs["drug_idx"].to_numpy(),
        dtype=torch.long,
        device=device,
    )

    for model_name in MODELS:
        print(
            f"Encoding {model_name} "
            f"seed={args.seed}..."
        )

        model = load_model(
            model_name=model_name,
            protocol=args.protocol,
            seed=args.seed,
            model_root=args.model_root,
            feature_mode=args.feature_mode,
            device=device,
        )

        with torch.no_grad():
            embeddings = model.encode(
                node_type=node_type,
                edge_index=edge_index,
                edge_type=edge_type,
                molecular_features=molecular_features,
                has_molecular_features=has_molecular_features,
            )

            drug_embeddings = embeddings[
                drug_indices
            ]

            gene_embeddings = embeddings[
                candidate_genes
            ]

            relation = model.decoder.relation

            scores = (
                drug_embeddings[:, None, :]
                * relation[None, None, :]
                * gene_embeddings[None, :, :]
            ).sum(dim=-1)

            ranks = torch.stack(
                [
                    ranks_from_scores(row)
                    for row in scores
                ],
                dim=0,
            )

        score_matrices[model_name] = (
            scores.detach().cpu()
        )
        rank_matrices[model_name] = (
            ranks.detach().cpu()
        )

        del model
        del embeddings

        if device.type == "cuda":
            torch.cuda.empty_cache()

    pair_lookup = build_pair_label_lookup(
        pair_assignments
    )

    checkpoint_root = (
        args.model_root / "molecular"
        if args.feature_mode == "molecular"
        else args.model_root
    )

    validation_mrr = {}

    for model_name in MODELS:
        checkpoint_path = (
            checkpoint_root
            / model_name
            / args.protocol
            / f"seed_{args.seed}"
            / "best_checkpoint.pt"
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        validation_mrr[model_name] = float(
            checkpoint["best_validation_mrr"]
        )

    validation_total = sum(
        validation_mrr.values()
    )

    if validation_total <= 0:
        raise ValueError(
            "Validation MRR values must sum to a "
            "positive number."
        )

    model_weights = {
        model_name: (
            validation_mrr[model_name]
            / validation_total
        )
        for model_name in MODELS
    }

    print("\nValidation-derived model weights")

    for model_name in MODELS:
        print(
            f"{model_name}: "
            f"validation_mrr="
            f"{validation_mrr[model_name]:.6f}, "
            f"weight={model_weights[model_name]:.6f}"
        )

    output_rows: list[dict[str, object]] = []
    labelled_rows: list[dict[str, object]] = []

    gene_idx_values = gene_nodes[
        "gene_idx"
    ].to_numpy()

    for drug_position, drug in enumerate(
        test_drugs.itertuples(index=False)
    ):
        equal_rrf_score = np.zeros(
            len(gene_indices),
            dtype=np.float64,
        )

        weighted_rrf_score = np.zeros(
            len(gene_indices),
            dtype=np.float64,
        )

        for model_name in MODELS:
            model_ranks = (
                rank_matrices[model_name][
                    drug_position
                ]
                .numpy()
            )

            reciprocal_rank = (
                1.0
                / (
                    args.rrf_constant
                    + model_ranks
                )
            )

            equal_rrf_score += reciprocal_rank

            weighted_rrf_score += (
                model_weights[model_name]
                * reciprocal_rank
            )

        equal_order = np.argsort(
            -equal_rrf_score,
            kind="stable",
        )

        equal_rrf_rank = np.empty(
            len(equal_order),
            dtype=np.int64,
        )

        equal_rrf_rank[equal_order] = np.arange(
            1,
            len(equal_order) + 1,
        )

        consensus_order = np.argsort(
            -weighted_rrf_score,
            kind="stable",
        )

        consensus_rank = np.empty(
            len(consensus_order),
            dtype=np.int64,
        )

        consensus_rank[consensus_order] = (
            np.arange(
                1,
                len(consensus_order) + 1,
            )
        )

        for gene_position in consensus_order[
            : args.top_k
        ]:
            gene_idx = int(
                gene_idx_values[gene_position]
            )

            label = pair_lookup.get(
                (int(drug.drug_idx), gene_idx)
            )

            association = (
                label["association"]
                if label is not None
                else "unlabelled"
            )

            evidence = (
                label["evidence_categories"]
                if label is not None
                else ""
            )

            ranks = {
                model_name: int(
                    rank_matrices[model_name][
                        drug_position,
                        gene_position,
                    ].item()
                )
                for model_name in MODELS
            }

            scores = {
                model_name: float(
                    score_matrices[model_name][
                        drug_position,
                        gene_position,
                    ].item()
                )
                for model_name in MODELS
            }

            gene = gene_nodes.iloc[
                gene_position
            ]

            output_rows.append(
                {
                    "seed": args.seed,
                    "feature_mode": args.feature_mode,
                    "feature_mode": args.feature_mode,
                    "protocol": args.protocol,
                    "drug_idx": int(
                        drug.drug_idx
                    ),
                    "drug_id": drug.drug_id,
                    "drug_name": drug.drug_name,
                    "gene_idx": gene_idx,
                    "gene_id": gene.gene_id,
                    "gene_name": gene.gene_name,
                    "association": association,
                    "evidence_categories": evidence,
                    "consensus_rank": int(
                        consensus_rank[
                            gene_position
                        ]
                    ),
                    "rrf_score": float(
                        weighted_rrf_score[
                            gene_position
                        ]
                    ),
                    "weighted_consensus_rank": int(
                        consensus_rank[
                            gene_position
                        ]
                    ),
                    "weighted_rrf_score": float(
                        weighted_rrf_score[
                            gene_position
                        ]
                    ),
                    "equal_rrf_rank": int(
                        equal_rrf_rank[
                            gene_position
                        ]
                    ),
                    "equal_rrf_score": float(
                        equal_rrf_score[
                            gene_position
                        ]
                    ),
                    "gcn_rank": ranks["gcn"],
                    "rgcn_rank": ranks["rgcn"],
                    "rgat_rank": ranks["rgat"],
                    "gcn_score": scores["gcn"],
                    "rgcn_score": scores["rgcn"],
                    "rgat_score": scores["rgat"],
                    "models_top_10": int(
                        sum(
                            rank <= 10
                            for rank in ranks.values()
                        )
                    ),
                    "models_top_25": int(
                        sum(
                            rank <= 25
                            for rank in ranks.values()
                        )
                    ),
                    "models_top_100": int(
                        sum(
                            rank <= 100
                            for rank in ranks.values()
                        )
                    ),
                }
            )

        labelled_for_drug = pair_assignments.loc[
            pair_assignments["drug_idx"].eq(
                int(drug.drug_idx)
            )
        ]

        gene_position_lookup = {
            int(gene_idx): position
            for position, gene_idx in enumerate(
                gene_idx_values
            )
        }

        for labelled in labelled_for_drug.itertuples(
            index=False
        ):
            gene_position = gene_position_lookup[
                int(labelled.gene_idx)
            ]

            gene = gene_nodes.iloc[
                gene_position
            ]

            labelled_rows.append(
                {
                    "seed": args.seed,
                    "feature_mode": args.feature_mode,
                    "protocol": args.protocol,
                    "drug_idx": int(
                        drug.drug_idx
                    ),
                    "drug_id": drug.drug_id,
                    "drug_name": drug.drug_name,
                    "gene_idx": int(
                        labelled.gene_idx
                    ),
                    "gene_id": gene.gene_id,
                    "gene_name": gene.gene_name,
                    "association": str(
                        labelled.association
                    ),
                    "evidence_categories": "|".join(
                        sorted(
                            str(value)
                            for value
                            in labelled.evidence_categories
                        )
                    ),
                    "consensus_rank": int(
                        consensus_rank[
                            gene_position
                        ]
                    ),
                    "rrf_score": float(
                        weighted_rrf_score[
                            gene_position
                        ]
                    ),
                    "weighted_consensus_rank": int(
                        consensus_rank[
                            gene_position
                        ]
                    ),
                    "weighted_rrf_score": float(
                        weighted_rrf_score[
                            gene_position
                        ]
                    ),
                    "equal_rrf_rank": int(
                        equal_rrf_rank[
                            gene_position
                        ]
                    ),
                    "equal_rrf_score": float(
                        equal_rrf_score[
                            gene_position
                        ]
                    ),
                    "gcn_rank": int(
                        rank_matrices["gcn"][
                            drug_position,
                            gene_position,
                        ].item()
                    ),
                    "rgcn_rank": int(
                        rank_matrices["rgcn"][
                            drug_position,
                            gene_position,
                        ].item()
                    ),
                    "rgat_rank": int(
                        rank_matrices["rgat"][
                            drug_position,
                            gene_position,
                        ].item()
                    ),
                }
            )

    candidates = pd.DataFrame(output_rows)
    labelled = pd.DataFrame(labelled_rows)

    output_dir = (
        args.output_root
        / args.feature_mode
        / args.protocol
        / f"seed_{args.seed}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates_path = (
        output_dir
        / "top_consensus_candidates.csv"
    )

    labelled_path = (
        output_dir
        / "labelled_pair_rankings.csv"
    )

    ambiguous_path = (
        output_dir
        / "ambiguous_pair_priorities.csv"
    )

    associated_path = (
        output_dir
        / "known_associated_recovery.csv"
    )

    unlabelled_path = (
        output_dir
        / "unlabelled_hypotheses.csv"
    )

    candidates.to_csv(
        candidates_path,
        index=False,
    )

    labelled.to_csv(
        labelled_path,
        index=False,
    )

    labelled.loc[
        labelled["association"].eq("ambiguous")
    ].sort_values(
        [
            "consensus_rank",
            "drug_name",
            "gene_name",
        ]
    ).to_csv(
        ambiguous_path,
        index=False,
    )

    labelled.loc[
        labelled["association"].eq("associated")
    ].sort_values(
        [
            "consensus_rank",
            "drug_name",
            "gene_name",
        ]
    ).to_csv(
        associated_path,
        index=False,
    )

    candidates.loc[
        candidates["association"].eq("unlabelled")
    ].sort_values(
        [
            "consensus_rank",
            "models_top_25",
            "rrf_score",
        ],
        ascending=[True, False, False],
    ).to_csv(
        unlabelled_path,
        index=False,
    )

    print("\nCompleted candidate ranking")
    print("Seed:", args.seed)
    print("Held-out drugs:", len(test_drugs))
    print("Candidate genes:", len(gene_indices))
    print("Top rows:", len(candidates))
    print("Labelled pairs:", len(labelled))
    print(
        "Ambiguous pairs:",
        int(
            labelled["association"]
            .eq("ambiguous")
            .sum()
        ),
    )
    print("Saved:", candidates_path)
    print("Saved:", labelled_path)
    print("Saved:", ambiguous_path)
    print("Saved:", associated_path)
    print("Saved:", unlabelled_path)


if __name__ == "__main__":
    main()
