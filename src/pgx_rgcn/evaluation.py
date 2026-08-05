"""Filtered drug-to-gene ranking evaluation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from pgx_rgcn.models import PGxGraphModel


def build_positive_lookup(
    known_positive_pairs: torch.Tensor,
) -> dict[int, set[int]]:
    """Map each drug index to all known positive gene indices."""
    lookup: dict[int, set[int]] = defaultdict(set)

    for drug_idx, gene_idx in known_positive_pairs.cpu().tolist():
        lookup[int(drug_idx)].add(int(gene_idx))

    return dict(lookup)


def metrics_from_ranks(
    ranks: torch.Tensor,
) -> dict[str, float | int]:
    """Calculate MRR and Hits@K from one-based ranks."""
    if ranks.numel() == 0:
        return {
            "mrr": 0.0,
            "hits_at_1": 0.0,
            "hits_at_3": 0.0,
            "hits_at_10": 0.0,
            "mean_rank": 0.0,
            "median_rank": 0.0,
            "n_queries": 0,
        }

    ranks_float = ranks.float()

    return {
        "mrr": float(
            (1.0 / ranks_float).mean().item()
        ),
        "hits_at_1": float(
            (ranks <= 1).float().mean().item()
        ),
        "hits_at_3": float(
            (ranks <= 3).float().mean().item()
        ),
        "hits_at_10": float(
            (ranks <= 10).float().mean().item()
        ),
        "mean_rank": float(
            ranks_float.mean().item()
        ),
        "median_rank": float(
            ranks_float.median().item()
        ),
        "n_queries": int(ranks.numel()),
    }


@torch.no_grad()
def filtered_gene_ranks(
    model: PGxGraphModel,
    node_type: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    evaluation_pairs: torch.Tensor,
    gene_indices: torch.Tensor,
    known_positive_pairs: torch.Tensor,
    query_batch_size: int = 64,
    molecular_features: torch.Tensor | None = None,
    has_molecular_features: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rank each true gene against every gene candidate."""
    model.eval()

    embeddings = model.encode(
        node_type=node_type,
        edge_index=edge_index,
        edge_type=edge_type,
        molecular_features=molecular_features,
        has_molecular_features=has_molecular_features,
    )

    positive_lookup = build_positive_lookup(
        known_positive_pairs
    )

    candidate_genes = gene_indices.to(
        embeddings.device
    )
    candidate_gene_embeddings = embeddings[
        candidate_genes
    ]

    relation = model.decoder.relation
    ranks: list[int] = []

    for start in range(
        0,
        len(evaluation_pairs),
        query_batch_size,
    ):
        batch = evaluation_pairs[
            start:start + query_batch_size
        ].to(embeddings.device)

        for drug_idx, true_gene_idx in batch.tolist():
            drug_embedding = embeddings[int(drug_idx)]

            scores = (
                drug_embedding.unsqueeze(0)
                * relation.unsqueeze(0)
                * candidate_gene_embeddings
            ).sum(dim=-1)

            true_positions = (
                candidate_genes == int(true_gene_idx)
            ).nonzero(as_tuple=False)

            if len(true_positions) != 1:
                raise AssertionError(
                    f"True gene {true_gene_idx} is not uniquely "
                    "present in the candidate gene set."
                )

            true_position = int(
                true_positions[0].item()
            )
            true_score = scores[true_position].clone()

            for known_gene in positive_lookup.get(
                int(drug_idx),
                set(),
            ):
                if known_gene == int(true_gene_idx):
                    continue

                known_positions = (
                    candidate_genes == known_gene
                ).nonzero(as_tuple=False)

                if len(known_positions) == 1:
                    scores[
                        int(known_positions[0].item())
                    ] = -torch.inf

            rank = int(
                1 + (scores > true_score).sum().item()
            )

            ranks.append(rank)

    return torch.tensor(
        ranks,
        dtype=torch.long,
    )


@torch.no_grad()
def evaluate_filtered_ranking(
    model: PGxGraphModel,
    node_type: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    evaluation_pairs: torch.Tensor,
    gene_indices: torch.Tensor,
    known_positive_pairs: torch.Tensor,
    query_batch_size: int = 64,
    molecular_features: torch.Tensor | None = None,
    has_molecular_features: torch.Tensor | None = None,
) -> tuple[dict[str, Any], torch.Tensor]:
    ranks = filtered_gene_ranks(
        model=model,
        node_type=node_type,
        edge_index=edge_index,
        edge_type=edge_type,
        evaluation_pairs=evaluation_pairs,
        gene_indices=gene_indices,
        known_positive_pairs=known_positive_pairs,
        query_batch_size=query_batch_size,
        molecular_features=molecular_features,
        has_molecular_features=has_molecular_features,
    )

    return metrics_from_ranks(ranks), ranks
