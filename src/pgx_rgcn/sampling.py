"""Negative sampling utilities for drug-gene link prediction."""

from __future__ import annotations

import torch


def pairs_to_set(pairs: torch.Tensor) -> set[tuple[int, int]]:
    """Convert a two-column tensor to a Python pair set."""
    return {
        (int(drug), int(gene))
        for drug, gene in pairs.cpu().tolist()
    }


def sample_negative_genes(
    positive_pairs: torch.Tensor,
    gene_indices: torch.Tensor,
    known_positive_pairs: torch.Tensor,
    generator: torch.Generator,
    negatives_per_positive: int = 1,
) -> torch.Tensor:
    """Sample genes that are not known positives for each drug."""
    if positive_pairs.ndim != 2 or positive_pairs.shape[1] != 2:
        raise ValueError(
            "positive_pairs must have shape [N, 2]."
        )

    if negatives_per_positive < 1:
        raise ValueError(
            "negatives_per_positive must be at least 1."
        )

    known = pairs_to_set(known_positive_pairs)
    genes = gene_indices.cpu()

    negatives: list[list[int]] = []

    for drug_idx, _ in positive_pairs.cpu().tolist():
        sampled = 0
        attempts = 0
        maximum_attempts = max(
            1000,
            negatives_per_positive * 100,
        )

        while sampled < negatives_per_positive:
            position = int(
                torch.randint(
                    low=0,
                    high=len(genes),
                    size=(1,),
                    generator=generator,
                ).item()
            )

            gene_idx = int(genes[position])
            candidate = (int(drug_idx), gene_idx)

            attempts += 1

            if candidate in known:
                if attempts >= maximum_attempts:
                    raise RuntimeError(
                        "Could not sample enough valid negatives."
                    )
                continue

            negatives.append(
                [int(drug_idx), gene_idx]
            )
            sampled += 1

    return torch.tensor(
        negatives,
        dtype=torch.long,
    )
