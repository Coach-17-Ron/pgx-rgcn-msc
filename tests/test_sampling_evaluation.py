"""Tests for negative sampling and ranking evaluation."""

from __future__ import annotations

import torch

from pgx_rgcn.evaluation import metrics_from_ranks
from pgx_rgcn.sampling import sample_negative_genes


def test_negative_sampler_excludes_known_positives():
    positives = torch.tensor(
        [[0, 10], [0, 11], [1, 12]],
        dtype=torch.long,
    )

    genes = torch.tensor(
        [10, 11, 12, 13, 14],
        dtype=torch.long,
    )

    generator = torch.Generator().manual_seed(42)

    negatives = sample_negative_genes(
        positive_pairs=positives,
        gene_indices=genes,
        known_positive_pairs=positives,
        generator=generator,
        negatives_per_positive=2,
    )

    known = {
        tuple(pair)
        for pair in positives.tolist()
    }

    assert negatives.shape == (6, 2)

    for pair in negatives.tolist():
        assert tuple(pair) not in known


def test_negative_sampler_is_reproducible():
    positives = torch.tensor(
        [[0, 10], [1, 11]],
        dtype=torch.long,
    )

    genes = torch.tensor(
        [10, 11, 12, 13],
        dtype=torch.long,
    )

    first = sample_negative_genes(
        positives,
        genes,
        positives,
        torch.Generator().manual_seed(123),
        negatives_per_positive=3,
    )

    second = sample_negative_genes(
        positives,
        genes,
        positives,
        torch.Generator().manual_seed(123),
        negatives_per_positive=3,
    )

    assert torch.equal(first, second)


def test_metrics_from_ranks():
    ranks = torch.tensor(
        [1, 2, 4, 10, 20],
        dtype=torch.long,
    )

    metrics = metrics_from_ranks(ranks)

    expected_mrr = (
        1.0
        + 0.5
        + 0.25
        + 0.1
        + 0.05
    ) / 5

    assert abs(metrics["mrr"] - expected_mrr) < 1e-6
    assert abs(metrics["hits_at_1"] - 0.2) < 1e-6
    assert abs(metrics["hits_at_3"] - 0.4) < 1e-6
    assert abs(metrics["hits_at_10"] - 0.8) < 1e-6
    assert metrics["n_queries"] == 5


def test_empty_rank_metrics():
    metrics = metrics_from_ranks(
        torch.empty(0, dtype=torch.long)
    )

    assert metrics["mrr"] == 0.0
    assert metrics["n_queries"] == 0
