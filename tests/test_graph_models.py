"""Tests for GCN, R-GCN and RGAT model implementations."""

from __future__ import annotations

import pytest
import torch

from pgx_rgcn.models import PGxGraphModel


@pytest.fixture
def small_graph():
    node_type = torch.tensor(
        [0, 1, 2, 3, 1, 0],
        dtype=torch.long,
    )

    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 1, 4],
            [1, 2, 3, 4, 5, 0, 5, 2],
        ],
        dtype=torch.long,
    )

    edge_type = torch.tensor(
        [0, 1, 2, 0, 1, 2, 1, 0],
        dtype=torch.long,
    )

    pairs = torch.tensor(
        [
            [0, 1],
            [5, 4],
        ],
        dtype=torch.long,
    )

    return node_type, edge_index, edge_type, pairs


@pytest.mark.parametrize(
    "model_name",
    ["gcn", "rgcn", "rgat"],
)
def test_model_output_shapes(
    model_name,
    small_graph,
):
    node_type, edge_index, edge_type, pairs = small_graph

    model = PGxGraphModel(
        model_name=model_name,
        num_nodes=6,
        num_node_types=4,
        num_relations=3,
        embedding_dim=8,
        num_bases=2,
        dropout=0.0,
    )

    embeddings = model.encode(
        node_type=node_type,
        edge_index=edge_index,
        edge_type=edge_type,
    )

    scores = model.score_pairs(
        embeddings=embeddings,
        pairs=pairs,
    )

    assert embeddings.shape == (6, 8)
    assert scores.shape == (2,)
    assert torch.isfinite(embeddings).all()
    assert torch.isfinite(scores).all()


@pytest.mark.parametrize(
    "model_name",
    ["gcn", "rgcn", "rgat"],
)
def test_model_backward_pass(
    model_name,
    small_graph,
):
    node_type, edge_index, edge_type, pairs = small_graph

    model = PGxGraphModel(
        model_name=model_name,
        num_nodes=6,
        num_node_types=4,
        num_relations=3,
        embedding_dim=8,
        num_bases=2,
        dropout=0.0,
    )

    scores = model(
        node_type=node_type,
        edge_index=edge_index,
        edge_type=edge_type,
        pairs=pairs,
    )

    loss = scores.square().mean()
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert any(
        gradient is not None
        for gradient in gradients
    )


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError):
        PGxGraphModel(
            model_name="unknown",
            num_nodes=6,
            num_node_types=4,
            num_relations=3,
        )


def test_molecular_features_replace_drug_id_embeddings():
    node_type = torch.tensor(
        [0, 1, 0, 2],
        dtype=torch.long,
    )

    molecular_features = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float,
    )

    has_molecular_features = torch.tensor(
        [True, False, True, False],
        dtype=torch.bool,
    )

    model = PGxGraphModel(
        model_name="gcn",
        num_nodes=4,
        num_node_types=4,
        num_relations=1,
        embedding_dim=8,
        dropout=0.0,
        feature_mode="molecular",
        molecular_feature_dim=4,
        drug_type_index=0,
    )

    with torch.no_grad():
        first = model.features(
            node_type=node_type,
            molecular_features=molecular_features,
            has_molecular_features=has_molecular_features,
        )

        model.features.node_embedding.weight.add_(100.0)

        second = model.features(
            node_type=node_type,
            molecular_features=molecular_features,
            has_molecular_features=has_molecular_features,
        )

    assert torch.allclose(first[0], second[0])
    assert torch.allclose(first[2], second[2])
    assert not torch.allclose(first[1], second[1])
    assert not torch.allclose(first[3], second[3])


def test_missing_smiles_drugs_share_fallback_representation():
    node_type = torch.tensor(
        [0, 0, 1],
        dtype=torch.long,
    )

    molecular_features = torch.zeros(
        (3, 4),
        dtype=torch.float,
    )

    has_molecular_features = torch.tensor(
        [False, False, False],
        dtype=torch.bool,
    )

    model = PGxGraphModel(
        model_name="gcn",
        num_nodes=3,
        num_node_types=4,
        num_relations=1,
        embedding_dim=8,
        dropout=0.0,
        feature_mode="molecular",
        molecular_feature_dim=4,
        drug_type_index=0,
    )

    features = model.features(
        node_type=node_type,
        molecular_features=molecular_features,
        has_molecular_features=has_molecular_features,
    )

    assert torch.allclose(features[0], features[1])
    assert not torch.allclose(features[0], features[2])


def test_molecular_mode_requires_feature_tensors():
    model = PGxGraphModel(
        model_name="gcn",
        num_nodes=3,
        num_node_types=4,
        num_relations=1,
        embedding_dim=8,
        dropout=0.0,
        feature_mode="molecular",
        molecular_feature_dim=4,
        drug_type_index=0,
    )

    node_type = torch.tensor(
        [0, 1, 2],
        dtype=torch.long,
    )

    with pytest.raises(ValueError):
        model.features(node_type=node_type)
