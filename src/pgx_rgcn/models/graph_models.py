"""Shared graph encoders and drug-gene link-prediction model."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GCNConv, RGCNConv, RGATConv


class NodeFeatureEncoder(nn.Module):
    """Create ID-based or molecular-feature-aware node representations."""

    def __init__(
        self,
        num_nodes: int,
        num_node_types: int,
        embedding_dim: int,
        feature_mode: str = "id",
        molecular_feature_dim: int | None = None,
        drug_type_index: int = 0,
    ) -> None:
        super().__init__()

        feature_mode = feature_mode.lower()

        if feature_mode not in {"id", "molecular"}:
            raise ValueError(
                "feature_mode must be either 'id' or 'molecular'."
            )

        if (
            feature_mode == "molecular"
            and (
                molecular_feature_dim is None
                or molecular_feature_dim <= 0
            )
        ):
            raise ValueError(
                "molecular_feature_dim must be positive when "
                "feature_mode='molecular'."
            )

        self.feature_mode = feature_mode
        self.drug_type_index = int(drug_type_index)

        self.node_embedding = nn.Embedding(
            num_nodes,
            embedding_dim,
        )
        self.node_type_embedding = nn.Embedding(
            num_node_types,
            embedding_dim,
        )

        nn.init.xavier_uniform_(self.node_embedding.weight)
        nn.init.xavier_uniform_(self.node_type_embedding.weight)

        if self.feature_mode == "molecular":
            self.molecular_projection = nn.Sequential(
                nn.Linear(
                    int(molecular_feature_dim),
                    embedding_dim,
                ),
                nn.ReLU(),
                nn.LayerNorm(embedding_dim),
            )

            self.missing_molecular_embedding = nn.Parameter(
                torch.empty(embedding_dim)
            )

            nn.init.normal_(
                self.missing_molecular_embedding,
                mean=0.0,
                std=embedding_dim ** -0.5,
            )
        else:
            self.molecular_projection = None
            self.register_parameter(
                "missing_molecular_embedding",
                None,
            )

    def forward(
        self,
        node_type: torch.Tensor,
        molecular_features: torch.Tensor | None = None,
        has_molecular_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        node_indices = torch.arange(
            node_type.shape[0],
            device=node_type.device,
        )

        node_id_features = self.node_embedding(node_indices)
        type_features = self.node_type_embedding(node_type)

        if self.feature_mode == "id":
            return node_id_features + type_features

        if molecular_features is None:
            raise ValueError(
                "molecular_features are required when "
                "feature_mode='molecular'."
            )

        if has_molecular_features is None:
            raise ValueError(
                "has_molecular_features is required when "
                "feature_mode='molecular'."
            )

        if molecular_features.shape[0] != node_type.shape[0]:
            raise ValueError(
                "molecular_features must have one row per node."
            )

        if has_molecular_features.shape[0] != node_type.shape[0]:
            raise ValueError(
                "has_molecular_features must have one value per node."
            )

        x = node_id_features + type_features

        drug_mask = node_type == self.drug_type_index
        available_drug_mask = (
            drug_mask & has_molecular_features.bool()
        )
        missing_drug_mask = (
            drug_mask & ~has_molecular_features.bool()
        )

        if available_drug_mask.any():
            projected = self.molecular_projection(
                molecular_features[
                    available_drug_mask
                ].float()
            )

            x[available_drug_mask] = (
                projected
                + type_features[available_drug_mask]
            )

        if missing_drug_mask.any():
            x[missing_drug_mask] = (
                self.missing_molecular_embedding.unsqueeze(0)
                + type_features[missing_drug_mask]
            )

        return x


class GCNEncoder(nn.Module):
    """Relation-agnostic GCN comparison model."""

    def __init__(
        self,
        channels: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.conv1 = GCNConv(channels, channels)
        self.conv2 = GCNConv(channels, channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        del edge_type

        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)

        return x


class RGCNEncoder(nn.Module):
    """Relational graph convolutional encoder."""

    def __init__(
        self,
        channels: int,
        num_relations: int,
        num_bases: int,
        dropout: float,
    ) -> None:
        super().__init__()

        bases = min(num_bases, num_relations)

        self.conv1 = RGCNConv(
            channels,
            channels,
            num_relations=num_relations,
            num_bases=bases,
        )

        self.conv2 = RGCNConv(
            channels,
            channels,
            num_relations=num_relations,
            num_bases=bases,
        )

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv1(x, edge_index, edge_type)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_type)

        return x


class RGATEncoder(nn.Module):
    """Relational graph-attention comparison model."""

    def __init__(
        self,
        channels: int,
        num_relations: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.conv1 = RGATConv(
            channels,
            channels,
            num_relations=num_relations,
            heads=1,
            concat=False,
            dropout=dropout,
        )

        self.conv2 = RGATConv(
            channels,
            channels,
            num_relations=num_relations,
            heads=1,
            concat=False,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv1(x, edge_index, edge_type)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_type)

        return x


class DrugGeneDistMultDecoder(nn.Module):
    """Task-specific DistMult decoder for drug-gene association."""

    def __init__(
        self,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        self.relation = nn.Parameter(
            torch.empty(embedding_dim)
        )

        nn.init.xavier_uniform_(
            self.relation.unsqueeze(0)
        )

    def forward(
        self,
        drug_embeddings: torch.Tensor,
        gene_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        return (
            drug_embeddings
            * self.relation
            * gene_embeddings
        ).sum(dim=-1)


class PGxGraphModel(nn.Module):
    """Common link-prediction model wrapping one graph encoder."""

    def __init__(
        self,
        model_name: str,
        num_nodes: int,
        num_node_types: int,
        num_relations: int,
        embedding_dim: int = 64,
        num_bases: int = 8,
        dropout: float = 0.2,
        feature_mode: str = "id",
        molecular_feature_dim: int | None = None,
        drug_type_index: int = 0,
    ) -> None:
        super().__init__()

        self.model_name = model_name.lower()

        self.feature_mode = feature_mode.lower()

        self.features = NodeFeatureEncoder(
            num_nodes=num_nodes,
            num_node_types=num_node_types,
            embedding_dim=embedding_dim,
            feature_mode=self.feature_mode,
            molecular_feature_dim=molecular_feature_dim,
            drug_type_index=drug_type_index,
        )

        if self.model_name == "gcn":
            self.encoder = GCNEncoder(
                channels=embedding_dim,
                dropout=dropout,
            )

        elif self.model_name == "rgcn":
            self.encoder = RGCNEncoder(
                channels=embedding_dim,
                num_relations=num_relations,
                num_bases=num_bases,
                dropout=dropout,
            )

        elif self.model_name == "rgat":
            self.encoder = RGATEncoder(
                channels=embedding_dim,
                num_relations=num_relations,
                dropout=dropout,
            )

        else:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                "Choose gcn, rgcn or rgat."
            )

        self.decoder = DrugGeneDistMultDecoder(
            embedding_dim=embedding_dim
        )

    def encode(
        self,
        node_type: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        molecular_features: torch.Tensor | None = None,
        has_molecular_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.features(
            node_type=node_type,
            molecular_features=molecular_features,
            has_molecular_features=has_molecular_features,
        )

        return self.encoder(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
        )

    def score_pairs(
        self,
        embeddings: torch.Tensor,
        pairs: torch.Tensor,
    ) -> torch.Tensor:
        drugs = embeddings[pairs[:, 0]]
        genes = embeddings[pairs[:, 1]]

        return self.decoder(
            drug_embeddings=drugs,
            gene_embeddings=genes,
        )

    def forward(
        self,
        node_type: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        pairs: torch.Tensor,
        molecular_features: torch.Tensor | None = None,
        has_molecular_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings = self.encode(
            node_type=node_type,
            edge_index=edge_index,
            edge_type=edge_type,
            molecular_features=molecular_features,
            has_molecular_features=has_molecular_features,
        )

        return self.score_pairs(
            embeddings=embeddings,
            pairs=pairs,
        )


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
