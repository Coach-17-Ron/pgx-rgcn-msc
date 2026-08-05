"""Shared graph encoders and drug-gene link-prediction model."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GCNConv, RGCNConv, RGATConv


class NodeFeatureEncoder(nn.Module):
    """Learn node embeddings enriched with node-type embeddings."""

    def __init__(
        self,
        num_nodes: int,
        num_node_types: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()

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

    def forward(
        self,
        node_type: torch.Tensor,
    ) -> torch.Tensor:
        node_indices = torch.arange(
            node_type.shape[0],
            device=node_type.device,
        )

        return (
            self.node_embedding(node_indices)
            + self.node_type_embedding(node_type)
        )


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
    ) -> None:
        super().__init__()

        self.model_name = model_name.lower()

        self.features = NodeFeatureEncoder(
            num_nodes=num_nodes,
            num_node_types=num_node_types,
            embedding_dim=embedding_dim,
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
    ) -> torch.Tensor:
        x = self.features(node_type)

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
    ) -> torch.Tensor:
        embeddings = self.encode(
            node_type=node_type,
            edge_index=edge_index,
            edge_type=edge_type,
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
