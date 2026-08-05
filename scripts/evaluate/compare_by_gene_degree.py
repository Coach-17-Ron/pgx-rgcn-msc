#!/usr/bin/env python3

import json
from pathlib import Path

import pandas as pd
import torch


MODELS = ["gcn", "rgcn", "rgat"]
PROTOCOLS = ["dg_context", "strict_cold"]
SEEDS = [42, 123, 2026]


def degree_group(degree: int) -> str:
    if degree == 0:
        return "0"
    if degree <= 2:
        return "1-2"
    if degree <= 9:
        return "3-9"
    return "10+"


records = []

for protocol in PROTOCOLS:
    for seed in SEEDS:
        tensor_path = (
            Path("artifacts/tensors")
            / protocol
            / f"seed_{seed}"
            / "tensor_bundle.pt"
        )

        bundle = torch.load(
            tensor_path,
            map_location="cpu",
            weights_only=False,
        )

        test_pairs = bundle["test_positive_pairs"].cpu()
        train_pairs = bundle["train_positive_pairs"].cpu()

        gene_degree: dict[int, int] = {}

        for gene_index in train_pairs[:, 1].tolist():
            gene_index = int(gene_index)
            gene_degree[gene_index] = (
                gene_degree.get(gene_index, 0) + 1
            )

        baseline_ranks = torch.load(
            Path("results/baselines")
            / "gene_degree"
            / protocol
            / f"seed_{seed}"
            / "test_ranks.pt",
            map_location="cpu",
            weights_only=False,
        ).float()

        for model in MODELS:
            model_ranks = torch.load(
                Path("results/models")
                / model
                / protocol
                / f"seed_{seed}"
                / "test_ranks.pt",
                map_location="cpu",
                weights_only=False,
            ).float()

            if model_ranks.shape[0] != test_pairs.shape[0]:
                raise ValueError(
                    f"Query mismatch: {model}, {protocol}, {seed}"
                )

            for query_index, pair in enumerate(test_pairs):
                gene_index = int(pair[1].item())
                degree = gene_degree.get(gene_index, 0)

                model_rank = float(
                    model_ranks[query_index].item()
                )
                baseline_rank = float(
                    baseline_ranks[query_index].item()
                )

                records.append(
                    {
                        "model": model,
                        "protocol": protocol,
                        "seed": seed,
                        "query_index": query_index,
                        "gene_index": gene_index,
                        "training_gene_degree": degree,
                        "degree_group": degree_group(degree),
                        "model_rank": model_rank,
                        "gene_degree_rank": baseline_rank,
                        "model_reciprocal_rank": 1.0 / model_rank,
                        "gene_degree_reciprocal_rank": (
                            1.0 / baseline_rank
                        ),
                        "rr_difference": (
                            1.0 / model_rank
                            - 1.0 / baseline_rank
                        ),
                        "model_better": (
                            model_rank < baseline_rank
                        ),
                    }
                )


query_results = pd.DataFrame(records)

group_order = ["0", "1-2", "3-9", "10+"]

summary = (
    query_results.groupby(
        ["model", "protocol", "degree_group"],
        observed=True,
    )
    .agg(
        n_queries=("query_index", "count"),
        model_mrr=("model_reciprocal_rank", "mean"),
        gene_degree_mrr=(
            "gene_degree_reciprocal_rank",
            "mean",
        ),
        mrr_difference=("rr_difference", "mean"),
        model_win_rate=("model_better", "mean"),
        model_median_rank=("model_rank", "median"),
        baseline_median_rank=(
            "gene_degree_rank",
            "median",
        ),
    )
    .reset_index()
)

summary["degree_group"] = pd.Categorical(
    summary["degree_group"],
    categories=group_order,
    ordered=True,
)

summary = summary.sort_values(
    ["model", "protocol", "degree_group"]
)

query_results.to_csv(
    "results/model_vs_gene_degree_by_query.csv",
    index=False,
)

summary.to_csv(
    "results/model_vs_gene_degree_by_degree_group.csv",
    index=False,
)

print(
    summary[
        [
            "model",
            "protocol",
            "degree_group",
            "n_queries",
            "model_mrr",
            "gene_degree_mrr",
            "mrr_difference",
            "model_win_rate",
        ]
    ]
    .round(6)
    .to_string(index=False)
)

print("\nSaved:")
print("results/model_vs_gene_degree_by_query.csv")
print("results/model_vs_gene_degree_by_degree_group.csv")
