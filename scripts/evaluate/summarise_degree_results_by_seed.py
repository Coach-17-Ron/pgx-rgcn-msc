#!/usr/bin/env python3

from pathlib import Path

import pandas as pd


input_path = Path(
    "results/model_vs_gene_degree_by_query.csv"
)

output_seed = Path(
    "results/model_vs_gene_degree_degree_group_by_seed.csv"
)

output_summary = Path(
    "results/model_vs_gene_degree_degree_group_summary.csv"
)

data = pd.read_csv(input_path)

seed_results = (
    data.groupby(
        [
            "model",
            "protocol",
            "seed",
            "degree_group",
        ],
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
        gene_degree_median_rank=(
            "gene_degree_rank",
            "median",
        ),
    )
    .reset_index()
)

summary = (
    seed_results.groupby(
        ["model", "protocol", "degree_group"],
        observed=True,
    )
    .agg(
        mean_queries=("n_queries", "mean"),
        model_mrr_mean=("model_mrr", "mean"),
        model_mrr_sd=("model_mrr", "std"),
        gene_degree_mrr_mean=(
            "gene_degree_mrr",
            "mean",
        ),
        gene_degree_mrr_sd=(
            "gene_degree_mrr",
            "std",
        ),
        mrr_difference_mean=(
            "mrr_difference",
            "mean",
        ),
        mrr_difference_sd=(
            "mrr_difference",
            "std",
        ),
        model_win_rate_mean=(
            "model_win_rate",
            "mean",
        ),
        model_win_rate_sd=(
            "model_win_rate",
            "std",
        ),
        n_seeds=("seed", "count"),
    )
    .reset_index()
)

group_order = ["0", "1-2", "3-9", "10+"]

seed_results["degree_group"] = pd.Categorical(
    seed_results["degree_group"],
    categories=group_order,
    ordered=True,
)

summary["degree_group"] = pd.Categorical(
    summary["degree_group"],
    categories=group_order,
    ordered=True,
)

seed_results = seed_results.sort_values(
    ["model", "protocol", "seed", "degree_group"]
)

summary = summary.sort_values(
    ["model", "protocol", "degree_group"]
)

seed_results.to_csv(output_seed, index=False)
summary.to_csv(output_summary, index=False)

display_columns = [
    "model",
    "protocol",
    "degree_group",
    "model_mrr_mean",
    "gene_degree_mrr_mean",
    "mrr_difference_mean",
    "mrr_difference_sd",
    "model_win_rate_mean",
    "n_seeds",
]

print(
    summary[display_columns]
    .round(6)
    .to_string(index=False)
)

print("\nSaved:")
print(output_seed)
print(output_summary)
