#!/usr/bin/env python
"""Build a consolidated GCN/RGCN/RGAT interpretability comparison."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(
    "results/consensus_inference/molecular/"
    "strict_cold/summary"
)

MODELS = ["gcn", "rgcn", "rgat"]

OUTPUT = ROOT / "model_interpretability_comparison.csv"
TOP_OUTPUT = ROOT / "model_interpretability_top_relations.csv"


def main() -> None:
    frames: list[pd.DataFrame] = []

    for model in MODELS:
        path = (
            ROOT
            / f"{model}_relation_family_ablation_global_summary.csv"
        )

        table = pd.read_csv(
            path,
            low_memory=False,
        )

        table.insert(0, "model", model.upper())

        table["absolute_mean_score_drop"] = (
            table["mean_score_drop"].abs()
        )

        table["absolute_mean_rank_change"] = (
            table["mean_rank_worsening"].abs()
        )

        table["score_drop_per_1000_removed_edges"] = (
            table["mean_score_drop"]
            / table["mean_removed_edge_count"]
            * 1000
        )

        table["rank_change_per_1000_removed_edges"] = (
            table["mean_rank_worsening"]
            / table["mean_removed_edge_count"]
            * 1000
        )

        table["relation_rank_within_model"] = (
            table["mean_score_drop"]
            .rank(
                method="min",
                ascending=False,
            )
            .astype(int)
        )

        frames.append(table)

    comparison = pd.concat(
        frames,
        ignore_index=True,
    )

    comparison.to_csv(
        OUTPUT,
        index=False,
    )

    top = comparison.loc[
        comparison[
            "relation_rank_within_model"
        ].le(10)
    ].copy()

    top = top.sort_values(
        [
            "model",
            "relation_rank_within_model",
        ]
    )

    top.to_csv(
        TOP_OUTPUT,
        index=False,
    )

    print("MODEL INTERPRETABILITY COMPARISON COMPLETE")

    for model in ["GCN", "RGCN", "RGAT"]:
        print(f"\nTOP RELATIONS — {model}")

        subset = comparison.loc[
            comparison["model"].eq(model)
        ].sort_values(
            "relation_rank_within_model"
        )

        print(
            subset[
                [
                    "relation_rank_within_model",
                    "relation_family",
                    "mean_score_drop",
                    "mean_rank_worsening",
                    "supportive_score_fraction",
                    "supportive_rank_fraction",
                    "mean_removed_edge_count",
                    "score_drop_per_1000_removed_edges",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

    selected = comparison.loc[
        comparison["relation_family"].isin(
            [
                "drug_gene_clinical_associated",
                "drug_gene_label_associated",
                "drug_gene_pathway_associated",
                "drug_variant_clinical_associated",
                "phenotype_variant_clinical_associated",
                "gene_phenotype_clinical_associated",
            ]
        )
    ].copy()

    print("\nSELECTED RELATIONS ACROSS MODELS")

    print(
        selected[
            [
                "model",
                "relation_family",
                "mean_score_drop",
                "mean_rank_worsening",
                "supportive_score_fraction",
                "relation_rank_within_model",
            ]
        ]
        .sort_values(
            [
                "relation_family",
                "model",
            ]
        )
        .to_string(index=False)
    )

    print("\nSaved:")
    print(OUTPUT)
    print(TOP_OUTPUT)


if __name__ == "__main__":
    main()
