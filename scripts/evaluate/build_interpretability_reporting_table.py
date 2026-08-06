#!/usr/bin/env python
"""Build a cautious manuscript-ready interpretability reporting table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "results/consensus_inference/molecular/"
    "strict_cold/summary"
)

COMPARISON_PATH = (
    ROOT / "model_interpretability_comparison.csv"
)

OUTPUT_PATH = (
    ROOT / "manuscript_interpretability_reporting_table.csv"
)

MODEL_SUMMARY_PATH = (
    ROOT / "manuscript_interpretability_model_summary.csv"
)


MODEL_RELIABILITY = {
    "GCN": "topology_sensitivity_comparator",
    "RGCN": "primary_relation_aware_explanation",
    "RGAT": "descriptive_unstable_attention_comparator",
}


def load_random_control(
    model: str,
) -> pd.DataFrame:
    path = (
        ROOT
        / f"{model.lower()}_relation_ablation_"
          "random_control_summary.csv"
    )

    if not path.exists():
        return pd.DataFrame()

    table = pd.read_csv(
        path,
        low_memory=False,
    )

    table.insert(0, "model", model)

    return table


def main() -> None:
    comparison = pd.read_csv(
        COMPARISON_PATH,
        low_memory=False,
    )

    random_frames = [
        load_random_control("GCN"),
        load_random_control("RGCN"),
    ]

    random_frames = [
        frame
        for frame in random_frames
        if not frame.empty
    ]

    random_control = pd.concat(
        random_frames,
        ignore_index=True,
    )

    random_columns = [
        "model",
        "relation_family",
        "mean_excess_drop_over_random",
        "median_excess_drop_over_random",
        "fraction_exceeding_random_mean",
        "fraction_empirical_p_le_0_05",
    ]

    reporting = comparison.merge(
        random_control[random_columns],
        on=[
            "model",
            "relation_family",
        ],
        how="left",
        validate="one_to_one",
    )

    reporting["interpretability_role"] = (
        reporting["model"].map(
            MODEL_RELIABILITY
        )
    )

    reporting["top5_within_model"] = (
        reporting[
            "relation_rank_within_model"
        ].le(5)
    )

    reporting["top10_within_model"] = (
        reporting[
            "relation_rank_within_model"
        ].le(10)
    )

    reporting[
        "majority_supportive_score_effect"
    ] = (
        reporting[
            "supportive_score_fraction"
        ] > 0.5
    )

    reporting[
        "majority_supportive_rank_effect"
    ] = (
        reporting[
            "supportive_rank_fraction"
        ] > 0.5
    )

    reporting[
        "exceeds_random_in_majority"
    ] = np.where(
        reporting[
            "fraction_exceeding_random_mean"
        ].notna(),
        reporting[
            "fraction_exceeding_random_mean"
        ] > 0.5,
        pd.NA,
    )

    reporting["reporting_strength"] = "limited"

    rgcn_primary = (
        reporting["model"].eq("RGCN")
        & reporting["top5_within_model"]
        & reporting[
            "majority_supportive_score_effect"
        ]
        & reporting[
            "exceeds_random_in_majority"
        ].fillna(False)
    )

    gcn_topology = (
        reporting["model"].eq("GCN")
        & reporting["top5_within_model"]
        & reporting[
            "majority_supportive_score_effect"
        ]
        & reporting[
            "exceeds_random_in_majority"
        ].fillna(False)
    )

    rgat_descriptive = (
        reporting["model"].eq("RGAT")
        & reporting["top5_within_model"]
    )

    reporting.loc[
        rgcn_primary,
        "reporting_strength",
    ] = "primary_relation_aware"

    reporting.loc[
        gcn_topology,
        "reporting_strength",
    ] = "supported_topology_sensitivity"

    reporting.loc[
        rgat_descriptive,
        "reporting_strength",
    ] = "descriptive_only"

    reporting["interpretation_note"] = ""

    reporting.loc[
        reporting["model"].eq("GCN"),
        "interpretation_note",
    ] = (
        "Relation-labelled edge removal is interpreted "
        "as topology sensitivity because GCN does not "
        "use relation-specific transformations."
    )

    reporting.loc[
        reporting["model"].eq("RGCN"),
        "interpretation_note",
    ] = (
        "Relation-family ablation reflects sensitivity "
        "of the relation-aware encoder; it does not "
        "establish biological causation."
    )

    reporting.loc[
        reporting["model"].eq("RGAT"),
        "interpretation_note",
    ] = (
        "Tiny score changes and large rank changes indicate "
        "near-tied scores and unstable rankings; attention "
        "and ablation are descriptive only."
    )

    reporting = reporting.sort_values(
        [
            "model",
            "relation_rank_within_model",
        ]
    )

    reporting.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    selected = reporting.loc[
        reporting["top5_within_model"]
    ].copy()

    model_summary_rows = []

    for model in ["GCN", "RGCN", "RGAT"]:
        subset = selected.loc[
            selected["model"].eq(model)
        ].sort_values(
            "relation_rank_within_model"
        )

        top_relations = "; ".join(
            subset["relation_family"].tolist()
        )

        if model == "GCN":
            conclusion = (
                "Predictions were most sensitive to broad "
                "phenotype–variant, drug–variant and "
                "drug–gene graph connectivity."
            )
        elif model == "RGCN":
            conclusion = (
                "Predictions were most sensitive to "
                "drug–gene clinical, pathway, label and "
                "guideline evidence families."
            )
        else:
            conclusion = (
                "Ablations produced very small score changes "
                "but large rank movements, indicating unstable "
                "near-tied candidate scores."
            )

        model_summary_rows.append(
            {
                "model": model,
                "interpretability_role": (
                    MODEL_RELIABILITY[model]
                ),
                "top_five_relation_families": (
                    top_relations
                ),
                "model_level_conclusion": conclusion,
                "raw_score_drops_comparable_across_models": (
                    False
                ),
            }
        )

    model_summary = pd.DataFrame(
        model_summary_rows
    )

    model_summary.to_csv(
        MODEL_SUMMARY_PATH,
        index=False,
    )

    print(
        "INTERPRETABILITY REPORTING TABLE COMPLETE"
    )

    print("\nPRIMARY/SUPPORTED RESULTS")
    print(
        reporting.loc[
            reporting["reporting_strength"].isin(
                [
                    "primary_relation_aware",
                    "supported_topology_sensitivity",
                ]
            ),
            [
                "model",
                "relation_rank_within_model",
                "relation_family",
                "supportive_score_fraction",
                "fraction_exceeding_random_mean",
                "reporting_strength",
            ],
        ].to_string(index=False)
    )

    print("\nMODEL-LEVEL SUMMARY")
    print(
        model_summary[
            [
                "model",
                "interpretability_role",
                "model_level_conclusion",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(OUTPUT_PATH)
    print(MODEL_SUMMARY_PATH)


if __name__ == "__main__":
    main()
