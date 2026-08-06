#!/usr/bin/env python
"""Create the manuscript table of externally supported candidate pairs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(
    "results/consensus_inference/molecular/"
    "strict_cold/summary"
)

INPUT_PATH = (
    ROOT / "integrated_candidate_evidence_table.csv"
)

OUTPUT_PATH = (
    ROOT / "manuscript_supported_candidate_case_table.csv"
)


def main() -> None:
    table = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    supported_tiers = {
        "direct_supported",
        "direct_mixed",
        "indirect_support",
    }

    supported = table.loc[
        table["external_support_tier"].isin(
            supported_tiers
        )
    ].copy()

    support_order = {
        "direct_supported": 1,
        "direct_mixed": 2,
        "indirect_support": 3,
    }

    supported["support_order"] = (
        supported["external_support_tier"]
        .map(support_order)
    )

    # Prefer the precise PMID/evidence fields already present in
    # the external-validation table without assuming their names.
    candidate_columns = [
        "external_validation_rank",
        "drug_name",
        "gene_name",
        "external_support_tier",
        "pubmed_pair_evidence_class",
        "supporting_pmids",
        "direct_support_pmids",
        "positive_support_pmids",
        "null_evidence_pmids",
        "strict_cold_eligible_seed_count",
        "strict_cold_eligible_seeds",
        "rgcn_top_relation_family",
        "rgcn_top_relation_mean_score_drop",
        "rgcn_top_relation_mean_rank_worsening",
        "rgcn_supportive_score_seed_fraction",
        "gcn_top_relation_family",
        "gcn_supportive_score_seed_fraction",
        "rgat_top_gene_incoming_relation",
        "rgat_mean_gene_attention_fraction",
    ]

    available_columns = [
        column
        for column in candidate_columns
        if column in supported.columns
    ]

    manuscript = supported[
        available_columns
    ].sort_values(
        [
            "support_order",
            "external_validation_rank",
        ]
        if "support_order" in available_columns
        else ["external_validation_rank"]
    )

    # support_order is only needed for sorting.
    manuscript = manuscript.drop(
        columns=["support_order"],
        errors="ignore",
    )

    manuscript.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("SUPPORTED CANDIDATE CASE TABLE COMPLETE")
    print("Pairs:", len(manuscript))

    print("\nSUPPORT DISTRIBUTION")
    print(
        supported["external_support_tier"]
        .value_counts()
        .to_string()
    )

    print("\nSUPPORTED CANDIDATES")
    print(
        manuscript[
            [
                column
                for column in [
                    "external_validation_rank",
                    "drug_name",
                    "gene_name",
                    "external_support_tier",
                    "strict_cold_eligible_seed_count",
                    "rgcn_top_relation_family",
                    "rgcn_supportive_score_seed_fraction",
                    "gcn_top_relation_family",
                ]
                if column in manuscript.columns
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
