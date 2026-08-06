#!/usr/bin/env python
"""Integrate model ranking, PubMed evidence, and pathway context."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(
    "results/consensus_inference/molecular/"
    "strict_cold/summary"
)

MODEL_PATH = ROOT / "external_validation_candidate_pairs.csv"
PUBMED_PATH = ROOT / "pubmed_pair_validation_summary.csv"
PATHWAY_PATH = ROOT / "candidate_pair_pathway_summary.csv"
KEGG_PATH = ROOT / "kegg_pair_pathway_validation_summary.csv"

OUTPUT_PATH = ROOT / "final_external_validation_evidence_table.csv"
SUMMARY_PATH = ROOT / "final_external_validation_summary.csv"


KEYS = [
    "external_validation_rank",
    "drug_name",
    "gene_name",
]


PUBMED_FIELDS = KEYS + [
    "pubmed_pair_evidence_class",
    "retrieved_unique_pmids",
    "direct_supporting_pmids",
    "indirect_supporting_pmids",
    "context_dependent_pmids",
    "null_pmids",
    "article_evidence_classes",
    "review_notes",
]


PATHWAY_FIELDS = KEYS + [
    "kegg_pathway_count",
    "reactome_pathway_count",
    "total_pathway_count",
    "pathway_sources",
    "pathway_names",
    "has_any_pathway_context",
]


KEGG_FIELDS = KEYS + [
    "kegg_drug_id",
    "drug_mapping_status",
    "drug_pathway_count",
    "kegg_biological_drug_pathway_count",
    "overlapping_kegg_pathway_count",
    "overlapping_kegg_pathway_ids",
    "overlapping_kegg_pathway_names",
    "has_kegg_drug_gene_pathway_overlap",
    "kegg_pair_pathway_result",
]


POSITIVE_SUPPORT_SCORE = {
    "direct_supported": 3,
    "direct_mixed_evidence": 2,
    "indirect_mechanistic_support": 1,
    "direct_null_only": 0,
    "no_supporting_pair_evidence": 0,
    "no_evidence_found_in_current_search": 0,
}


EVIDENCE_CATEGORY = {
    "direct_supported": "positive_direct_support",
    "direct_mixed_evidence": "positive_but_mixed",
    "indirect_mechanistic_support": "positive_indirect_support",
    "direct_null_only": "null_evidence_only",
    "no_supporting_pair_evidence": "retrieved_but_not_supporting",
    "no_evidence_found_in_current_search": "no_evidence_retrieved",
}


def assert_unique(
    frame: pd.DataFrame,
    name: str,
) -> None:
    duplicated = frame.duplicated(KEYS, keep=False)

    if duplicated.any():
        examples = frame.loc[
            duplicated,
            KEYS,
        ].head(10)

        raise ValueError(
            f"{name} contains duplicate candidate-pair keys:\n"
            f"{examples.to_string(index=False)}"
        )


def nonempty(value: object) -> bool:
    if pd.isna(value):
        return False

    return bool(str(value).strip())


def main() -> None:
    model = pd.read_csv(
        MODEL_PATH,
        low_memory=False,
    )

    pubmed = pd.read_csv(
        PUBMED_PATH,
        low_memory=False,
    )[PUBMED_FIELDS]

    pathway = pd.read_csv(
        PATHWAY_PATH,
        low_memory=False,
    )[PATHWAY_FIELDS]

    kegg = pd.read_csv(
        KEGG_PATH,
        low_memory=False,
    )[KEGG_FIELDS]

    for frame, name in [
        (model, "model table"),
        (pubmed, "PubMed table"),
        (pathway, "pathway table"),
        (kegg, "KEGG table"),
    ]:
        assert_unique(frame, name)

    final = (
        model.merge(
            pubmed,
            on=KEYS,
            how="left",
            validate="one_to_one",
        )
        .merge(
            pathway,
            on=KEYS,
            how="left",
            validate="one_to_one",
        )
        .merge(
            kegg,
            on=KEYS,
            how="left",
            validate="one_to_one",
        )
    )

    if len(final) != 30:
        raise ValueError(
            f"Expected 30 candidate pairs, found {len(final)}."
        )

    final["positive_support_score"] = (
        final["pubmed_pair_evidence_class"]
        .map(POSITIVE_SUPPORT_SCORE)
        .fillna(0)
        .astype(int)
    )

    final["external_validation_category"] = (
        final["pubmed_pair_evidence_class"]
        .map(EVIDENCE_CATEGORY)
        .fillna("unclassified")
    )

    final["has_positive_pubmed_support"] = (
        final["positive_support_score"] > 0
    )

    final["has_direct_pubmed_support"] = (
        final["pubmed_pair_evidence_class"]
        .isin(
            [
                "direct_supported",
                "direct_mixed_evidence",
            ]
        )
    )

    final["has_null_evidence"] = (
        final["null_pmids"].map(nonempty)
    )

    final["pathway_interpretation"] = (
        "gene_pathway_context_only"
    )

    final.loc[
        final["total_pathway_count"].fillna(0).eq(0),
        "pathway_interpretation",
    ] = "no_gene_pathway_context"

    final["kegg_pair_interpretation"] = (
        "no_positive_pair_level_support"
    )

    final.loc[
        final[
            "has_kegg_drug_gene_pathway_overlap"
        ].fillna(False),
        "kegg_pair_interpretation",
    ] = "exact_shared_pathway_context"

    # Conservative prioritisation:
    # 1. external evidence class;
    # 2. original model-derived validation priority;
    # 3. original external-validation rank.
    final = final.sort_values(
        [
            "positive_support_score",
            "validation_priority_score",
            "external_validation_rank",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    ).reset_index(drop=True)

    final["integrated_evidence_rank"] = (
        final.index + 1
    )

    preferred_columns = [
        "integrated_evidence_rank",
        "external_validation_rank",
        "drug_name",
        "gene_name",
        "candidate_class",
        "validation_priority_score",
        "weighted_consensus_rank",
        "weighted_rrf_score",
        "gcn_rank",
        "rgcn_rank",
        "rgat_rank",
        "models_top_25",
        "seeds_observed",
        "positive_support_score",
        "external_validation_category",
        "pubmed_pair_evidence_class",
        "has_positive_pubmed_support",
        "has_direct_pubmed_support",
        "has_null_evidence",
        "retrieved_unique_pmids",
        "direct_supporting_pmids",
        "indirect_supporting_pmids",
        "context_dependent_pmids",
        "null_pmids",
        "review_notes",
        "kegg_pathway_count",
        "reactome_pathway_count",
        "total_pathway_count",
        "pathway_sources",
        "pathway_interpretation",
        "kegg_drug_id",
        "drug_mapping_status",
        "kegg_biological_drug_pathway_count",
        "overlapping_kegg_pathway_count",
        "has_kegg_drug_gene_pathway_overlap",
        "kegg_pair_pathway_result",
        "kegg_pair_interpretation",
        "pathway_names",
    ]

    remaining_columns = [
        column
        for column in final.columns
        if column not in preferred_columns
    ]

    final = final[
        preferred_columns + remaining_columns
    ]

    final.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    summary = (
        final.groupby(
            [
                "positive_support_score",
                "external_validation_category",
                "pubmed_pair_evidence_class",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            candidate_pairs=(
                "external_validation_rank",
                "nunique",
            ),
            unique_drugs=(
                "drug_name",
                "nunique",
            ),
            unique_genes=(
                "gene_name",
                "nunique",
            ),
            median_validation_priority_score=(
                "validation_priority_score",
                "median",
            ),
        )
        .sort_values(
            [
                "positive_support_score",
                "candidate_pairs",
            ],
            ascending=[
                False,
                False,
            ],
        )
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print("FINAL EXTERNAL-VALIDATION TABLE COMPLETE")
    print("Candidate pairs:", len(final))
    print(
        "Pairs with positive PubMed support:",
        int(final["has_positive_pubmed_support"].sum()),
    )
    print(
        "Pairs with direct PubMed support:",
        int(final["has_direct_pubmed_support"].sum()),
    )
    print(
        "Pairs with null evidence:",
        int(final["has_null_evidence"].sum()),
    )
    print(
        "Pairs with exact KEGG overlap:",
        int(
            final[
                "has_kegg_drug_gene_pathway_overlap"
            ]
            .fillna(False)
            .sum()
        ),
    )

    print("\nEVIDENCE-CLASS SUMMARY")
    print(summary.to_string(index=False))

    print("\nINTEGRATED RANKING")
    print(
        final[
            [
                "integrated_evidence_rank",
                "external_validation_rank",
                "drug_name",
                "gene_name",
                "pubmed_pair_evidence_class",
                "positive_support_score",
                "validation_priority_score",
                "total_pathway_count",
                "kegg_pair_interpretation",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(OUTPUT_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
