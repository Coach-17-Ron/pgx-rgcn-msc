#!/usr/bin/env python
"""Combine article-level PubMed review into pair-level evidence summaries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(
    "results/consensus_inference/molecular/"
    "strict_cold/summary"
)

PUBMED_DIR = ROOT / "pubmed_validation"
MANIFEST_PATH = ROOT / "external_validation_manifest.csv"

REVIEW_PATHS = [
    PUBMED_DIR / "pubmed_manual_review_ranks_1_10.csv",
    PUBMED_DIR / "pubmed_manual_review_ranks_11_20.csv",
    PUBMED_DIR / "pubmed_manual_review_ranks_21_30.csv",
]


def combine_unique(values: pd.Series) -> str:
    return ";".join(
        sorted(
            {
                str(value).strip()
                for value in values.dropna()
                if str(value).strip()
            }
        )
    )


def pair_class(group: pd.DataFrame) -> str:
    classes = set(
        group["manual_evidence_class"]
        .dropna()
        .astype(str)
    )

    if "direct_support" in classes:
        if classes & {"direct_null", "direct_mixed"}:
            return "direct_mixed_evidence"
        return "direct_supported"

    if "direct_mixed" in classes:
        return "direct_mixed_evidence"

    if "indirect_mechanistic" in classes:
        return "indirect_mechanistic_support"

    if "direct_null" in classes:
        return "direct_null_only"

    return "no_supporting_pair_evidence"


def main() -> None:
    manifest = pd.read_csv(
        MANIFEST_PATH,
        low_memory=False,
    )

    reviews = pd.concat(
        [
            pd.read_csv(
                path,
                low_memory=False,
                keep_default_na=False,
            )
            for path in REVIEW_PATHS
        ],
        ignore_index=True,
    )

    reviews["manual_evidence_class"] = (
        reviews["manual_evidence_class"]
        .astype("string")
    )

    for column in [
        "manual_evidence_class",
        "manual_support_direction",
    ]:
        reviews[column] = (
            reviews[column]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    reviewed = reviews.loc[
        reviews["manual_evidence_class"].ne("")
        & reviews["manual_evidence_class"].ne("nan")
    ].copy()

    pair_rows = []

    for rank, group in reviewed.groupby(
        "external_validation_rank"
    ):
        pair_rows.append(
            {
                "external_validation_rank": int(rank),
                "pubmed_pair_evidence_class": pair_class(group),
                "retrieved_unique_pmids": group["pmid"].nunique(),
                "direct_supporting_pmids": combine_unique(
                    group.loc[
                        group["manual_support_direction"].eq(
                            "supportive"
                        ),
                        "pmid",
                    ]
                ),
                "indirect_supporting_pmids": combine_unique(
                    group.loc[
                        group["manual_support_direction"].eq(
                            "supportive_indirect"
                        ),
                        "pmid",
                    ]
                ),
                "context_dependent_pmids": combine_unique(
                    group.loc[
                        group["manual_support_direction"].eq(
                            "context_dependent"
                        ),
                        "pmid",
                    ]
                ),
                "null_pmids": combine_unique(
                    group.loc[
                        group["manual_support_direction"].eq("null"),
                        "pmid",
                    ]
                ),
                "article_evidence_classes": combine_unique(
                    group["manual_evidence_class"]
                ),
            }
        )

    pair_summary = pd.DataFrame(pair_rows)

    # Remove placeholder columns already present in the manifest
    # so reviewed PubMed summary columns retain their expected names.
    overlapping_columns = [
        column
        for column in pair_summary.columns
        if (
            column != "external_validation_rank"
            and column in manifest.columns
        )
    ]

    manifest = manifest.drop(
        columns=overlapping_columns,
        errors="ignore",
    )

    result = manifest.merge(
        pair_summary,
        on="external_validation_rank",
        how="left",
    )

    result["pubmed_pair_evidence_class"] = (
        result["pubmed_pair_evidence_class"]
        .fillna("no_evidence_found_in_current_search")
    )

    result["retrieved_unique_pmids"] = (
        result["retrieved_unique_pmids"]
        .fillna(0)
        .astype(int)
    )

    output = ROOT / "pubmed_pair_validation_summary.csv"

    result.to_csv(
        output,
        index=False,
    )

    print("PUBMED PAIR SUMMARY COMPLETE")
    print(
        result[
            [
                "external_validation_rank",
                "drug_name",
                "gene_name",
                "pubmed_pair_evidence_class",
                "retrieved_unique_pmids",
                "direct_supporting_pmids",
                "indirect_supporting_pmids",
                "context_dependent_pmids",
                "null_pmids",
            ]
        ].to_string(index=False)
    )

    print("\nEVIDENCE CLASS COUNTS")
    print(
        result["pubmed_pair_evidence_class"]
        .value_counts()
        .to_string()
    )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
