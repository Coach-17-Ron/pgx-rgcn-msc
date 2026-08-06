#!/usr/bin/env python
"""Combine KEGG and human Reactome pathways for shortlisted drug–gene pairs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(
    "results/consensus_inference/molecular/"
    "strict_cold/summary"
)

PAIR_PATH = ROOT / "pubmed_pair_validation_summary.csv"
KEGG_PATH = ROOT / "kegg_gene_pathway_results.csv"
REACTOME_PATH = ROOT / "reactome_gene_pathway_results.csv"

LONG_OUTPUT = ROOT / "combined_gene_pathway_catalogue.csv"
PAIR_OUTPUT = ROOT / "candidate_pair_pathway_summary.csv"


def combine_unique(values: pd.Series) -> str:
    return ";".join(
        sorted(
            {
                str(value).strip()
                for value in values.dropna()
                if str(value).strip()
                and str(value).strip().lower() != "nan"
            }
        )
    )


def main() -> None:
    pairs = pd.read_csv(
        PAIR_PATH,
        low_memory=False,
    )

    kegg = pd.read_csv(
        KEGG_PATH,
        low_memory=False,
    )

    reactome = pd.read_csv(
        REACTOME_PATH,
        low_memory=False,
    )

    kegg_long = (
        kegg.loc[
            kegg["mapping_status"].eq("mapped"),
            [
                "gene_name",
                "kegg_pathway_id",
                "kegg_pathway_name",
            ],
        ]
        .drop_duplicates()
        .rename(
            columns={
                "kegg_pathway_id": "pathway_id",
                "kegg_pathway_name": "pathway_name",
            }
        )
    )

    kegg_long["pathway_source"] = "KEGG"

    reactome_long = (
        reactome.loc[
            reactome["mapping_status"].eq("mapped"),
            [
                "gene_name",
                "reactome_pathway_id",
                "reactome_pathway_name",
            ],
        ]
        .drop_duplicates()
        .rename(
            columns={
                "reactome_pathway_id": "pathway_id",
                "reactome_pathway_name": "pathway_name",
            }
        )
    )

    reactome_long["pathway_source"] = "Reactome"

    pathways = pd.concat(
        [kegg_long, reactome_long],
        ignore_index=True,
    ).drop_duplicates(
        [
            "gene_name",
            "pathway_source",
            "pathway_id",
        ]
    )

    catalogue = pairs.merge(
        pathways,
        on="gene_name",
        how="left",
    )

    catalogue["pathway_mapping_status"] = (
        catalogue["pathway_id"]
        .notna()
        .map(
            {
                True: "mapped",
                False: "no_pathway_mapping",
            }
        )
    )

    catalogue.to_csv(
        LONG_OUTPUT,
        index=False,
    )

    pair_summary = (
        catalogue.groupby(
            [
                "external_validation_rank",
                "drug_name",
                "gene_name",
                "pubmed_pair_evidence_class",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            kegg_pathway_count=(
                "pathway_source",
                lambda values: int(
                    (values == "KEGG").sum()
                ),
            ),
            reactome_pathway_count=(
                "pathway_source",
                lambda values: int(
                    (values == "Reactome").sum()
                ),
            ),
            total_pathway_count=(
                "pathway_id",
                lambda values: values.notna().sum(),
            ),
            pathway_sources=(
                "pathway_source",
                combine_unique,
            ),
            pathway_names=(
                "pathway_name",
                combine_unique,
            ),
        )
    )

    pair_summary["has_any_pathway_context"] = (
        pair_summary["total_pathway_count"] > 0
    )

    pair_summary.to_csv(
        PAIR_OUTPUT,
        index=False,
    )

    print("COMBINED PATHWAY CATALOGUE COMPLETE")
    print(
        "Candidate pairs:",
        pair_summary.shape[0],
    )
    print(
        "Pairs with pathway context:",
        int(
            pair_summary[
                "has_any_pathway_context"
            ].sum()
        ),
    )
    print(
        "Pairs without pathway context:",
        int(
            (~pair_summary[
                "has_any_pathway_context"
            ]).sum()
        ),
    )
    print(
        "Unique genes:",
        catalogue["gene_name"].nunique(),
    )
    print(
        "Unique KEGG pathways:",
        pathways.loc[
            pathways["pathway_source"].eq("KEGG"),
            "pathway_id",
        ].nunique(),
    )
    print(
        "Unique Reactome pathways:",
        pathways.loc[
            pathways["pathway_source"].eq("Reactome"),
            "pathway_id",
        ].nunique(),
    )

    print("\nPAIR PATHWAY COUNTS")
    print(
        pair_summary[
            [
                "external_validation_rank",
                "drug_name",
                "gene_name",
                "kegg_pathway_count",
                "reactome_pathway_count",
                "total_pathway_count",
                "pubmed_pair_evidence_class",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(LONG_OUTPUT)
    print(PAIR_OUTPUT)


if __name__ == "__main__":
    main()
