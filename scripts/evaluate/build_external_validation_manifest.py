#!/usr/bin/env python
"""Create a structured manifest for external drug–gene validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(
    "results/consensus_inference/molecular/"
    "strict_cold/summary"
)

INPUT_PATH = ROOT / "external_validation_candidate_pairs.csv"
OUTPUT_PATH = ROOT / "external_validation_manifest.csv"


def quote_term(value: object) -> str:
    text = str(value).strip()
    return f'"{text}"'


def main() -> None:
    pairs = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    pairs = pairs.sort_values(
        "external_validation_rank"
    ).copy()

    pairs["pubmed_direct_query"] = pairs.apply(
        lambda row: (
            f'{quote_term(row["drug_name"])} AND '
            f'{quote_term(row["gene_name"])}'
        ),
        axis=1,
    )

    pairs["pubmed_pgx_query"] = pairs.apply(
        lambda row: (
            f'{quote_term(row["drug_name"])} AND '
            f'{quote_term(row["gene_name"])} AND '
            "(pharmacogenomics OR pharmacogenetics "
            "OR metabolism OR transporter OR response "
            "OR toxicity)"
        ),
        axis=1,
    )

    pairs["pubmed_mechanism_query"] = pairs.apply(
        lambda row: (
            f'{quote_term(row["drug_name"])} AND '
            f'({quote_term(row["gene_name"])} OR '
            f'"{row["gene_name"]} protein") AND '
            "(mechanism OR pathway OR expression "
            "OR inhibition OR induction OR substrate)"
        ),
        axis=1,
    )

    pairs["pathway_gene_query"] = pairs[
        "gene_name"
    ].astype(str)

    pairs["reactome_query"] = (
        "Reactome " + pairs["gene_name"].astype(str)
    )

    pairs["kegg_query"] = (
        "KEGG "
        + pairs["drug_name"].astype(str)
        + " "
        + pairs["gene_name"].astype(str)
    )

    pairs["validation_status"] = "not_reviewed"
    pairs["direct_drug_gene_evidence"] = ""
    pairs["indirect_mechanistic_evidence"] = ""
    pairs["pathway_overlap"] = ""
    pairs["contradictory_evidence"] = ""
    pairs["external_evidence_class"] = ""
    pairs["supporting_pmids"] = ""
    pairs["review_notes"] = ""

    output_columns = [
        "external_validation_rank",
        "seed",
        "drug_id",
        "drug_name",
        "gene_id",
        "gene_name",
        "weighted_consensus_rank",
        "gcn_rank",
        "rgcn_rank",
        "rgat_rank",
        "models_top_25",
        "candidate_class",
        "seeds_observed",
        "validation_priority_score",
        "pubmed_direct_query",
        "pubmed_pgx_query",
        "pubmed_mechanism_query",
        "pathway_gene_query",
        "reactome_query",
        "kegg_query",
        "validation_status",
        "direct_drug_gene_evidence",
        "indirect_mechanistic_evidence",
        "pathway_overlap",
        "contradictory_evidence",
        "external_evidence_class",
        "supporting_pmids",
        "review_notes",
    ]

    output_columns = [
        column
        for column in output_columns
        if column in pairs.columns
    ]

    manifest = pairs[output_columns]

    manifest.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("EXTERNAL VALIDATION MANIFEST COMPLETE")
    print("Pairs:", len(manifest))
    print(
        "Unique drugs:",
        manifest["drug_name"].nunique(),
    )
    print(
        "Unique genes:",
        manifest["gene_name"].nunique(),
    )

    print("\nFIRST VALIDATION QUERIES")
    print(
        manifest[
            [
                "external_validation_rank",
                "drug_name",
                "gene_name",
                "pubmed_direct_query",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
