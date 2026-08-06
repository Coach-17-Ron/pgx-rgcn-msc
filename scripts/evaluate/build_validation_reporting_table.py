#!/usr/bin/env python
"""Create a concise manuscript-ready external-validation table."""

from pathlib import Path

import pandas as pd


ROOT = Path(
    "results/consensus_inference/molecular/"
    "strict_cold/summary"
)

INPUT = ROOT / "final_external_validation_evidence_table.csv"
OUTPUT = ROOT / "manuscript_external_validation_table.csv"


def main() -> None:
    df = pd.read_csv(INPUT, low_memory=False)

    evidence_order = {
        "direct_supported": 1,
        "direct_mixed_evidence": 2,
        "indirect_mechanistic_support": 3,
        "direct_null_only": 4,
        "no_supporting_pair_evidence": 5,
        "no_evidence_found_in_current_search": 6,
    }

    df["evidence_order"] = (
        df["pubmed_pair_evidence_class"]
        .map(evidence_order)
        .fillna(99)
    )

    report = df[
        [
            "external_validation_rank",
            "drug_name",
            "gene_name",
            "weighted_consensus_rank",
            "weighted_rrf_score",
            "gcn_rank",
            "rgcn_rank",
            "rgat_rank",
            "pubmed_pair_evidence_class",
            "direct_supporting_pmids",
            "indirect_supporting_pmids",
            "context_dependent_pmids",
            "null_pmids",
            "kegg_pathway_count",
            "reactome_pathway_count",
            "kegg_pair_interpretation",
            "evidence_order",
        ]
    ].copy()

    report = report.sort_values(
        [
            "evidence_order",
            "weighted_consensus_rank",
            "external_validation_rank",
        ]
    )

    report.insert(
        0,
        "reporting_rank",
        range(1, len(report) + 1),
    )

    report.to_csv(OUTPUT, index=False)

    print("MMANUSCRIPT VALIDATION TABLE COMPLETE")
    print("Rows:", len(report))

    print("\nEVIDENCE COUNTS")
    print(
        report["pubmed_pair_evidence_class"]
        .value_counts()
        .to_string()
    )

    print("\nREPORTING TABLE")
    print(
        report[
            [
                "reporting_rank",
                "drug_name",
                "gene_name",
                "weighted_consensus_rank",
                "pubmed_pair_evidence_class",
                "direct_supporting_pmids",
                "null_pmids",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
