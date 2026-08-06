#!/usr/bin/env python
"""Audit seed eligibility and relation coverage for pair-level explanations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch


SEEDS = [42, 123, 2026]

SUMMARY_ROOT = Path(
    "results/consensus_inference/molecular/"
    "strict_cold/summary"
)

PAIR_PATH = (
    SUMMARY_ROOT
    / "external_validation_candidate_pairs.csv"
)

SPLIT_ROOT = Path("artifacts/splits/strict_cold")
TENSOR_ROOT = Path("artifacts/tensors_molecular/strict_cold")

COVERAGE_OUTPUT = (
    SUMMARY_ROOT
    / "interpretability_candidate_seed_coverage.csv"
)

RELATION_OUTPUT = (
    SUMMARY_ROOT
    / "interpretability_relation_edge_counts.csv"
)


def main() -> None:
    pairs = pd.read_csv(
        PAIR_PATH,
        low_memory=False,
    )

    pair_columns = [
        "external_validation_rank",
        "drug_idx",
        "drug_id",
        "drug_name",
        "gene_idx",
        "gene_id",
        "gene_name",
        "pubmed_pair_evidence_class",
    ]

    # The original candidate file may predate PubMed integration.
    available_columns = [
        column
        for column in pair_columns
        if column in pairs.columns
    ]

    pairs = pairs[available_columns].drop_duplicates(
        [
            "external_validation_rank",
            "drug_idx",
            "gene_idx",
        ]
    )

    coverage_rows: list[dict[str, object]] = []
    relation_rows: list[dict[str, object]] = []

    for seed in SEEDS:
        print(f"Auditing seed {seed}...")

        split_dir = SPLIT_ROOT / f"seed_{seed}"
        tensor_path = (
            TENSOR_ROOT
            / f"seed_{seed}"
            / "tensor_bundle.pt"
        )

        assignments = pd.read_parquet(
            split_dir / "drug_assignments.parquet"
        )

        assignment_lookup = (
            assignments.set_index("drug_idx")["split"]
            .astype(str)
            .to_dict()
        )

        bundle = torch.load(
            tensor_path,
            map_location="cpu",
            weights_only=False,
        )

        edge_type = bundle["edge_type"].long()

        relation_to_index = bundle.get(
            "relation_to_index"
        )

        if relation_to_index is None:
            raise KeyError(
                f"relation_to_index missing from {tensor_path}"
            )

        index_to_relation = {
            int(index): str(relation)
            for relation, index
            in relation_to_index.items()
        }

        counts = torch.bincount(
            edge_type,
            minlength=len(index_to_relation),
        )

        for relation_index in range(
            len(index_to_relation)
        ):
            relation_rows.append(
                {
                    "seed": seed,
                    "relation_index": relation_index,
                    "relation_name": index_to_relation[
                        relation_index
                    ],
                    "edge_count": int(
                        counts[relation_index].item()
                    ),
                }
            )

        for row in pairs.itertuples(index=False):
            drug_idx = int(row.drug_idx)
            split = assignment_lookup.get(
                drug_idx,
                "not_in_assignment_table",
            )

            record = {
                "seed": seed,
                "external_validation_rank": int(
                    row.external_validation_rank
                ),
                "drug_idx": drug_idx,
                "drug_id": row.drug_id,
                "drug_name": row.drug_name,
                "gene_idx": int(row.gene_idx),
                "gene_id": row.gene_id,
                "gene_name": row.gene_name,
                "drug_split": split,
                "eligible_for_strict_cold_explanation": (
                    split == "test"
                ),
            }

            if hasattr(
                row,
                "pubmed_pair_evidence_class",
            ):
                record[
                    "pubmed_pair_evidence_class"
                ] = row.pubmed_pair_evidence_class

            coverage_rows.append(record)

    coverage = pd.DataFrame(coverage_rows)
    relations = pd.DataFrame(relation_rows)

    coverage.to_csv(
        COVERAGE_OUTPUT,
        index=False,
    )

    relations.to_csv(
        RELATION_OUTPUT,
        index=False,
    )

    eligible = coverage.loc[
        coverage[
            "eligible_for_strict_cold_explanation"
        ]
    ]

    pair_seed_counts = (
        eligible.groupby(
            [
                "external_validation_rank",
                "drug_name",
                "gene_name",
            ],
            as_index=False,
        )["seed"]
        .nunique()
        .rename(
            columns={
                "seed": "eligible_seed_count"
            }
        )
    )

    print("\nINTERPRETABILITY SEED-COVERAGE AUDIT COMPLETE")
    print("Candidate pairs:", pairs.shape[0])
    print(
        "Eligible pair–seed evaluations:",
        eligible.shape[0],
    )
    print(
        "Pairs eligible in at least one seed:",
        pair_seed_counts.shape[0],
    )

    print("\nELIGIBLE-SEED COUNT DISTRIBUTION")
    print(
        pair_seed_counts[
            "eligible_seed_count"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    missing_pairs = (
        set(pairs["external_validation_rank"])
        - set(
            pair_seed_counts[
                "external_validation_rank"
            ]
        )
    )

    print("\nPAIRS WITH NO ELIGIBLE STRICT-COLD SEED")
    if missing_pairs:
        print(sorted(missing_pairs))
    else:
        print("None")

    print("\nPAIR COVERAGE")
    print(
        pair_seed_counts.sort_values(
            "external_validation_rank"
        ).to_string(index=False)
    )

    print("\nRELATION EDGE-COUNT RANGE ACROSS SEEDS")
    relation_summary = (
        relations.groupby(
            "relation_name",
            as_index=False,
        )
        .agg(
            minimum_edges=("edge_count", "min"),
            maximum_edges=("edge_count", "max"),
            mean_edges=("edge_count", "mean"),
        )
        .sort_values(
            "mean_edges",
            ascending=False,
        )
    )

    print(
        relation_summary.head(20).to_string(
            index=False
        )
    )

    print("\nZERO-EDGE RELATIONS")
    zero = relation_summary.loc[
        relation_summary["maximum_edges"].eq(0)
    ]

    if zero.empty:
        print("None")
    else:
        print(
            zero["relation_name"].to_string(
                index=False
            )
        )

    print("\nSaved:")
    print(COVERAGE_OUTPUT)
    print(RELATION_OUTPUT)


if __name__ == "__main__":
    main()
