#!/usr/bin/env python
"""Audit drug-specific shared host-variant phenotype support."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


ROOT = Path(
    "results/consensus_inference/molecular/"
    "strict_cold/summary"
)

RANKED_PATH = ROOT / "ranked_gene_centred_phenotypes.csv"
HOST_PATH = ROOT / "canonical_host_variant_phenotype_paths.csv"
EDGE_PATH = Path("artifacts/graph/edges.parquet")


def combine_values(values: pd.Series) -> str:
    collected: set[str] = set()

    for value in values.dropna():
        for item in re.split(r"[;,|]", str(value)):
            item = item.strip()
            if item and item.lower() != "nan":
                collected.add(item)

    return ";".join(sorted(collected))


def main() -> None:
    ranked = pd.read_csv(
        RANKED_PATH,
        low_memory=False,
    )

    host_paths = pd.read_csv(
        HOST_PATH,
        low_memory=False,
    )

    edges = pd.read_parquet(EDGE_PATH)

    drug_variant = edges.loc[
        (
            edges["source_type"].eq("drug")
            & edges["target_type"].eq("variant")
        )
        | (
            edges["source_type"].eq("variant")
            & edges["target_type"].eq("drug")
        )
    ].copy()

    drug_variant["drug_idx"] = (
        drug_variant["source_idx"].where(
            drug_variant["source_type"].eq("drug"),
            drug_variant["target_idx"],
        ).astype(int)
    )

    drug_variant["variant_idx"] = (
        drug_variant["target_idx"].where(
            drug_variant["target_type"].eq("variant"),
            drug_variant["source_idx"],
        ).astype(int)
    )

    drug_variant = (
        drug_variant.groupby(
            ["drug_idx", "variant_idx"],
            as_index=False,
        )
        .agg(
            drug_variant_relation_types=(
                "relation_type",
                combine_values,
            ),
            drug_variant_evidence=(
                "evidence",
                combine_values,
            ),
            drug_variant_associations=(
                "association",
                combine_values,
            ),
            drug_variant_pmids=(
                "pmids",
                combine_values,
            ),
        )
    )

    host_keys = host_paths[
        [
            "gene_idx",
            "variant_idx",
            "variant_id",
            "variant_name",
            "phenotype_idx",
            "phenotype_id",
            "phenotype_name",
        ]
    ].drop_duplicates()

    shared_support = (
        ranked.merge(
            host_keys,
            on=[
                "gene_idx",
                "phenotype_idx",
                "phenotype_id",
                "phenotype_name",
            ],
            how="left",
            suffixes=("", "_host"),
        )
        .merge(
            drug_variant,
            on=["drug_idx", "variant_idx"],
            how="left",
        )
    )

    shared_support[
        "has_shared_host_variant_support"
    ] = shared_support[
        "drug_variant_evidence"
    ].notna()

    supported = shared_support.loc[
        shared_support[
            "has_shared_host_variant_support"
        ]
    ].copy()

    canonical_supported = (
        supported.groupby(
            [
                "seed",
                "drug_idx",
                "drug_id",
                "drug_name",
                "gene_idx",
                "gene_id",
                "gene_name",
                "phenotype_idx",
                "phenotype_id",
                "phenotype_name",
                "weighted_consensus_rank",
                "phenotype_priority_score",
                "within_pair_phenotype_rank",
                "within_drug_phenotype_rank",
                "primary_association",
                "phenotype_context_type",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            matching_host_variants=(
                "variant_idx",
                "nunique",
            ),
            matching_variant_ids=(
                "variant_id",
                combine_values,
            ),
            matching_variant_names=(
                "variant_name",
                combine_values,
            ),
            drug_variant_evidence=(
                "drug_variant_evidence",
                combine_values,
            ),
            drug_variant_associations=(
                "drug_variant_associations",
                combine_values,
            ),
            drug_variant_pmids=(
                "drug_variant_pmids",
                combine_values,
            ),
        )
    )

    pair_support = (
        canonical_supported[
            [
                "seed",
                "drug_idx",
                "gene_idx",
            ]
        ]
        .drop_duplicates()
    )

    drug_support = (
        canonical_supported[
            ["seed", "drug_idx"]
        ]
        .drop_duplicates()
    )

    output = (
        ROOT
        / "shared_host_variant_drug_phenotype_support.csv"
    )

    canonical_supported.sort_values(
        [
            "within_drug_phenotype_rank",
            "phenotype_priority_score",
        ],
        ascending=[True, False],
    ).to_csv(
        output,
        index=False,
    )

    print("SHARED HOST-VARIANT SUPPORT AUDIT")
    print("Ranked phenotype rows:", len(ranked))
    print(
        "Supported phenotype rows:",
        len(canonical_supported),
    )
    print(
        "Supported candidate pairs:",
        len(pair_support),
    )
    print(
        "Supported drug evaluations:",
        len(drug_support),
    )
    print(
        "Unique matching variants:",
        canonical_supported[
            "matching_variant_ids"
        ].nunique(),
    )

    print("\nSUPPORT BY PHENOTYPE RANK")
    if len(canonical_supported):
        print(
            ranked.assign(
                supported=ranked.set_index(
                    [
                        "seed",
                        "drug_idx",
                        "gene_idx",
                        "phenotype_idx",
                    ]
                ).index.isin(
                    canonical_supported.set_index(
                        [
                            "seed",
                            "drug_idx",
                            "gene_idx",
                            "phenotype_idx",
                        ]
                    ).index
                )
            )
            .groupby(
                "within_drug_phenotype_rank"
            )["supported"]
            .agg(["sum", "count", "mean"])
            .head(20)
            .to_string()
        )
    else:
        print("No shared host-variant support found.")

    print("\nTOP SUPPORTED CONTEXTS")
    if len(canonical_supported):
        print(
            canonical_supported[
                [
                    "seed",
                    "drug_name",
                    "gene_name",
                    "phenotype_name",
                    "weighted_consensus_rank",
                    "within_drug_phenotype_rank",
                    "matching_variant_names",
                    "drug_variant_evidence",
                    "drug_variant_associations",
                    "phenotype_priority_score",
                ]
            ]
            .sort_values(
                [
                    "within_drug_phenotype_rank",
                    "phenotype_priority_score",
                ],
                ascending=[True, False],
            )
            .head(100)
            .to_string(index=False)
        )
    else:
        print("No supported contexts.")

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
