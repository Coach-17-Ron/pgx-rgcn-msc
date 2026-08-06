#!/usr/bin/env python
"""Audit phenotype paths using authoritative variant host-gene mappings."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


CANDIDATE_PATH = Path(
    "results/consensus_inference/molecular/strict_cold/"
    "summary/phenotype_propagation_candidates.csv"
)
EDGE_PATH = Path("artifacts/graph/edges.parquet")
NODE_PATH = Path("artifacts/graph/nodes.parquet")
VARIANT_PATH = Path("data/variants.tsv")


def split_values(value: object) -> list[str]:
    if pd.isna(value):
        return []

    return [
        item.strip()
        for item in re.split(r"[,;|]", str(value))
        if item.strip()
    ]


def canonical_variant_phenotype(
    edges: pd.DataFrame,
) -> pd.DataFrame:
    subset = edges.loc[
        (
            edges["source_type"].eq("variant")
            & edges["target_type"].eq("phenotype")
        )
        | (
            edges["source_type"].eq("phenotype")
            & edges["target_type"].eq("variant")
        )
    ].copy()

    subset["variant_idx"] = subset[
        "source_idx"
    ].where(
        subset["source_type"].eq("variant"),
        subset["target_idx"],
    ).astype(int)

    subset["phenotype_idx"] = subset[
        "target_idx"
    ].where(
        subset["target_type"].eq("phenotype"),
        subset["source_idx"],
    ).astype(int)

    return subset


def main() -> None:
    candidates = pd.read_csv(CANDIDATE_PATH)

    edges = pd.read_parquet(EDGE_PATH)

    nodes = pd.read_parquet(NODE_PATH)[
        [
            "node_index",
            "node_id",
            "node_name",
            "node_type",
        ]
    ]

    variants = pd.read_csv(
        VARIANT_PATH,
        sep="\t",
        low_memory=False,
    )

    variant_nodes = nodes.loc[
        nodes["node_type"].eq("variant"),
        ["node_index", "node_id", "node_name"],
    ].rename(
        columns={
            "node_index": "variant_idx",
            "node_id": "variant_id",
            "node_name": "variant_name",
        }
    )

    gene_nodes = nodes.loc[
        nodes["node_type"].eq("gene"),
        ["node_index", "node_id", "node_name"],
    ].rename(
        columns={
            "node_index": "gene_idx",
            "node_id": "gene_id",
            "node_name": "gene_name",
        }
    )

    host_rows: list[dict[str, object]] = []

    for _, row in variants.iterrows():
        gene_ids = split_values(row.get("Gene IDs"))
        gene_symbols = split_values(
            row.get("Gene Symbols")
        )

        maximum = max(
            len(gene_ids),
            len(gene_symbols),
        )

        for position in range(maximum):
            host_rows.append(
                {
                    "variant_id": str(
                        row["Variant ID"]
                    ),
                    "variant_name": str(
                        row["Variant Name"]
                    ),
                    "host_gene_id": (
                        gene_ids[position]
                        if position < len(gene_ids)
                        else None
                    ),
                    "host_gene_symbol": (
                        gene_symbols[position]
                        if position < len(gene_symbols)
                        else None
                    ),
                    "variant_location": row.get(
                        "Location"
                    ),
                }
            )

    host_mapping = pd.DataFrame(host_rows)

    host_mapping = host_mapping.merge(
        variant_nodes,
        on=["variant_id", "variant_name"],
        how="inner",
        validate="many_to_one",
    )

    host_mapping = host_mapping.merge(
        gene_nodes,
        left_on="host_gene_id",
        right_on="gene_id",
        how="left",
    )

    missing_gene = host_mapping["gene_idx"].isna()

    if missing_gene.any():
        symbol_lookup = (
            gene_nodes.drop_duplicates("gene_name")
            .set_index("gene_name")["gene_idx"]
        )

        host_mapping.loc[
            missing_gene,
            "gene_idx",
        ] = host_mapping.loc[
            missing_gene,
            "host_gene_symbol",
        ].map(symbol_lookup)

    host_mapping = host_mapping.dropna(
        subset=["gene_idx"]
    ).copy()

    host_mapping["gene_idx"] = host_mapping[
        "gene_idx"
    ].astype(int)

    variant_phenotype = canonical_variant_phenotype(
        edges
    )

    host_variant_phenotype = (
        host_mapping.merge(
            variant_phenotype[
                [
                    "variant_idx",
                    "phenotype_idx",
                    "relation_type",
                    "evidence",
                    "association",
                    "pmids",
                ]
            ],
            on="variant_idx",
            how="inner",
        )
        .merge(
            nodes[
                [
                    "node_index",
                    "node_id",
                    "node_name",
                ]
            ],
            left_on="phenotype_idx",
            right_on="node_index",
            how="left",
            validate="many_to_one",
        )
        .rename(
            columns={
                "node_id": "phenotype_id",
                "node_name": "phenotype_name",
            }
        )
    )

    candidate_keys = candidates[
        [
            "seed",
            "drug_idx",
            "drug_id",
            "drug_name",
            "gene_idx",
            "gene_id",
            "gene_name",
            "weighted_consensus_rank",
            "candidate_class",
            "within_drug_priority_rank",
        ]
    ].drop_duplicates()

    confirmed = candidate_keys.merge(
        host_variant_phenotype,
        on="gene_idx",
        how="inner",
        suffixes=("", "_host"),
    )

    pair_columns = [
        "seed",
        "drug_idx",
        "gene_idx",
    ]

    confirmed_pairs = confirmed[
        pair_columns
    ].drop_duplicates()

    print("HOST-GENE MAPPING")
    print(
        "variants with graph host mappings:",
        host_mapping["variant_idx"].nunique(),
    )
    print(
        "unique mapped host genes:",
        host_mapping["gene_idx"].nunique(),
    )

    print("\nCANDIDATE AUDIT")
    print(
        "candidate pairs:",
        len(candidate_keys),
    )
    print(
        "host-confirmed gene-variant-phenotype pairs:",
        len(confirmed_pairs),
    )
    print(
        "host-confirmed candidate genes:",
        confirmed_pairs["gene_idx"].nunique(),
    )
    print(
        "host-confirmed path rows:",
        len(confirmed),
    )

    print("\nHOST-CONFIRMED GENES")
    gene_summary = (
        confirmed.groupby(
            ["gene_idx", "gene_name"],
            as_index=False,
        )
        .agg(
            candidate_pairs=(
                "drug_idx",
                "size",
            ),
            unique_variants=(
                "variant_idx",
                "nunique",
            ),
            unique_phenotypes=(
                "phenotype_idx",
                "nunique",
            ),
        )
        .sort_values(
            [
                "candidate_pairs",
                "unique_variants",
            ],
            ascending=False,
        )
    )

    print(
        gene_summary.head(30).to_string(
            index=False
        )
    )

    output = (
        "results/consensus_inference/molecular/"
        "strict_cold/summary/"
        "host_confirmed_variant_phenotype_paths.csv"
    )

    confirmed.drop(
        columns=["node_index"],
        errors="ignore",
    ).drop_duplicates().to_csv(
        output,
        index=False,
    )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
