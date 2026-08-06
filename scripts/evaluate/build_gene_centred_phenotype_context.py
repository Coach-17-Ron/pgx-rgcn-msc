#!/usr/bin/env python
"""Build canonical gene-centred phenotype context for drug–gene candidates."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


ROOT = Path(
    "results/consensus_inference/molecular/"
    "strict_cold/summary"
)

CANDIDATE_PATH = ROOT / "phenotype_propagation_candidates.csv"
EDGE_PATH = Path("artifacts/graph/edges.parquet")
NODE_PATH = Path("artifacts/graph/nodes.parquet")
VARIANT_PATH = ROOT / "canonical_host_variant_phenotype_paths.csv"


def combine_values(values: pd.Series) -> str:
    collected: set[str] = set()

    for value in values.dropna():
        for item in re.split(r"[;,|]", str(value)):
            item = item.strip()
            if item and item.lower() != "nan":
                collected.add(item)

    return ";".join(sorted(collected))


def canonical_gene_phenotype(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
) -> pd.DataFrame:
    subset = edges.loc[
        (
            edges["source_type"].eq("gene")
            & edges["target_type"].eq("phenotype")
        )
        | (
            edges["source_type"].eq("phenotype")
            & edges["target_type"].eq("gene")
        )
    ].copy()

    subset["gene_idx"] = subset["source_idx"].where(
        subset["source_type"].eq("gene"),
        subset["target_idx"],
    ).astype(int)

    subset["phenotype_idx"] = subset["target_idx"].where(
        subset["target_type"].eq("phenotype"),
        subset["source_idx"],
    ).astype(int)

    phenotype_nodes = nodes.loc[
        nodes["node_type"].eq("phenotype"),
        ["node_index", "node_id", "node_name"],
    ].rename(
        columns={
            "node_index": "phenotype_idx",
            "node_id": "phenotype_id",
            "node_name": "phenotype_name",
        }
    )

    subset = subset.merge(
        phenotype_nodes,
        on="phenotype_idx",
        how="left",
        validate="many_to_one",
    )

    return (
        subset.groupby(
            [
                "gene_idx",
                "phenotype_idx",
                "phenotype_id",
                "phenotype_name",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            direct_relation_types=(
                "relation_type",
                combine_values,
            ),
            direct_evidence=(
                "evidence",
                combine_values,
            ),
            direct_associations=(
                "association",
                combine_values,
            ),
            direct_pmids=(
                "pmids",
                combine_values,
            ),
        )
    )


def main() -> None:
    candidates = pd.read_csv(CANDIDATE_PATH)
    edges = pd.read_parquet(EDGE_PATH)
    nodes = pd.read_parquet(NODE_PATH)

    direct = canonical_gene_phenotype(
        edges,
        nodes,
    )

    variant = pd.read_csv(
        VARIANT_PATH,
        low_memory=False,
    )

    variant_summary = (
        variant.groupby(
            [
                "gene_idx",
                "phenotype_idx",
                "phenotype_id",
                "phenotype_name",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            verified_host_variants=(
                "variant_idx",
                "nunique",
            ),
            host_variant_ids=(
                "variant_id",
                combine_values,
            ),
            host_variant_names=(
                "variant_name",
                combine_values,
            ),
            variant_relation_types=(
                "relation_type",
                combine_values,
            ),
            variant_evidence=(
                "evidence",
                combine_values,
            ),
            variant_associations=(
                "association",
                combine_values,
            ),
            variant_pmids=(
                "pmids",
                combine_values,
            ),
        )
    )

    gene_phenotype_context = direct.merge(
        variant_summary,
        on=[
            "gene_idx",
            "phenotype_idx",
            "phenotype_id",
            "phenotype_name",
        ],
        how="outer",
        indicator=True,
    )

    gene_phenotype_context["phenotype_context_type"] = (
        gene_phenotype_context["_merge"].map(
            {
                "left_only": "direct_gene_phenotype",
                "right_only": "host_variant_phenotype",
                "both": "direct_and_host_variant",
            }
        )
    )

    gene_phenotype_context = (
        gene_phenotype_context.drop(columns="_merge")
    )

    candidate_columns = [
        column
        for column in [
            "seed",
            "drug_idx",
            "drug_id",
            "drug_name",
            "gene_idx",
            "gene_id",
            "gene_name",
            "weighted_consensus_rank",
            "weighted_rrf_score",
            "candidate_class",
            "within_drug_priority_rank",
            "specificity_priority_score",
            "models_top_25",
            "gcn_rank",
            "rgcn_rank",
            "rgat_rank",
        ]
        if column in candidates.columns
    ]

    candidate_keys = candidates[
        candidate_columns
    ].drop_duplicates(
        ["seed", "drug_idx", "gene_idx"]
    )

    expanded = candidate_keys.merge(
        gene_phenotype_context,
        on="gene_idx",
        how="left",
    )

    expanded["phenotype_context_type"] = (
        expanded["phenotype_context_type"]
        .astype("string")
        .fillna("no_graph_context")
    )

    expanded["has_direct_support"] = (
        expanded["phenotype_context_type"].isin(
            [
                "direct_gene_phenotype",
                "direct_and_host_variant",
            ]
        )
    )

    expanded["has_host_variant_support"] = (
        expanded["phenotype_context_type"].isin(
            [
                "host_variant_phenotype",
                "direct_and_host_variant",
            ]
        )
    )

    pair_summary = (
        expanded.groupby(
            [
                "seed",
                "drug_idx",
                "drug_id",
                "drug_name",
                "gene_idx",
                "gene_id",
                "gene_name",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            contextualised_phenotypes=(
                "phenotype_idx",
                "nunique",
            ),
            direct_supported_phenotypes=(
                "has_direct_support",
                "sum",
            ),
            host_variant_supported_phenotypes=(
                "has_host_variant_support",
                "sum",
            ),
            verified_host_variants=(
                "verified_host_variants",
                "max",
            ),
        )
    )

    context_path = (
        ROOT / "gene_centred_phenotype_context.csv"
    )

    pair_summary_path = (
        ROOT / "candidate_phenotype_context_summary.csv"
    )

    gene_context_path = (
        ROOT / "canonical_gene_phenotype_context.csv"
    )

    gene_phenotype_context.to_csv(
        gene_context_path,
        index=False,
    )

    expanded.to_csv(
        context_path,
        index=False,
    )

    pair_summary.to_csv(
        pair_summary_path,
        index=False,
    )

    print("GENE-CENTRED PHENOTYPE CONTEXT COMPLETE")
    print("Candidate pairs:", len(candidate_keys))
    print(
        "Expanded candidate–phenotype rows:",
        len(expanded),
    )
    print(
        "Unique contextualised phenotypes:",
        expanded["phenotype_idx"].nunique(),
    )

    print("\nCONTEXT TYPE COUNTS")
    print(
        expanded["phenotype_context_type"]
        .value_counts()
        .to_string()
    )

    print("\nCANDIDATE-PAIR COVERAGE")
    print(
        pair_summary[
            [
                "contextualised_phenotypes",
                "direct_supported_phenotypes",
                "host_variant_supported_phenotypes",
            ]
        ]
        .describe()
        .to_string()
    )

    print("\nSaved:")
    print(gene_context_path)
    print(context_path)
    print(pair_summary_path)


if __name__ == "__main__":
    main()
