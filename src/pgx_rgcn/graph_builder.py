"""Build the audited four-node ClinPGx pharmacogenomic graph.

Node types
----------
drug, gene, phenotype, variant

Rules
-----
- Stable PharmGKB IDs are the primary keys.
- Disease relationship endpoints map to phenotype.
- Haplotype relationships are explicitly excluded.
- Missing in-scope endpoint IDs are supplemented from relationships.tsv.
- Existing source directionality is preserved.
- Reverse edges are not generated.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Final

import pandas as pd


NODE_TYPE_ORDER: Final[dict[str, int]] = {
    "drug": 0,
    "gene": 1,
    "phenotype": 2,
    "variant": 3,
}

EXPECTED_RAW_RELATIONSHIPS: Final[int] = 61_770
EXPECTED_HAPLOTYPE_ROWS: Final[int] = 14_698
EXPECTED_IN_SCOPE_ROWS: Final[int] = 47_072
EXPECTED_NODES: Final[int] = 39_076
EXPECTED_EDGES: Final[int] = 47_072
EXPECTED_RELATION_TYPES: Final[int] = 56

BASE_NODE_FILES: Final[dict[str, tuple[str, str, str]]] = {
    "drug": ("chemicals.tsv", "PharmGKB Accession Id", "Name"),
    "gene": ("genes.tsv", "PharmGKB Accession Id", "Symbol"),
    "phenotype": ("phenotypes.tsv", "PharmGKB Accession Id", "Name"),
    "variant": ("variants.tsv", "Variant ID", "Variant Name"),
}


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column whitespace and remove byte-order marks."""
    result = df.copy()
    result.columns = [
        re.sub(r"\s+", " ", str(column).replace("\ufeff", "").strip())
        for column in result.columns
    ]
    return result


def read_tsv(path: Path) -> pd.DataFrame:
    """Read a TSV using strings to preserve stable identifier formatting."""
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")

    return clean_columns(
        pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=True,
            low_memory=False,
        )
    )


def clean_text(value: object) -> str | None:
    """Return cleaned non-empty text or None."""
    if pd.isna(value):
        return None

    text = re.sub(r"\s+", " ", str(value).strip())
    if text.lower() in {"", "nan", "none", "null"}:
        return None

    return text


def canonical_type(value: object) -> str | None:
    """Map raw ClinPGx entity types to the predefined graph schema."""
    text = clean_text(value)
    if text is None:
        return None

    raw = text.lower()

    if raw in {"chemical", "chemicals", "drug", "compound"}:
        return "drug"
    if raw in {"gene", "genes"}:
        return "gene"
    if raw in {
        "phenotype",
        "phenotypes",
        "disease",
        "diseases",
        "clinical phenotype",
    }:
        return "phenotype"
    if raw in {"variant", "variants"}:
        return "variant"
    if raw in {"haplotype", "haplotypes"}:
        return None

    raise ValueError(f"Unexpected relationship entity type: {value!r}")


def normalise_evidence(value: object) -> str:
    """Map raw evidence combinations to one evidence category.

    Precedence preserves the legacy experiment definition.
    """
    text = str(value).lower()

    if "guideline" in text or "cpic" in text:
        return "guideline"
    if "label" in text or "fda" in text:
        return "label"
    if "pathway" in text or "reactome" in text:
        return "pathway"
    if "clinical" in text or "annotation" in text:
        return "clinical"
    if "literature" in text or "pmid" in text:
        return "literature"

    return "other"


def normalise_association(value: object) -> str:
    """Canonicalise association labels for relation names."""
    text = clean_text(value)
    if text is None:
        raise ValueError("Blank association label encountered.")

    normalised = re.sub(r"[\s-]+", "_", text.lower())

    allowed = {"associated", "ambiguous", "not_associated"}
    if normalised not in allowed:
        raise ValueError(
            f"Unexpected association label {value!r}; "
            f"normalised as {normalised!r}."
        )

    return normalised


def build_base_nodes(data_dir: Path) -> pd.DataFrame:
    """Construct nodes from the four primary entity tables."""
    subsets: list[pd.DataFrame] = []

    for node_type, (filename, id_column, name_column) in BASE_NODE_FILES.items():
        table = read_tsv(data_dir / filename)

        missing_columns = {
            id_column,
            name_column,
        }.difference(table.columns)
        if missing_columns:
            raise ValueError(
                f"{filename} is missing columns: {sorted(missing_columns)}"
            )

        subset = pd.DataFrame(
            {
                "node_id": table[id_column].map(clean_text),
                "node_name": table[name_column].map(clean_text),
                "node_type": node_type,
                "node_source": filename,
                "is_supplemental": False,
            }
        )

        if subset["node_id"].isna().any():
            raise ValueError(f"{filename} contains blank entity IDs.")
        if subset["node_name"].isna().any():
            raise ValueError(f"{filename} contains blank entity names.")

        duplicate_keys = subset.duplicated(
            subset=["node_type", "node_id"],
            keep=False,
        )
        if duplicate_keys.any():
            examples = subset.loc[
                duplicate_keys,
                ["node_type", "node_id", "node_name"],
            ].head(20)
            raise ValueError(
                f"Duplicate stable IDs found in {filename}:\n"
                f"{examples.to_string(index=False)}"
            )

        subsets.append(subset)

    return pd.concat(subsets, ignore_index=True)


def prepare_relationships(data_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Load, filter and canonicalise relationships."""
    relationships = read_tsv(data_dir / "relationships.tsv")

    required = {
        "Entity1_id",
        "Entity1_name",
        "Entity1_type",
        "Entity2_id",
        "Entity2_name",
        "Entity2_type",
        "Evidence",
        "Association",
    }
    missing = required.difference(relationships.columns)
    if missing:
        raise ValueError(
            f"relationships.tsv is missing columns: {sorted(missing)}"
        )

    if len(relationships) != EXPECTED_RAW_RELATIONSHIPS:
        raise AssertionError(
            f"Expected {EXPECTED_RAW_RELATIONSHIPS:,} raw relationships, "
            f"found {len(relationships):,}."
        )

    rel = relationships[
        [
            "Entity1_id",
            "Entity1_name",
            "Entity1_type",
            "Entity2_id",
            "Entity2_name",
            "Entity2_type",
            "Evidence",
            "Association",
            "PK",
            "PD",
            "PMIDs",
        ]
    ].copy()

    rel.columns = [
        "source_id",
        "source_name",
        "source_type_raw",
        "target_id",
        "target_name",
        "target_type_raw",
        "evidence_raw",
        "association_raw",
        "pk",
        "pd",
        "pmids",
    ]

    for column in ["source_id", "source_name", "target_id", "target_name"]:
        rel[column] = rel[column].map(clean_text)

    blank_endpoint_mask = rel[
        ["source_id", "source_name", "target_id", "target_name"]
    ].isna().any(axis=1)

    if blank_endpoint_mask.any():
        raise ValueError(
            f"Found {int(blank_endpoint_mask.sum()):,} relationships with "
            "blank endpoint IDs or names."
        )

    source_haplotype = (
        rel["source_type_raw"].astype(str).str.strip().str.lower()
        == "haplotype"
    )
    target_haplotype = (
        rel["target_type_raw"].astype(str).str.strip().str.lower()
        == "haplotype"
    )
    haplotype_mask = source_haplotype | target_haplotype

    excluded_haplotype_rows = int(haplotype_mask.sum())
    if excluded_haplotype_rows != EXPECTED_HAPLOTYPE_ROWS:
        raise AssertionError(
            f"Expected {EXPECTED_HAPLOTYPE_ROWS:,} haplotype-related rows, "
            f"found {excluded_haplotype_rows:,}."
        )

    rel = rel.loc[~haplotype_mask].copy()

    rel["source_type"] = rel["source_type_raw"].map(canonical_type)
    rel["target_type"] = rel["target_type_raw"].map(canonical_type)

    if rel[["source_type", "target_type"]].isna().any().any():
        raise ValueError(
            "Unexpected null canonical type after haplotype exclusion."
        )

    rel["evidence"] = rel["evidence_raw"].map(normalise_evidence)
    rel["association"] = rel["association_raw"].map(
        normalise_association
    )

    if len(rel) != EXPECTED_IN_SCOPE_ROWS:
        raise AssertionError(
            f"Expected {EXPECTED_IN_SCOPE_ROWS:,} in-scope rows, "
            f"found {len(rel):,}."
        )

    statistics = {
        "raw_relationship_rows": int(len(relationships)),
        "excluded_haplotype_rows": excluded_haplotype_rows,
        "in_scope_relationship_rows": int(len(rel)),
    }

    return rel.reset_index(drop=True), statistics


def supplement_missing_nodes(
    base_nodes: pd.DataFrame,
    relationships: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add valid relationship endpoints absent from primary entity tables."""
    source_endpoints = relationships[
        ["source_id", "source_name", "source_type"]
    ].copy()
    source_endpoints.columns = ["node_id", "endpoint_name", "node_type"]

    target_endpoints = relationships[
        ["target_id", "target_name", "target_type"]
    ].copy()
    target_endpoints.columns = ["node_id", "endpoint_name", "node_type"]

    endpoints = pd.concat(
        [source_endpoints, target_endpoints],
        ignore_index=True,
    )

    endpoint_aliases = (
        endpoints.groupby(["node_type", "node_id"], as_index=False)
        .agg(
            endpoint_names=(
                "endpoint_name",
                lambda values: sorted(set(values)),
            )
        )
    )

    endpoint_aliases["endpoint_name_count"] = endpoint_aliases[
        "endpoint_names"
    ].map(len)

    existing_keys = set(
        base_nodes[["node_type", "node_id"]]
        .itertuples(index=False, name=None)
    )

    missing = endpoint_aliases[
        ~endpoint_aliases.apply(
            lambda row: (row["node_type"], row["node_id"])
            in existing_keys,
            axis=1,
        )
    ].copy()

    missing["node_name"] = missing["endpoint_names"].map(
        lambda names: names[0]
    )
    missing["node_source"] = "relationships.tsv"
    missing["is_supplemental"] = True

    supplemental = missing[
        [
            "node_id",
            "node_name",
            "node_type",
            "node_source",
            "is_supplemental",
        ]
    ]

    nodes = pd.concat(
        [base_nodes, supplemental],
        ignore_index=True,
    )

    duplicate_keys = nodes.duplicated(
        subset=["node_type", "node_id"],
        keep=False,
    )
    if duplicate_keys.any():
        raise AssertionError(
            "Duplicate node-type and stable-ID keys remain after "
            "supplementation."
        )

    return nodes, endpoint_aliases


def assign_node_indices(nodes: pd.DataFrame) -> pd.DataFrame:
    """Assign deterministic global node indices."""
    result = nodes.copy()
    result["_type_order"] = result["node_type"].map(NODE_TYPE_ORDER)

    if result["_type_order"].isna().any():
        unexpected = sorted(
            result.loc[result["_type_order"].isna(), "node_type"].unique()
        )
        raise ValueError(f"Unexpected node types: {unexpected}")

    result = result.sort_values(
        ["_type_order", "node_id"],
        kind="stable",
    ).reset_index(drop=True)

    result["node_index"] = result.index.astype(int)

    return result[
        [
            "node_index",
            "node_id",
            "node_name",
            "node_type",
            "node_source",
            "is_supplemental",
        ]
    ]


def build_edges(
    nodes: pd.DataFrame,
    relationships: pd.DataFrame,
) -> pd.DataFrame:
    """Map in-scope relationship endpoints using stable IDs."""
    node_lookup = nodes.set_index(
        ["node_type", "node_id"]
    )["node_index"].to_dict()

    rel = relationships.copy()

    rel["source_idx"] = [
        node_lookup.get((node_type, node_id))
        for node_type, node_id in zip(
            rel["source_type"],
            rel["source_id"],
        )
    ]
    rel["target_idx"] = [
        node_lookup.get((node_type, node_id))
        for node_type, node_id in zip(
            rel["target_type"],
            rel["target_id"],
        )
    ]

    unmapped = rel[
        rel[["source_idx", "target_idx"]].isna().any(axis=1)
    ]

    if not unmapped.empty:
        raise AssertionError(
            f"{len(unmapped):,} in-scope relationships could not be mapped.\n"
            f"{unmapped.head(20).to_string(index=False)}"
        )

    rel["source_idx"] = rel["source_idx"].astype(int)
    rel["target_idx"] = rel["target_idx"].astype(int)

    rel["relation_type"] = (
        rel["source_type"]
        + "_"
        + rel["target_type"]
        + "_"
        + rel["evidence"]
        + "_"
        + rel["association"]
    )

    edges = rel[
        [
            "source_idx",
            "target_idx",
            "source_id",
            "target_id",
            "source_type",
            "target_type",
            "relation_type",
            "evidence",
            "association",
            "evidence_raw",
            "association_raw",
            "pk",
            "pd",
            "pmids",
        ]
    ].reset_index(drop=True)

    duplicate_mask = edges.duplicated(
        subset=[
            "source_idx",
            "target_idx",
            "relation_type",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        duplicates = edges.loc[
            duplicate_mask,
            [
                "source_idx",
                "target_idx",
                "relation_type",
                "source_id",
                "target_id",
            ],
        ]
        raise AssertionError(
            f"Found {int(duplicate_mask.sum()):,} duplicate directed edge "
            f"records:\n{duplicates.head(20).to_string(index=False)}"
        )

    return edges


def validate_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> None:
    """Enforce the audited graph invariants."""
    if len(nodes) != EXPECTED_NODES:
        raise AssertionError(
            f"Expected {EXPECTED_NODES:,} nodes, found {len(nodes):,}."
        )

    if len(edges) != EXPECTED_EDGES:
        raise AssertionError(
            f"Expected {EXPECTED_EDGES:,} edges, found {len(edges):,}."
        )

    relation_count = int(edges["relation_type"].nunique())
    if relation_count != EXPECTED_RELATION_TYPES:
        raise AssertionError(
            f"Expected {EXPECTED_RELATION_TYPES} relation types, "
            f"found {relation_count}."
        )

    expected_node_counts = {
        "drug": 5_210,
        "gene": 25_042,
        "phenotype": 1_619,
        "variant": 7_205,
    }
    actual_node_counts = (
        nodes["node_type"].value_counts().sort_index().to_dict()
    )

    if actual_node_counts != expected_node_counts:
        raise AssertionError(
            f"Unexpected node counts.\n"
            f"Expected: {expected_node_counts}\n"
            f"Observed: {actual_node_counts}"
        )

    expected_supplemental = {
        "gene": 1,
        "phenotype": 5,
        "variant": 114,
    }
    actual_supplemental = (
        nodes[nodes["is_supplemental"]]
        ["node_type"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    if actual_supplemental != expected_supplemental:
        raise AssertionError(
            f"Unexpected supplemental-node counts.\n"
            f"Expected: {expected_supplemental}\n"
            f"Observed: {actual_supplemental}"
        )


def build_graph(
    data_dir: Path,
    graph_dir: Path,
    mappings_dir: Path,
) -> dict:
    """Build, validate and save the corrected graph."""
    graph_dir.mkdir(parents=True, exist_ok=True)
    mappings_dir.mkdir(parents=True, exist_ok=True)

    base_nodes = build_base_nodes(data_dir)
    relationships, relationship_stats = prepare_relationships(data_dir)

    nodes, endpoint_aliases = supplement_missing_nodes(
        base_nodes,
        relationships,
    )
    nodes = assign_node_indices(nodes)
    edges = build_edges(nodes, relationships)

    validate_graph(nodes, edges)

    relation_types = sorted(edges["relation_type"].unique())
    relation_to_idx = {
        relation: index
        for index, relation in enumerate(relation_types)
    }
    idx_to_relation = {
        str(index): relation
        for relation, index in relation_to_idx.items()
    }

    edges["rel_idx"] = edges["relation_type"].map(
        relation_to_idx
    ).astype(int)

    nodes_path = graph_dir / "nodes.parquet"
    edges_path = graph_dir / "edges.parquet"
    aliases_path = graph_dir / "endpoint_aliases.parquet"

    nodes.to_parquet(nodes_path, index=False)
    edges.to_parquet(edges_path, index=False)
    endpoint_aliases.to_parquet(aliases_path, index=False)

    node_id_to_idx = {
        f"{row.node_type}:{row.node_id}": int(row.node_index)
        for row in nodes.itertuples(index=False)
    }

    with (mappings_dir / "node_id_to_idx.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(node_id_to_idx, handle, indent=2, sort_keys=True)

    with (mappings_dir / "relation_to_idx.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(relation_to_idx, handle, indent=2, sort_keys=True)

    with (mappings_dir / "idx_to_relation.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(idx_to_relation, handle, indent=2, sort_keys=True)

    manifest = {
        **relationship_stats,
        "base_node_rows": int(len(base_nodes)),
        "supplemental_node_rows": int(nodes["is_supplemental"].sum()),
        "total_nodes": int(len(nodes)),
        "total_edges": int(len(edges)),
        "relation_type_count": int(len(relation_types)),
        "node_type_counts": {
            key: int(value)
            for key, value in nodes["node_type"].value_counts().items()
        },
        "supplemental_node_type_counts": {
            key: int(value)
            for key, value in (
                nodes[nodes["is_supplemental"]]
                ["node_type"]
                .value_counts()
                .items()
            )
        },
        "endpoint_ids_with_multiple_names": int(
            (endpoint_aliases["endpoint_name_count"] > 1).sum()
        ),
        "maximum_endpoint_names_per_id": int(
            endpoint_aliases["endpoint_name_count"].max()
        ),
        "artifacts": {
            "nodes": str(nodes_path),
            "edges": str(edges_path),
            "endpoint_aliases": str(aliases_path),
        },
    }

    with (graph_dir / "graph_manifest.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the audited four-node PGx graph."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
    )
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=Path("artifacts/graph"),
    )
    parser.add_argument(
        "--mappings-dir",
        type=Path,
        default=Path("artifacts/mappings"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifest = build_graph(
        data_dir=args.data_dir,
        graph_dir=args.graph_dir,
        mappings_dir=args.mappings_dir,
    )

    print(json.dumps(manifest, indent=2, sort_keys=True))
    print("\nGraph construction passed all audited invariants.")


if __name__ == "__main__":
    main()
