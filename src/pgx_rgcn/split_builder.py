"""Construct reproducible drug-disjoint pharmacogenomic graph splits.

Protocols
---------
dg_context
    Remove held-out drug-gene edges from the training graph while retaining
    other contextual edges involving held-out drugs.

strict_cold
    Remove every edge incident to held-out validation and test drugs.

Evaluation
----------
Validation and test positives are canonical associated drug-gene pairs.
Both directed source rows are removed from the training graph together.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


VALID_PROTOCOLS = {"dg_context", "strict_cold"}
VALID_SPLITS = {"train", "validation", "test"}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "seeds",
        "train_fraction",
        "validation_fraction",
        "test_fraction",
        "minimum_associated_pairs",
        "evaluation_association",
        "protocols",
        "split_unit",
    }

    missing = required.difference(config)
    if missing:
        raise ValueError(
            f"Split configuration is missing fields: {sorted(missing)}"
        )

    total_fraction = sum(
        float(config[key])
        for key in [
            "train_fraction",
            "validation_fraction",
            "test_fraction",
        ]
    )

    if not np.isclose(total_fraction, 1.0):
        raise ValueError(
            f"Split fractions must sum to 1.0, found {total_fraction}."
        )

    invalid_protocols = set(config["protocols"]).difference(VALID_PROTOCOLS)
    if invalid_protocols:
        raise ValueError(
            f"Unsupported protocols: {sorted(invalid_protocols)}"
        )

    if config["split_unit"] != "drug":
        raise ValueError("This builder supports drug-level splits only.")

    if int(config["minimum_associated_pairs"]) < 1:
        raise ValueError("minimum_associated_pairs must be at least 1.")


def load_graph(
    graph_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes_path = graph_dir / "nodes.parquet"
    edges_path = graph_dir / "edges.parquet"

    if not nodes_path.exists():
        raise FileNotFoundError(f"Missing graph nodes: {nodes_path}")
    if not edges_path.exists():
        raise FileNotFoundError(f"Missing graph edges: {edges_path}")

    nodes = pd.read_parquet(nodes_path)
    edges = pd.read_parquet(edges_path).reset_index(drop=True)

    required_node_columns = {
        "node_index",
        "node_id",
        "node_name",
        "node_type",
    }
    required_edge_columns = {
        "source_idx",
        "target_idx",
        "source_type",
        "target_type",
        "association",
        "evidence",
        "relation_type",
    }

    missing_nodes = required_node_columns.difference(nodes.columns)
    missing_edges = required_edge_columns.difference(edges.columns)

    if missing_nodes:
        raise ValueError(
            f"nodes.parquet is missing columns: {sorted(missing_nodes)}"
        )
    if missing_edges:
        raise ValueError(
            f"edges.parquet is missing columns: {sorted(missing_edges)}"
        )

    edges.insert(0, "edge_index", edges.index.astype(int))

    return nodes, edges


def identify_drug_gene_edges(edges: pd.DataFrame) -> pd.DataFrame:
    result = edges.copy()

    result["is_drug_gene"] = (
        (
            (result["source_type"] == "drug")
            & (result["target_type"] == "gene")
        )
        |
        (
            (result["source_type"] == "gene")
            & (result["target_type"] == "drug")
        )
    )

    result["drug_idx"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64",
    )
    result["gene_idx"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64",
    )

    drug_gene_mask = result["is_drug_gene"]

    result.loc[drug_gene_mask, "drug_idx"] = (
        result.loc[drug_gene_mask, "source_idx"]
        .where(
            result.loc[drug_gene_mask, "source_type"] == "drug",
            result.loc[drug_gene_mask, "target_idx"],
        )
        .astype("Int64")
    )

    result.loc[drug_gene_mask, "gene_idx"] = (
        result.loc[drug_gene_mask, "target_idx"]
        .where(
            result.loc[drug_gene_mask, "target_type"] == "gene",
            result.loc[drug_gene_mask, "source_idx"],
        )
        .astype("Int64")
    )

    return result


def build_canonical_pair_table(
    edges: pd.DataFrame,
) -> pd.DataFrame:
    dg = edges[edges["is_drug_gene"]].copy()

    pair_table = (
        dg.groupby(["drug_idx", "gene_idx"], as_index=False)
        .agg(
            directed_edge_count=("edge_index", "size"),
            associations=(
                "association",
                lambda values: sorted(set(values)),
            ),
            evidence_categories=(
                "evidence",
                lambda values: sorted(set(values)),
            ),
            directed_edge_indices=(
                "edge_index",
                lambda values: sorted(int(value) for value in values),
            ),
        )
    )

    if not (pair_table["directed_edge_count"] == 2).all():
        invalid = pair_table[
            pair_table["directed_edge_count"] != 2
        ].head(20)

        raise AssertionError(
            "Each canonical drug-gene pair must contain exactly two "
            f"directed rows.\n{invalid.to_string(index=False)}"
        )

    conflicting = pair_table[
        pair_table["associations"].map(len) != 1
    ]

    if not conflicting.empty:
        raise AssertionError(
            "Conflicting association labels were found for canonical "
            f"drug-gene pairs.\n{conflicting.head(20).to_string(index=False)}"
        )

    pair_table["association"] = pair_table["associations"].map(
        lambda values: values[0]
    )

    return pair_table


def calculate_split_sizes(
    eligible_count: int,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[int, int, int]:
    n_train = int(round(eligible_count * train_fraction))
    n_validation = int(round(eligible_count * validation_fraction))
    n_test = eligible_count - n_train - n_validation

    if min(n_train, n_validation, n_test) <= 0:
        raise ValueError(
            "The requested fractions produce an empty split: "
            f"train={n_train}, validation={n_validation}, test={n_test}."
        )

    return n_train, n_validation, n_test


def allocate_stratified_counts(
    stratum_sizes: pd.Series,
    target_count: int,
    fraction: float,
    capacities: pd.Series | None = None,
) -> dict[str, int]:
    """Allocate a global split target proportionally across strata.

    Uses the largest-remainder method while respecting available capacity.
    """
    sizes = stratum_sizes.astype(int).copy()

    if capacities is None:
        capacities = sizes.copy()
    else:
        capacities = capacities.reindex(sizes.index).astype(int)

    expected = sizes.astype(float) * float(fraction)

    allocation = np.floor(expected).astype(int)
    allocation = pd.Series(
        np.minimum(allocation.values, capacities.values),
        index=sizes.index,
        dtype=int,
    )

    remaining = int(target_count - allocation.sum())

    if remaining < 0:
        raise AssertionError(
            "Initial stratified allocation exceeds the requested target."
        )

    remainders = expected - np.floor(expected)

    ranked_strata = sorted(
        sizes.index,
        key=lambda label: (
            -float(remainders.loc[label]),
            str(label),
        ),
    )

    while remaining > 0:
        progress = False

        for label in ranked_strata:
            if allocation.loc[label] >= capacities.loc[label]:
                continue

            allocation.loc[label] += 1
            remaining -= 1
            progress = True

            if remaining == 0:
                break

        if not progress:
            raise ValueError(
                "Insufficient stratum capacity to meet the requested "
                f"target of {target_count}."
            )

    if int(allocation.sum()) != int(target_count):
        raise AssertionError(
            "Stratified allocation does not match the requested target."
        )

    return {
        str(label): int(value)
        for label, value in allocation.items()
    }


def assign_drugs(
    nodes: pd.DataFrame,
    pair_table: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    """Assign eligible drugs using associated-pair-count stratification."""
    evaluation_association = config["evaluation_association"]
    minimum_pairs = int(config["minimum_associated_pairs"])

    positive_pairs = pair_table[
        pair_table["association"] == evaluation_association
    ]

    positive_counts = (
        positive_pairs.groupby("drug_idx")
        .size()
        .rename("associated_pair_count")
    )

    all_pair_counts = (
        pair_table.groupby("drug_idx")
        .size()
        .rename("all_pair_count")
    )

    drug_nodes = nodes[nodes["node_type"] == "drug"][
        [
            "node_index",
            "node_id",
            "node_name",
        ]
    ].copy()

    drug_nodes.columns = [
        "drug_idx",
        "drug_id",
        "drug_name",
    ]

    drug_nodes["associated_pair_count"] = (
        drug_nodes["drug_idx"]
        .map(positive_counts)
        .fillna(0)
        .astype(int)
    )

    drug_nodes["all_pair_count"] = (
        drug_nodes["drug_idx"]
        .map(all_pair_counts)
        .fillna(0)
        .astype(int)
    )

    drug_nodes["eligible"] = (
        drug_nodes["associated_pair_count"] >= minimum_pairs
    )

    stratum_labels = [
        "2",
        "3-4",
        "5-9",
        "10-19",
        "20+",
    ]

    drug_nodes["degree_stratum"] = pd.cut(
        drug_nodes["associated_pair_count"],
        bins=[1, 2, 4, 9, 19, np.inf],
        labels=stratum_labels,
        right=True,
    )

    eligible = drug_nodes[
        drug_nodes["eligible"]
    ].copy()

    if eligible["degree_stratum"].isna().any():
        invalid = eligible[
            eligible["degree_stratum"].isna()
        ]

        raise AssertionError(
            "Eligible drugs were not assigned to a degree stratum:\n"
            f"{invalid.head(20).to_string(index=False)}"
        )

    eligible_count = len(eligible)

    n_train, n_validation, n_test = calculate_split_sizes(
        eligible_count=eligible_count,
        train_fraction=float(config["train_fraction"]),
        validation_fraction=float(config["validation_fraction"]),
    )

    stratum_sizes = (
        eligible.groupby(
            "degree_stratum",
            observed=True,
        )
        .size()
        .reindex(stratum_labels)
        .fillna(0)
        .astype(int)
    )

    validation_allocation = allocate_stratified_counts(
        stratum_sizes=stratum_sizes,
        target_count=n_validation,
        fraction=float(config["validation_fraction"]),
    )

    validation_series = pd.Series(
        validation_allocation,
        dtype=int,
    ).reindex(stratum_labels).fillna(0).astype(int)

    remaining_capacity = (
        stratum_sizes - validation_series
    )

    test_allocation = allocate_stratified_counts(
        stratum_sizes=stratum_sizes,
        target_count=n_test,
        fraction=float(config["test_fraction"]),
        capacities=remaining_capacity,
    )

    rng = np.random.default_rng(seed)

    train_drugs: set[int] = set()
    validation_drugs: set[int] = set()
    test_drugs: set[int] = set()

    for stratum in stratum_labels:
        members = (
            eligible.loc[
                eligible["degree_stratum"].astype(str) == stratum,
                "drug_idx",
            ]
            .astype(int)
            .sort_values()
            .to_numpy()
        )

        shuffled = rng.permutation(members)

        n_stratum_validation = int(
            validation_allocation.get(stratum, 0)
        )
        n_stratum_test = int(
            test_allocation.get(stratum, 0)
        )

        validation_members = shuffled[
            :n_stratum_validation
        ]
        test_members = shuffled[
            n_stratum_validation:
            n_stratum_validation + n_stratum_test
        ]
        train_members = shuffled[
            n_stratum_validation + n_stratum_test:
        ]

        validation_drugs.update(
            int(value) for value in validation_members
        )
        test_drugs.update(
            int(value) for value in test_members
        )
        train_drugs.update(
            int(value) for value in train_members
        )

    if len(train_drugs) != n_train:
        raise AssertionError(
            f"Expected {n_train} train drugs, found {len(train_drugs)}."
        )

    if len(validation_drugs) != n_validation:
        raise AssertionError(
            f"Expected {n_validation} validation drugs, "
            f"found {len(validation_drugs)}."
        )

    if len(test_drugs) != n_test:
        raise AssertionError(
            f"Expected {n_test} test drugs, found {len(test_drugs)}."
        )

    if train_drugs & validation_drugs:
        raise AssertionError("Train and validation drugs overlap.")
    if train_drugs & test_drugs:
        raise AssertionError("Train and test drugs overlap.")
    if validation_drugs & test_drugs:
        raise AssertionError("Validation and test drugs overlap.")

    assignments: dict[int, str] = {}

    assignments.update(
        {drug_idx: "train" for drug_idx in train_drugs}
    )
    assignments.update(
        {
            drug_idx: "validation"
            for drug_idx in validation_drugs
        }
    )
    assignments.update(
        {drug_idx: "test" for drug_idx in test_drugs}
    )

    drug_nodes["split"] = drug_nodes["drug_idx"].map(assignments)
    drug_nodes["split"] = drug_nodes["split"].fillna(
        "training_context_only"
    )

    drug_nodes["seed"] = int(seed)

    observed_eligible = set(
        drug_nodes.loc[
            drug_nodes["eligible"],
            "drug_idx",
        ].astype(int)
    )

    assigned_eligible = (
        train_drugs
        | validation_drugs
        | test_drugs
    )

    if observed_eligible != assigned_eligible:
        raise AssertionError(
            "Not every eligible drug received exactly one split."
        )

    return (
        drug_nodes
        .sort_values("drug_idx")
        .reset_index(drop=True)
    )


def assign_pairs(
    pair_table: pd.DataFrame,
    drug_assignments: pd.DataFrame,
) -> pd.DataFrame:
    split_lookup = drug_assignments.set_index(
        "drug_idx"
    )["split"]

    result = pair_table.copy()
    result["split"] = result["drug_idx"].map(split_lookup)

    if result["split"].isna().any():
        raise AssertionError(
            "One or more canonical drug-gene pairs could not be "
            "assigned to a drug split."
        )

    return result


def assign_edges(
    edges: pd.DataFrame,
    drug_assignments: pd.DataFrame,
    protocol: str,
) -> pd.DataFrame:
    if protocol not in VALID_PROTOCOLS:
        raise ValueError(f"Unsupported protocol: {protocol}")

    result = edges.copy()

    split_lookup = drug_assignments.set_index(
        "drug_idx"
    )["split"]

    result["supervision_split"] = "not_drug_gene"

    dg_mask = result["is_drug_gene"]
    result.loc[dg_mask, "supervision_split"] = (
        result.loc[dg_mask, "drug_idx"]
        .astype(int)
        .map(split_lookup)
    )

    heldout_drugs = set(
        drug_assignments.loc[
            drug_assignments["split"].isin(
                ["validation", "test"]
            ),
            "drug_idx",
        ].astype(int)
    )

    heldout_dg_mask = (
        result["is_drug_gene"]
        & result["drug_idx"].astype("Int64").isin(heldout_drugs)
    )

    source_heldout = result["source_idx"].isin(heldout_drugs)
    target_heldout = result["target_idx"].isin(heldout_drugs)
    incident_to_heldout = source_heldout | target_heldout

    result["in_training_graph"] = True
    result["removal_reason"] = "retained"

    result.loc[
        heldout_dg_mask,
        "in_training_graph",
    ] = False

    result.loc[
        heldout_dg_mask,
        "removal_reason",
    ] = "heldout_drug_gene"

    if protocol == "strict_cold":
        strict_context_mask = (
            incident_to_heldout
            & ~heldout_dg_mask
        )

        result.loc[
            strict_context_mask,
            "in_training_graph",
        ] = False

        result.loc[
            strict_context_mask,
            "removal_reason",
        ] = "heldout_drug_context"

    return result


def build_evaluation_pairs(
    pair_assignments: pd.DataFrame,
    split: str,
    evaluation_association: str,
) -> pd.DataFrame:
    if split not in {"validation", "test"}:
        raise ValueError(
            "Evaluation pairs can only be validation or test."
        )

    return (
        pair_assignments[
            (pair_assignments["split"] == split)
            & (
                pair_assignments["association"]
                == evaluation_association
            )
        ]
        .copy()
        .sort_values(["drug_idx", "gene_idx"])
        .reset_index(drop=True)
    )


def variant_mediated_exposure(
    training_edges: pd.DataFrame,
    evaluation_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Identify held-out pairs connected through drug-variant-gene paths."""
    drug_variant = training_edges[
        (
            (training_edges["source_type"] == "drug")
            & (training_edges["target_type"] == "variant")
        )
        |
        (
            (training_edges["source_type"] == "variant")
            & (training_edges["target_type"] == "drug")
        )
    ].copy()

    variant_gene = training_edges[
        (
            (training_edges["source_type"] == "variant")
            & (training_edges["target_type"] == "gene")
        )
        |
        (
            (training_edges["source_type"] == "gene")
            & (training_edges["target_type"] == "variant")
        )
    ].copy()

    drug_variant["drug_idx"] = drug_variant["source_idx"].where(
        drug_variant["source_type"] == "drug",
        drug_variant["target_idx"],
    ).astype(int)

    drug_variant["variant_idx"] = drug_variant["target_idx"].where(
        drug_variant["target_type"] == "variant",
        drug_variant["source_idx"],
    ).astype(int)

    variant_gene["variant_idx"] = variant_gene["source_idx"].where(
        variant_gene["source_type"] == "variant",
        variant_gene["target_idx"],
    ).astype(int)

    variant_gene["gene_idx"] = variant_gene["target_idx"].where(
        variant_gene["target_type"] == "gene",
        variant_gene["source_idx"],
    ).astype(int)

    drug_variant_pairs = (
        drug_variant[
            ["drug_idx", "variant_idx"]
        ]
        .drop_duplicates()
    )

    variant_gene_pairs = (
        variant_gene[
            ["variant_idx", "gene_idx"]
        ]
        .drop_duplicates()
    )

    mediated = drug_variant_pairs.merge(
        variant_gene_pairs,
        on="variant_idx",
        how="inner",
    )

    mediated_grouped = (
        mediated.groupby(["drug_idx", "gene_idx"])["variant_idx"]
        .agg(lambda values: sorted(set(int(value) for value in values)))
        .rename("mediating_variants")
        .reset_index()
    )

    exposure = evaluation_pairs[
        ["drug_idx", "gene_idx"]
    ].copy()

    exposure = exposure.merge(
        mediated_grouped,
        on=["drug_idx", "gene_idx"],
        how="left",
    )

    exposure["mediating_variants"] = exposure[
        "mediating_variants"
    ].map(
        lambda value: value
        if isinstance(value, list)
        else []
    )

    exposure["variant_mediated_exposure"] = exposure[
        "mediating_variants"
    ].map(bool)

    exposure["mediating_variant_count"] = exposure[
        "mediating_variants"
    ].map(len)

    return exposure


def perform_leakage_audit(
    edge_assignments: pd.DataFrame,
    training_edges: pd.DataFrame,
    validation_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
    protocol: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    heldout_dg_remaining = training_edges[
        training_edges["is_drug_gene"]
        & training_edges["supervision_split"].isin(
            ["validation", "test"]
        )
    ]

    validation_exposure = variant_mediated_exposure(
        training_edges,
        validation_pairs,
    )
    validation_exposure["split"] = "validation"

    test_exposure = variant_mediated_exposure(
        training_edges,
        test_pairs,
    )
    test_exposure["split"] = "test"

    exposure = pd.concat(
        [validation_exposure, test_exposure],
        ignore_index=True,
    )

    heldout_drugs = set(
        edge_assignments.loc[
            edge_assignments["supervision_split"].isin(
                ["validation", "test"]
            ),
            "drug_idx",
        ]
        .dropna()
        .astype(int)
    )

    retained_incident_context = training_edges[
        (
            training_edges["source_idx"].isin(heldout_drugs)
            | training_edges["target_idx"].isin(heldout_drugs)
        )
        & ~training_edges["is_drug_gene"]
    ]

    audit = {
        "protocol": protocol,
        "direct_heldout_drug_gene_edges_in_training_graph": int(
            len(heldout_dg_remaining)
        ),
        "heldout_non_drug_gene_context_edges_retained": int(
            len(retained_incident_context)
        ),
        "validation_pairs": int(len(validation_pairs)),
        "validation_variant_exposed_pairs": int(
            validation_exposure[
                "variant_mediated_exposure"
            ].sum()
        ),
        "validation_variant_exposure_fraction": (
            float(
                validation_exposure[
                    "variant_mediated_exposure"
                ].mean()
            )
            if len(validation_exposure)
            else 0.0
        ),
        "test_pairs": int(len(test_pairs)),
        "test_variant_exposed_pairs": int(
            test_exposure[
                "variant_mediated_exposure"
            ].sum()
        ),
        "test_variant_exposure_fraction": (
            float(
                test_exposure[
                    "variant_mediated_exposure"
                ].mean()
            )
            if len(test_exposure)
            else 0.0
        ),
    }

    if audit[
        "direct_heldout_drug_gene_edges_in_training_graph"
    ] != 0:
        raise AssertionError(
            "Direct held-out drug-gene edges remain in the "
            "training graph."
        )

    if (
        protocol == "strict_cold"
        and audit[
            "heldout_non_drug_gene_context_edges_retained"
        ] != 0
    ):
        raise AssertionError(
            "Strict-cold training graph retains context edges "
            "incident to held-out drugs."
        )

    return audit, exposure


def serialisable_manifest(
    *,
    seed: int,
    protocol: str,
    config: dict[str, Any],
    drug_assignments: pd.DataFrame,
    pair_assignments: pd.DataFrame,
    edge_assignments: pd.DataFrame,
    train_edges: pd.DataFrame,
    validation_edges: pd.DataFrame,
    test_edges: pd.DataFrame,
    validation_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
    audit: dict[str, Any],
) -> dict[str, Any]:
    drug_counts = (
        drug_assignments["split"]
        .value_counts()
        .to_dict()
    )

    eligible_counts = (
        drug_assignments[
            drug_assignments["eligible"]
        ]["split"]
        .value_counts()
        .to_dict()
    )

    return {
        "seed": int(seed),
        "protocol": protocol,
        "minimum_associated_pairs": int(
            config["minimum_associated_pairs"]
        ),
        "evaluation_association": config[
            "evaluation_association"
        ],
        "eligible_drug_count": int(
            drug_assignments["eligible"].sum()
        ),
        "eligible_drug_split_counts": {
            key: int(value)
            for key, value in eligible_counts.items()
        },
        "all_drug_assignment_counts": {
            key: int(value)
            for key, value in drug_counts.items()
        },
        "canonical_drug_gene_pairs": int(
            len(pair_assignments)
        ),
        "training_graph_edges": int(len(train_edges)),
        "validation_directed_positive_edges": int(
            len(validation_edges)
        ),
        "test_directed_positive_edges": int(
            len(test_edges)
        ),
        "validation_positive_pairs": int(
            len(validation_pairs)
        ),
        "test_positive_pairs": int(len(test_pairs)),
        "edge_removal_counts": {
            key: int(value)
            for key, value in (
                edge_assignments[
                    "removal_reason"
                ]
                .value_counts()
                .items()
            )
        },
        "leakage_audit": audit,
    }


def save_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            data,
            handle,
            indent=2,
            sort_keys=True,
        )


def build_one_split(
    *,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    pair_table: pd.DataFrame,
    config: dict[str, Any],
    output_root: Path,
    seed: int,
    protocol: str,
) -> dict[str, Any]:
    output_dir = (
        output_root
        / protocol
        / f"seed_{seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    drug_assignments = assign_drugs(
        nodes=nodes,
        pair_table=pair_table,
        config=config,
        seed=seed,
    )

    pair_assignments = assign_pairs(
        pair_table=pair_table,
        drug_assignments=drug_assignments,
    )

    edge_assignments = assign_edges(
        edges=edges,
        drug_assignments=drug_assignments,
        protocol=protocol,
    )

    train_edges = (
        edge_assignments[
            edge_assignments["in_training_graph"]
        ]
        .copy()
        .reset_index(drop=True)
    )

    evaluation_association = config[
        "evaluation_association"
    ]

    validation_pairs = build_evaluation_pairs(
        pair_assignments,
        split="validation",
        evaluation_association=evaluation_association,
    )

    test_pairs = build_evaluation_pairs(
        pair_assignments,
        split="test",
        evaluation_association=evaluation_association,
    )

    validation_edges = (
        edge_assignments[
            (
                edge_assignments["supervision_split"]
                == "validation"
            )
            & (
                edge_assignments["association"]
                == evaluation_association
            )
            & edge_assignments["is_drug_gene"]
        ]
        .copy()
        .reset_index(drop=True)
    )

    test_edges = (
        edge_assignments[
            (
                edge_assignments["supervision_split"]
                == "test"
            )
            & (
                edge_assignments["association"]
                == evaluation_association
            )
            & edge_assignments["is_drug_gene"]
        ]
        .copy()
        .reset_index(drop=True)
    )

    if len(validation_edges) != 2 * len(validation_pairs):
        raise AssertionError(
            "Validation positive pairs do not have exactly two "
            "directed edge rows."
        )

    if len(test_edges) != 2 * len(test_pairs):
        raise AssertionError(
            "Test positive pairs do not have exactly two "
            "directed edge rows."
        )

    audit, exposure = perform_leakage_audit(
        edge_assignments=edge_assignments,
        training_edges=train_edges,
        validation_pairs=validation_pairs,
        test_pairs=test_pairs,
        protocol=protocol,
    )

    manifest = serialisable_manifest(
        seed=seed,
        protocol=protocol,
        config=config,
        drug_assignments=drug_assignments,
        pair_assignments=pair_assignments,
        edge_assignments=edge_assignments,
        train_edges=train_edges,
        validation_edges=validation_edges,
        test_edges=test_edges,
        validation_pairs=validation_pairs,
        test_pairs=test_pairs,
        audit=audit,
    )

    drug_assignments.to_parquet(
        output_dir / "drug_assignments.parquet",
        index=False,
    )
    pair_assignments.to_parquet(
        output_dir / "pair_assignments.parquet",
        index=False,
    )
    edge_assignments.to_parquet(
        output_dir / "edge_assignments.parquet",
        index=False,
    )
    train_edges.to_parquet(
        output_dir / "train_edges.parquet",
        index=False,
    )
    validation_edges.to_parquet(
        output_dir / "validation_edges.parquet",
        index=False,
    )
    test_edges.to_parquet(
        output_dir / "test_edges.parquet",
        index=False,
    )
    validation_pairs.to_parquet(
        output_dir / "validation_pairs.parquet",
        index=False,
    )
    test_pairs.to_parquet(
        output_dir / "test_pairs.parquet",
        index=False,
    )
    exposure.to_parquet(
        output_dir / "variant_exposure.parquet",
        index=False,
    )

    save_json(
        manifest,
        output_dir / "split_manifest.json",
    )
    save_json(
        audit,
        output_dir / "leakage_audit.json",
    )

    return manifest


def build_all_splits(
    graph_dir: Path,
    config_path: Path,
    output_root: Path,
) -> list[dict[str, Any]]:
    config = load_json(config_path)
    validate_config(config)

    nodes, edges = load_graph(graph_dir)
    edges = identify_drug_gene_edges(edges)
    pair_table = build_canonical_pair_table(edges)

    manifests: list[dict[str, Any]] = []

    for protocol in config["protocols"]:
        for seed in config["seeds"]:
            manifest = build_one_split(
                nodes=nodes,
                edges=edges,
                pair_table=pair_table,
                config=config,
                output_root=output_root,
                seed=int(seed),
                protocol=protocol,
            )

            manifests.append(manifest)

            audit = manifest["leakage_audit"]

            print(
                f"{protocol:12s} seed={seed}: "
                f"train_edges={manifest['training_graph_edges']:,}, "
                f"val_pairs={manifest['validation_positive_pairs']:,}, "
                f"test_pairs={manifest['test_positive_pairs']:,}, "
                f"test_variant_exposure="
                f"{audit['test_variant_exposure_fraction']:.2%}"
            )

    summary = {
        "config": config,
        "runs": manifests,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    save_json(
        summary,
        output_root / "split_summary.json",
    )

    return manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reproducible cold-drug graph splits."
    )
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=Path("artifacts/graph"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cold_drug_splits.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/splits"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifests = build_all_splits(
        graph_dir=args.graph_dir,
        config_path=args.config,
        output_root=args.output_dir,
    )

    print(
        f"\nCompleted {len(manifests)} cold-drug split runs."
    )


if __name__ == "__main__":
    main()
