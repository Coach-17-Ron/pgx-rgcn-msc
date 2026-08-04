"""Tests for cold-drug split construction."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


SPLIT_ROOT = Path("artifacts/splits")
SEEDS = [42, 123, 2026]
PROTOCOLS = ["dg_context", "strict_cold"]

EXPECTED_DRUG_COUNTS = {
    "train": 555,
    "validation": 69,
    "test": 70,
}

EXPECTED_STRATA = {
    "2": {"train": 166, "validation": 21, "test": 21},
    "3-4": {"train": 159, "validation": 20, "test": 20},
    "5-9": {"train": 125, "validation": 15, "test": 15},
    "10-19": {"train": 61, "validation": 8, "test": 8},
    "20+": {"train": 44, "validation": 5, "test": 6},
}


def load_run(protocol: str, seed: int):
    run_dir = SPLIT_ROOT / protocol / f"seed_{seed}"

    assignments = pd.read_parquet(
        run_dir / "drug_assignments.parquet"
    )
    edge_assignments = pd.read_parquet(
        run_dir / "edge_assignments.parquet"
    )
    train_edges = pd.read_parquet(
        run_dir / "train_edges.parquet"
    )
    validation_pairs = pd.read_parquet(
        run_dir / "validation_pairs.parquet"
    )
    test_pairs = pd.read_parquet(
        run_dir / "test_pairs.parquet"
    )

    manifest = json.loads(
        (run_dir / "split_manifest.json").read_text()
    )

    return {
        "assignments": assignments,
        "edge_assignments": edge_assignments,
        "train_edges": train_edges,
        "validation_pairs": validation_pairs,
        "test_pairs": test_pairs,
        "manifest": manifest,
    }


def test_all_split_runs_exist():
    for protocol in PROTOCOLS:
        for seed in SEEDS:
            run_dir = SPLIT_ROOT / protocol / f"seed_{seed}"
            assert run_dir.exists()

            for filename in [
                "drug_assignments.parquet",
                "pair_assignments.parquet",
                "edge_assignments.parquet",
                "train_edges.parquet",
                "validation_edges.parquet",
                "test_edges.parquet",
                "validation_pairs.parquet",
                "test_pairs.parquet",
                "variant_exposure.parquet",
                "split_manifest.json",
                "leakage_audit.json",
            ]:
                assert (run_dir / filename).exists()


def test_drug_split_counts_and_disjointness():
    for protocol in PROTOCOLS:
        for seed in SEEDS:
            run = load_run(protocol, seed)
            eligible = run["assignments"][
                run["assignments"]["eligible"]
            ]

            counts = eligible["split"].value_counts().to_dict()

            for split, expected in EXPECTED_DRUG_COUNTS.items():
                assert counts.get(split, 0) == expected

            split_sets = {
                split: set(
                    eligible.loc[
                        eligible["split"] == split,
                        "drug_idx",
                    ].astype(int)
                )
                for split in EXPECTED_DRUG_COUNTS
            }

            assert not (
                split_sets["train"]
                & split_sets["validation"]
            )
            assert not (
                split_sets["train"]
                & split_sets["test"]
            )
            assert not (
                split_sets["validation"]
                & split_sets["test"]
            )

            assert len(
                split_sets["train"]
                | split_sets["validation"]
                | split_sets["test"]
            ) == 694


def test_degree_stratification_counts():
    for protocol in PROTOCOLS:
        for seed in SEEDS:
            run = load_run(protocol, seed)
            eligible = run["assignments"][
                run["assignments"]["eligible"]
            ]

            table = pd.crosstab(
                eligible["degree_stratum"],
                eligible["split"],
            )

            for stratum, split_counts in EXPECTED_STRATA.items():
                for split, expected in split_counts.items():
                    assert int(table.loc[stratum, split]) == expected


def test_no_direct_heldout_drug_gene_edges_in_training():
    for protocol in PROTOCOLS:
        for seed in SEEDS:
            run = load_run(protocol, seed)
            train_edges = run["train_edges"]

            leaked = train_edges[
                train_edges["is_drug_gene"]
                & train_edges["supervision_split"].isin(
                    ["validation", "test"]
                )
            ]

            assert len(leaked) == 0


def test_reverse_orientation_consistency():
    for protocol in PROTOCOLS:
        for seed in SEEDS:
            run = load_run(protocol, seed)
            edge_assignments = run["edge_assignments"]

            dg = edge_assignments[
                edge_assignments["is_drug_gene"]
            ]

            grouped = (
                dg.groupby(
                    [
                        "drug_idx",
                        "gene_idx",
                        "association",
                        "evidence",
                    ]
                )
                .agg(
                    directed_rows=("edge_index", "size"),
                    split_count=("supervision_split", "nunique"),
                    training_flag_count=(
                        "in_training_graph",
                        "nunique",
                    ),
                )
            )

            assert (grouped["directed_rows"] == 2).all()
            assert (grouped["split_count"] == 1).all()
            assert (grouped["training_flag_count"] == 1).all()


def test_evaluation_pairs_are_associated():
    for protocol in PROTOCOLS:
        for seed in SEEDS:
            run = load_run(protocol, seed)

            assert (
                run["validation_pairs"]["association"]
                == "associated"
            ).all()

            assert (
                run["test_pairs"]["association"]
                == "associated"
            ).all()


def test_strict_cold_removes_all_heldout_drug_context():
    for seed in SEEDS:
        run = load_run("strict_cold", seed)

        assignments = run["assignments"]
        train_edges = run["train_edges"]

        heldout = set(
            assignments.loc[
                assignments["split"].isin(
                    ["validation", "test"]
                ),
                "drug_idx",
            ].astype(int)
        )

        incident = train_edges[
            train_edges["source_idx"].isin(heldout)
            | train_edges["target_idx"].isin(heldout)
        ]

        assert len(incident) == 0


def test_dg_context_retains_non_target_context():
    for seed in SEEDS:
        run = load_run("dg_context", seed)
        manifest = run["manifest"]

        retained = manifest["leakage_audit"][
            "heldout_non_drug_gene_context_edges_retained"
        ]

        assert retained > 0


def test_variant_exposure_protocol_behaviour():
    for seed in SEEDS:
        strict_run = load_run("strict_cold", seed)
        context_run = load_run("dg_context", seed)

        strict_audit = strict_run["manifest"]["leakage_audit"]
        context_audit = context_run["manifest"]["leakage_audit"]

        assert (
            strict_audit["test_variant_exposed_pairs"] == 0
        )

        assert (
            context_audit["test_variant_exposed_pairs"] > 0
        )
