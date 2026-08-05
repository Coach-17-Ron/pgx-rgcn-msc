#!/usr/bin/env python
"""Build RDKit Morgan fingerprints aligned to graph drug nodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Morgan fingerprints for graph drug nodes "
            "from PharmGKB chemical SMILES."
        )
    )

    parser.add_argument(
        "--chemicals",
        type=Path,
        default=Path("data/chemicals.tsv"),
    )
    parser.add_argument(
        "--nodes",
        type=Path,
        default=Path("artifacts/graph/nodes.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/molecular_features"),
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--n-bits",
        type=int,
        default=2048,
    )

    return parser.parse_args()


def normalise_smiles(value: object) -> str | None:
    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    return text


def main() -> None:
    args = parse_args()

    chemicals = pd.read_csv(
        args.chemicals,
        sep="\t",
        dtype=str,
    )

    nodes = pd.read_parquet(args.nodes)

    required_chemical_columns = {
        "PharmGKB Accession Id",
        "Name",
        "SMILES",
    }
    missing_chemical_columns = (
        required_chemical_columns - set(chemicals.columns)
    )

    if missing_chemical_columns:
        raise ValueError(
            "Missing chemical columns: "
            f"{sorted(missing_chemical_columns)}"
        )

    required_node_columns = {
        "node_index",
        "node_id",
        "node_name",
        "node_type",
    }
    missing_node_columns = (
        required_node_columns - set(nodes.columns)
    )

    if missing_node_columns:
        raise ValueError(
            "Missing node columns: "
            f"{sorted(missing_node_columns)}"
        )

    drugs = (
        nodes.loc[
            nodes["node_type"].eq("drug"),
            [
                "node_index",
                "node_id",
                "node_name",
            ],
        ]
        .sort_values("node_index")
        .reset_index(drop=True)
    )

    chemical_lookup = chemicals[
        [
            "PharmGKB Accession Id",
            "SMILES",
        ]
    ].rename(
        columns={
            "PharmGKB Accession Id": "node_id",
            "SMILES": "smiles",
        }
    )

    if chemical_lookup["node_id"].duplicated().any():
        duplicated_ids = sorted(
            chemical_lookup.loc[
                chemical_lookup["node_id"].duplicated(
                    keep=False
                ),
                "node_id",
            ].dropna().unique()
        )

        raise ValueError(
            "Duplicate chemical accession IDs: "
            f"{duplicated_ids[:10]}"
        )

    merged = drugs.merge(
        chemical_lookup,
        on="node_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = merged.loc[
        merged["_merge"].ne("both"),
        "node_id",
    ].tolist()

    if unmatched:
        raise ValueError(
            "Graph drugs missing from chemicals.tsv: "
            f"{unmatched[:10]}"
        )

    merged = merged.drop(columns="_merge")

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=args.radius,
        fpSize=args.n_bits,
        includeChirality=True,
    )

    fingerprints = np.zeros(
        (len(merged), args.n_bits),
        dtype=np.uint8,
    )

    metadata_rows: list[dict[str, object]] = []

    for fingerprint_row, row in merged.iterrows():
        smiles = normalise_smiles(row["smiles"])

        status = "missing"
        canonical_smiles = ""
        has_fingerprint = False

        if smiles is not None:
            molecule = Chem.MolFromSmiles(smiles)

            if molecule is None:
                status = "invalid"
            else:
                status = "valid"
                canonical_smiles = Chem.MolToSmiles(
                    molecule,
                    canonical=True,
                    isomericSmiles=True,
                )

                bit_vector = generator.GetFingerprint(
                    molecule
                )

                DataStructs.ConvertToNumpyArray(
                    bit_vector,
                    fingerprints[fingerprint_row],
                )

                has_fingerprint = True

        metadata_rows.append(
            {
                "node_index": int(row["node_index"]),
                "node_id": str(row["node_id"]),
                "node_name": str(row["node_name"]),
                "smiles_status": status,
                "canonical_smiles": canonical_smiles,
                "has_fingerprint": has_fingerprint,
                "fingerprint_row": fingerprint_row,
            }
        )

    metadata = pd.DataFrame(metadata_rows)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    matrix_path = (
        args.output_dir
        / f"morgan_radius{args.radius}_{args.n_bits}.npy"
    )

    metadata_path = (
        args.output_dir
        / (
            f"morgan_radius{args.radius}_"
            f"{args.n_bits}_metadata.parquet"
        )
    )

    summary_path = (
        args.output_dir
        / (
            f"morgan_radius{args.radius}_"
            f"{args.n_bits}_summary.json"
        )
    )

    np.save(matrix_path, fingerprints)
    metadata.to_parquet(
        metadata_path,
        index=False,
    )

    valid_mask = metadata[
        "has_fingerprint"
    ].to_numpy()

    active_bits = fingerprints[
        valid_mask
    ].sum(axis=1)

    summary = {
        "radius": args.radius,
        "n_bits": args.n_bits,
        "include_chirality": True,
        "drug_rows": int(len(metadata)),
        "valid_smiles": int(
            metadata["smiles_status"].eq("valid").sum()
        ),
        "missing_smiles": int(
            metadata["smiles_status"].eq("missing").sum()
        ),
        "invalid_smiles": int(
            metadata["smiles_status"].eq("invalid").sum()
        ),
        "unique_canonical_smiles": int(
            metadata.loc[
                metadata["has_fingerprint"],
                "canonical_smiles",
            ].nunique()
        ),
        "mean_active_bits": (
            float(active_bits.mean())
            if len(active_bits)
            else None
        ),
        "minimum_active_bits": (
            int(active_bits.min())
            if len(active_bits)
            else None
        ),
        "maximum_active_bits": (
            int(active_bits.max())
            if len(active_bits)
            else None
        ),
        "matrix_shape": list(fingerprints.shape),
        "matrix_dtype": str(fingerprints.dtype),
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    print("Saved:", matrix_path)
    print("Saved:", metadata_path)
    print("Saved:", summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
