#!/usr/bin/env python
"""Retrieve KEGG human pathways for shortlisted candidate genes."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(
    "results/consensus_inference/molecular/"
    "strict_cold/summary"
)

INPUT_PATH = ROOT / "pubmed_pair_validation_summary.csv"
OUTPUT_PATH = ROOT / "kegg_gene_pathway_results.csv"

BASE_URL = "https://rest.kegg.jp"


def get_text(
    session: requests.Session,
    endpoint: str,
    attempts: int = 4,
) -> str:
    url = f"{BASE_URL}/{endpoint}"

    for attempt in range(1, attempts + 1):
        try:
            response = session.get(url, timeout=60)
            response.raise_for_status()

            # Stay below KEGG's maximum request rate.
            time.sleep(0.4)
            return response.text

        except requests.RequestException:
            if attempt == attempts:
                raise

            time.sleep(2 ** attempt)

    return ""


def parse_find_genes(text: str) -> list[tuple[str, str]]:
    rows = []

    for line in text.splitlines():
        if not line.strip():
            continue

        gene_id, description = line.split("\t", 1)
        rows.append((gene_id, description))

    return rows


def parse_links(text: str) -> list[tuple[str, str]]:
    rows = []

    for line in text.splitlines():
        if not line.strip():
            continue

        source, target = line.split("\t", 1)
        rows.append((source, target))

    return rows


def parse_pathway_names(text: str) -> dict[str, str]:
    names = {}

    for line in text.splitlines():
        if not line.strip():
            continue

        pathway_id, pathway_name = line.split("\t", 1)
        names[pathway_id] = pathway_name

    return names


def main() -> None:
    pairs = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    genes = (
        pairs[["gene_name"]]
        .drop_duplicates()
        .sort_values("gene_name")
    )

    session = requests.Session()
    session.headers.update(
        {"User-Agent": "MLPKGA007-MSc-PathwayValidation/1.0"}
    )

    pathway_names = parse_pathway_names(
        get_text(session, "list/pathway/hsa")
    )

    output_rows = []

    for number, gene_name in enumerate(
        genes["gene_name"],
        start=1,
    ):
        print(
            f"[{number}/{len(genes)}] "
            f"KEGG gene search: {gene_name}"
        )

        gene_matches = parse_find_genes(
            get_text(
                session,
                f"find/hsa/{gene_name}",
            )
        )

        exact_matches = [
            (gene_id, description)
            for gene_id, description in gene_matches
            if gene_name.upper()
            in description.split(";")[0].upper().split(",")
        ]

        if not exact_matches:
            exact_matches = gene_matches[:1]

        if not exact_matches:
            output_rows.append(
                {
                    "gene_name": gene_name,
                    "kegg_gene_id": "",
                    "kegg_gene_description": "",
                    "kegg_pathway_id": "",
                    "kegg_pathway_name": "",
                    "mapping_status": "gene_not_found",
                }
            )
            continue

        for gene_id, description in exact_matches:
            pathway_links = parse_links(
                get_text(
                    session,
                    f"link/pathway/{gene_id}",
                )
            )

            if not pathway_links:
                output_rows.append(
                    {
                        "gene_name": gene_name,
                        "kegg_gene_id": gene_id,
                        "kegg_gene_description": description,
                        "kegg_pathway_id": "",
                        "kegg_pathway_name": "",
                        "mapping_status": "gene_found_no_pathway",
                    }
                )
                continue

            for _, pathway_id in pathway_links:
                output_rows.append(
                    {
                        "gene_name": gene_name,
                        "kegg_gene_id": gene_id,
                        "kegg_gene_description": description,
                        "kegg_pathway_id": pathway_id,
                        "kegg_pathway_name": pathway_names.get(
                            pathway_id,
                            "",
                        ),
                        "mapping_status": "mapped",
                    }
                )

    pathways = pd.DataFrame(output_rows)

    result = pairs.merge(
        pathways,
        on="gene_name",
        how="left",
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nKEGG PATHWAY RETRIEVAL COMPLETE")
    print("Candidate pairs:", pairs.shape[0])
    print("Unique genes:", genes.shape[0])
    print(
        "Genes mapped:",
        pathways.loc[
            pathways["mapping_status"].eq("mapped"),
            "gene_name",
        ].nunique(),
    )
    print(
        "Unique pathways:",
        pathways["kegg_pathway_id"]
        .replace("", pd.NA)
        .nunique(),
    )

    print("\nPATHWAY COUNTS BY GENE")
    print(
        pathways.loc[
            pathways["mapping_status"].eq("mapped")
        ]
        .groupby("gene_name")["kegg_pathway_id"]
        .nunique()
        .sort_values(ascending=False)
        .to_string()
    )

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
