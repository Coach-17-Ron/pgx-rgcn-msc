#!/usr/bin/env python
"""Retrieve Reactome human pathways for shortlisted candidate genes."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ROOT = Path(
    "results/consensus_inference/molecular/"
    "strict_cold/summary"
)

INPUT_PATH = ROOT / "pubmed_pair_validation_summary.csv"
OUTPUT_PATH = ROOT / "reactome_gene_pathway_results.csv"

BASE_URL = "https://reactome.org/ContentService"


def request_json(
    session: requests.Session,
    url: str,
    attempts: int = 4,
) -> Any:
    for attempt in range(1, attempts + 1):
        try:
            response = session.get(
                url,
                timeout=60,
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
            time.sleep(0.35)
            return response.json()

        except requests.RequestException:
            if attempt == attempts:
                raise

            time.sleep(2 ** attempt)

    return None


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
        {
            "User-Agent": (
                "MLPKGA007-MSc-ReactomeValidation/1.0"
            )
        }
    )

    output_rows = []

    for number, gene_name in enumerate(
        genes["gene_name"],
        start=1,
    ):
        print(
            f"[{number}/{len(genes)}] "
            f"Reactome gene search: {gene_name}"
        )

        mapping_url = (
            f"{BASE_URL}/data/mapping/"
            f"UniProt/{gene_name}/pathways"
        )

        try:
            pathways = request_json(
                session,
                mapping_url,
            )
        except requests.RequestException:
            pathways = None

        if not pathways:
            search_url = (
                f"{BASE_URL}/search/query"
                f"?query={gene_name}"
                f"&species=Homo%20sapiens"
                f"&types=Protein"
                f"&cluster=true"
            )

            try:
                search_payload = request_json(
                    session,
                    search_url,
                )
            except requests.RequestException:
                search_payload = None

            entries = []

            if isinstance(search_payload, dict):
                for group in search_payload.get(
                    "results",
                    [],
                ):
                    entries.extend(
                        group.get("entries", [])
                    )

            protein_hits = [
                entry
                for entry in entries
                if str(
                    entry.get("name", "")
                ).upper() == gene_name.upper()
                or gene_name.upper()
                in str(
                    entry.get("referenceName", "")
                ).upper()
            ]

            if not protein_hits:
                output_rows.append(
                    {
                        "gene_name": gene_name,
                        "reactome_pathway_id": "",
                        "reactome_pathway_name": "",
                        "reactome_species": "",
                        "mapping_status": "gene_not_found",
                    }
                )
                continue

            pathway_ids: set[str] = set()

            for hit in protein_hits:
                stable_id = (
                    hit.get("stId")
                    or hit.get("stableIdentifier")
                    or ""
                )

                if not stable_id:
                    continue

                contained_url = (
                    f"{BASE_URL}/data/pathways/"
                    f"containedEvents/{stable_id}"
                )

                try:
                    contained = request_json(
                        session,
                        contained_url,
                    )
                except requests.RequestException:
                    contained = None

                if isinstance(contained, list):
                    for pathway in contained:
                        pathway_id = pathway.get(
                            "stId",
                            "",
                        )

                        if pathway_id:
                            pathway_ids.add(
                                pathway_id
                            )

            if not pathway_ids:
                output_rows.append(
                    {
                        "gene_name": gene_name,
                        "reactome_pathway_id": "",
                        "reactome_pathway_name": "",
                        "reactome_species": "",
                        "mapping_status": (
                            "gene_found_no_pathway"
                        ),
                    }
                )
                continue

            pathways = []

            for pathway_id in sorted(pathway_ids):
                detail_url = (
                    f"{BASE_URL}/data/query/"
                    f"{pathway_id}"
                )

                try:
                    detail = request_json(
                        session,
                        detail_url,
                    )
                except requests.RequestException:
                    detail = None

                if isinstance(detail, dict):
                    pathways.append(detail)

        if not isinstance(pathways, list):
            output_rows.append(
                {
                    "gene_name": gene_name,
                    "reactome_pathway_id": "",
                    "reactome_pathway_name": "",
                    "reactome_species": "",
                    "mapping_status": "gene_not_found",
                }
            )
            continue

        human_pathways = []

        for pathway in pathways:
            species_name = ""

            species = pathway.get("species")

            if isinstance(species, list):
                species_name = ";".join(
                    sorted(
                        {
                            str(
                                item.get(
                                    "displayName",
                                    "",
                                )
                            )
                            for item in species
                            if isinstance(item, dict)
                        }
                    )
                )
            elif isinstance(species, dict):
                species_name = str(
                    species.get(
                        "displayName",
                        "",
                    )
                )

            if (
                species_name
                and "Homo sapiens"
                not in species_name
            ):
                continue

            human_pathways.append(
                {
                    "gene_name": gene_name,
                    "reactome_pathway_id": pathway.get(
                        "stId",
                        "",
                    ),
                    "reactome_pathway_name": pathway.get(
                        "displayName",
                        pathway.get("name", ""),
                    ),
                    "reactome_species": species_name,
                    "mapping_status": "mapped",
                }
            )

        if human_pathways:
            output_rows.extend(human_pathways)
        else:
            output_rows.append(
                {
                    "gene_name": gene_name,
                    "reactome_pathway_id": "",
                    "reactome_pathway_name": "",
                    "reactome_species": "",
                    "mapping_status": "gene_found_no_pathway",
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

    print("\nREACTOME PATHWAY RETRIEVAL COMPLETE")
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
        pathways["reactome_pathway_id"]
        .replace("", pd.NA)
        .nunique(),
    )

    print("\nPATHWAY COUNTS BY GENE")
    mapped = pathways.loc[
        pathways["mapping_status"].eq("mapped")
    ]

    if mapped.empty:
        print("No Reactome pathways mapped.")
    else:
        print(
            mapped.groupby(
                "gene_name"
            )["reactome_pathway_id"]
            .nunique()
            .sort_values(ascending=False)
            .to_string()
        )

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
