#!/usr/bin/env python
"""Evaluate KEGG pathway overlap between candidate drugs and predicted genes."""

from __future__ import annotations

import re
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests


ROOT = Path(
    "results/consensus_inference/molecular/"
    "strict_cold/summary"
)

PAIR_PATH = ROOT / "pubmed_pair_validation_summary.csv"
GENE_PATHWAY_PATH = ROOT / "kegg_gene_pathway_results.csv"

DRUG_OUTPUT = ROOT / "kegg_drug_pathway_results.csv"
OVERLAP_OUTPUT = ROOT / "kegg_drug_gene_pathway_overlap.csv"
SUMMARY_OUTPUT = ROOT / "kegg_pair_pathway_validation_summary.csv"

BASE_URL = "https://rest.kegg.jp"


DRUG_ALIASES: dict[str, list[str]] = {
    "acenocoumarol": ["nicoumalone"],
    "chlorthalidone": ["chlortalidone"],
    "ethinyl estradiol": ["ethinylestradiol"],
    "escitalopram": ["S-citalopram"],
}


def request_text(
    session: requests.Session,
    endpoint: str,
    attempts: int = 5,
) -> str:
    url = f"{BASE_URL}/{endpoint}"

    for attempt in range(1, attempts + 1):
        try:
            response = session.get(url, timeout=60)

            if response.status_code == 404:
                return ""

            response.raise_for_status()

            # Remain below KEGG's three-request-per-second limit.
            time.sleep(0.4)
            return response.text

        except requests.RequestException:
            if attempt == attempts:
                raise

            time.sleep(min(2 ** attempt, 30))

    return ""


def parse_tab_rows(text: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []

    for line in text.splitlines():
        if not line.strip() or "\t" not in line:
            continue

        left, right = line.split("\t", 1)
        rows.append((left.strip(), right.strip()))

    return rows


def normalise_name(value: str) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value).lower(),
    )


def pathway_number(value: object) -> str:
    if pd.isna(value):
        return ""

    match = re.search(
        r"(\d{5})$",
        str(value),
    )

    return match.group(1) if match else ""


def choose_drug_match(
    drug_name: str,
    matches: list[tuple[str, str]],
) -> tuple[str, str] | None:
    """Select the best KEGG drug-name match.

    KEGG names often include salts, formulations, standards,
    stereochemical labels, or parenthetical annotations.
    """
    if not matches:
        return None

    expected = [
        drug_name,
        *DRUG_ALIASES.get(
            drug_name.lower(),
            [],
        ),
    ]

    expected_normalised = {
        normalise_name(value)
        for value in expected
    }

    scored_matches: list[
        tuple[int, str, str]
    ] = []

    for drug_id, description in matches:
        names_section = description.split(";")[0]

        candidate_names = [
            item.strip()
            for item in re.split(
                r"[,/]",
                names_section,
            )
            if item.strip()
        ]

        candidate_normalised = {
            normalise_name(value)
            for value in candidate_names
        }

        score = 0

        if expected_normalised & candidate_normalised:
            score = 3
        else:
            for expected_name in expected_normalised:
                for candidate_name in candidate_normalised:
                    if not expected_name or not candidate_name:
                        continue

                    if (
                        candidate_name.startswith(expected_name)
                        or expected_name.startswith(candidate_name)
                    ):
                        score = max(score, 2)

                    elif (
                        expected_name in candidate_name
                        or candidate_name in expected_name
                    ):
                        score = max(score, 1)

        if score:
            scored_matches.append(
                (
                    score,
                    drug_id,
                    description,
                )
            )

    if not scored_matches:
        return None

    scored_matches.sort(
        key=lambda row: (
            -row[0],
            len(row[2]),
            row[1],
        )
    )

    _, drug_id, description = scored_matches[0]
    return drug_id, description


def combine_unique(values: pd.Series) -> str:
    return ";".join(
        sorted(
            {
                str(value).strip()
                for value in values.dropna()
                if str(value).strip()
                and str(value).strip().lower() != "nan"
            }
        )
    )


def main() -> None:
    pairs = pd.read_csv(
        PAIR_PATH,
        low_memory=False,
    )

    gene_pathways = pd.read_csv(
        GENE_PATHWAY_PATH,
        low_memory=False,
    )

    gene_pathways = (
        gene_pathways.loc[
            gene_pathways["mapping_status"].eq("mapped"),
            [
                "gene_name",
                "kegg_pathway_id",
                "kegg_pathway_name",
            ],
        ]
        .drop_duplicates()
        .copy()
    )

    gene_pathways["pathway_number"] = (
        gene_pathways["kegg_pathway_id"]
        .map(pathway_number)
    )

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "MLPKGA007-MSc-KEGGDrugPathwayValidation/1.0"
            )
        }
    )

    drug_rows: list[dict[str, object]] = []

    unique_drugs = (
        pairs[["drug_name"]]
        .drop_duplicates()
        .sort_values("drug_name")
    )

    for number, drug_name in enumerate(
        unique_drugs["drug_name"],
        start=1,
    ):
        print(
            f"[{number}/{len(unique_drugs)}] "
            f"KEGG drug search: {drug_name}"
        )

        search_terms = [
            drug_name,
            *DRUG_ALIASES.get(
                str(drug_name).lower(),
                [],
            ),
        ]

        selected: tuple[str, str] | None = None
        matched_query = ""

        for search_term in search_terms:
            matches = parse_tab_rows(
                request_text(
                    session,
                    "find/drug/"
                    + quote(
                        str(search_term),
                        safe="",
                    ),
                )
            )

            selected = choose_drug_match(
                str(drug_name),
                matches,
            )

            if selected is not None:
                matched_query = str(search_term)
                break

        if selected is None:
            drug_rows.append(
                {
                    "drug_name": drug_name,
                    "kegg_drug_id": "",
                    "kegg_drug_description": "",
                    "matched_query": "",
                    "drug_pathway_id": "",
                    "pathway_number": "",
                    "drug_mapping_status": "drug_not_found",
                }
            )
            continue

        drug_id, description = selected

        links = parse_tab_rows(
            request_text(
                session,
                f"link/pathway/{drug_id}",
            )
        )

        if not links:
            drug_rows.append(
                {
                    "drug_name": drug_name,
                    "kegg_drug_id": drug_id,
                    "kegg_drug_description": description,
                    "matched_query": matched_query,
                    "drug_pathway_id": "",
                    "pathway_number": "",
                    "drug_mapping_status": (
                        "drug_found_no_pathway"
                    ),
                }
            )
            continue

        for _, linked_pathway in links:
            drug_rows.append(
                {
                    "drug_name": drug_name,
                    "kegg_drug_id": drug_id,
                    "kegg_drug_description": description,
                    "matched_query": matched_query,
                    "drug_pathway_id": linked_pathway,
                    "pathway_number": pathway_number(
                        linked_pathway
                    ),
                    "drug_mapping_status": "mapped",
                }
            )

    drug_pathways = pd.DataFrame(drug_rows)
    drug_pathways.to_csv(
        DRUG_OUTPUT,
        index=False,
    )

    candidate_drug_paths = pairs[
        [
            "external_validation_rank",
            "drug_name",
            "gene_name",
            "pubmed_pair_evidence_class",
        ]
    ].merge(
        drug_pathways,
        on="drug_name",
        how="left",
    )

    overlap = candidate_drug_paths.merge(
        gene_pathways,
        on=[
            "gene_name",
            "pathway_number",
        ],
        how="inner",
        suffixes=("_drug", "_gene"),
    )

    overlap = overlap.drop_duplicates(
        [
            "external_validation_rank",
            "drug_name",
            "gene_name",
            "pathway_number",
        ]
    )

    overlap.to_csv(
        OVERLAP_OUTPUT,
        index=False,
    )

    drug_summary = (
        drug_pathways.groupby(
            "drug_name",
            as_index=False,
        )
        .agg(
            kegg_drug_id=(
                "kegg_drug_id",
                combine_unique,
            ),
            drug_mapping_status=(
                "drug_mapping_status",
                combine_unique,
            ),
            drug_pathway_count=(
                "pathway_number",
                lambda values: len(
                    {
                        value
                        for value in values
                        if str(value).strip()
                    }
                ),
            ),
        )
    )

    if overlap.empty:
        overlap_summary = pd.DataFrame(
            columns=[
                "external_validation_rank",
                "overlapping_kegg_pathway_count",
                "overlapping_kegg_pathway_ids",
                "overlapping_kegg_pathway_names",
            ]
        )
    else:
        overlap_summary = (
            overlap.groupby(
                "external_validation_rank",
                as_index=False,
            )
            .agg(
                overlapping_kegg_pathway_count=(
                    "pathway_number",
                    "nunique",
                ),
                overlapping_kegg_pathway_ids=(
                    "kegg_pathway_id",
                    combine_unique,
                ),
                overlapping_kegg_pathway_names=(
                    "kegg_pathway_name",
                    combine_unique,
                ),
            )
        )

    summary = (
        pairs.merge(
            drug_summary,
            on="drug_name",
            how="left",
        )
        .merge(
            overlap_summary,
            on="external_validation_rank",
            how="left",
        )
    )

    summary[
        "overlapping_kegg_pathway_count"
    ] = (
        summary[
            "overlapping_kegg_pathway_count"
        ]
        .fillna(0)
        .astype(int)
    )

    summary[
        "has_kegg_drug_gene_pathway_overlap"
    ] = (
        summary[
            "overlapping_kegg_pathway_count"
        ] > 0
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print("\nKEGG DRUG–GENE PATHWAY OVERLAP COMPLETE")
    print("Candidate pairs:", len(summary))
    print(
        "Unique drugs mapped:",
        drug_pathways.loc[
            drug_pathways[
                "drug_mapping_status"
            ].ne("drug_not_found"),
            "drug_name",
        ].nunique(),
    )
    print(
        "Drugs with KEGG pathways:",
        drug_pathways.loc[
            drug_pathways[
                "drug_mapping_status"
            ].eq("mapped"),
            "drug_name",
        ].nunique(),
    )
    print(
        "Pairs with pathway overlap:",
        int(
            summary[
                "has_kegg_drug_gene_pathway_overlap"
            ].sum()
        ),
    )

    print("\nPAIR OVERLAP SUMMARY")
    print(
        summary[
            [
                "external_validation_rank",
                "drug_name",
                "gene_name",
                "drug_mapping_status",
                "drug_pathway_count",
                "overlapping_kegg_pathway_count",
                "overlapping_kegg_pathway_names",
                "pubmed_pair_evidence_class",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(DRUG_OUTPUT)
    print(OVERLAP_OUTPUT)
    print(SUMMARY_OUTPUT)


if __name__ == "__main__":
    main()
