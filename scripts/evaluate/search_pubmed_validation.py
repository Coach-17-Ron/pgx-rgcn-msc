#!/usr/bin/env python
"""Retrieve PubMed evidence for shortlisted drug–gene predictions.

This script performs reproducible PubMed searches through NCBI E-utilities.
It retrieves article metadata and abstracts but does not automatically classify
a drug–gene relationship as supported or contradicted.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ROOT = Path(
    "results/consensus_inference/molecular/"
    "strict_cold/summary"
)

DEFAULT_MANIFEST = ROOT / "external_validation_manifest.csv"
DEFAULT_OUTPUT_DIR = ROOT / "pubmed_validation"

EUTILS_BASE = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
)

TOOL_NAME = "MLPKGA007_MSC_PubMed_Validation"
DEFAULT_EMAIL = os.environ.get(
    "NCBI_EMAIL",
    "replace-with-your-email@example.com",
)

QUERY_TYPES = (
    "direct",
    "pgx",
    "mechanism",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search PubMed for external evidence concerning "
            "shortlisted drug–gene predictions."
        )
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--rank-start",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--rank-end",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--retmax",
        type=int,
        default=25,
        help="Maximum PubMed records per query.",
    )

    parser.add_argument(
        "--query-types",
        nargs="+",
        choices=QUERY_TYPES,
        default=list(QUERY_TYPES),
    )

    parser.add_argument(
        "--email",
        default=DEFAULT_EMAIL,
    )

    parser.add_argument(
        "--api-key",
        default=os.environ.get("NCBI_API_KEY"),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Repeat queries even when cached JSON exists.",
    )

    return parser.parse_args()


def clean_text(value: str | None) -> str:
    if not value:
        return ""

    return re.sub(
        r"\s+",
        " ",
        value,
    ).strip()


def element_text(element: ET.Element | None) -> str:
    if element is None:
        return ""

    return clean_text(
        "".join(element.itertext())
    )


def request_delay(api_key: str | None) -> float:
    # Conservative rates below NCBI's normal limits.
    return 0.11 if api_key else 0.35


def request_with_retry(
    session: requests.Session,
    endpoint: str,
    params: dict[str, Any],
    api_key: str | None,
    attempts: int = 5,
) -> requests.Response:
    url = f"{EUTILS_BASE}/{endpoint}"

    if api_key:
        params["api_key"] = api_key

    delay = request_delay(api_key)

    for attempt in range(1, attempts + 1):
        try:
            response = session.get(
                url,
                params=params,
                timeout=60,
            )

            if response.status_code == 429:
                raise requests.HTTPError(
                    "NCBI rate limit exceeded",
                    response=response,
                )

            response.raise_for_status()
            time.sleep(delay)
            return response

        except (
            requests.RequestException,
            requests.HTTPError,
        ) as error:
            if attempt == attempts:
                raise RuntimeError(
                    f"NCBI request failed after {attempts} "
                    f"attempts: {url}"
                ) from error

            sleep_seconds = min(
                2 ** attempt,
                30,
            )

            print(
                f"Request failed on attempt {attempt}; "
                f"retrying in {sleep_seconds}s..."
            )

            time.sleep(sleep_seconds)

    raise RuntimeError("Unreachable retry state.")


def esearch(
    session: requests.Session,
    query: str,
    retmax: int,
    email: str,
    api_key: str | None,
) -> dict[str, Any]:
    response = request_with_retry(
        session=session,
        endpoint="esearch.fcgi",
        params={
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": retmax,
            "sort": "relevance",
            "tool": TOOL_NAME,
            "email": email,
        },
        api_key=api_key,
    )

    payload = response.json()
    result = payload.get("esearchresult", {})

    return {
        "count": int(result.get("count", 0)),
        "pmids": result.get("idlist", []),
        "query_translation": result.get(
            "querytranslation",
            "",
        ),
    }


def parse_pubmed_article(
    article: ET.Element,
) -> dict[str, Any]:
    medline = article.find("MedlineCitation")
    pubmed_data = article.find("PubmedData")

    if medline is None:
        return {}

    pmid = element_text(
        medline.find("PMID")
    )

    article_node = medline.find(
        "Article"
    )

    if article_node is None:
        return {"pmid": pmid}

    title = element_text(
        article_node.find("ArticleTitle")
    )

    abstract_parts: list[str] = []

    for abstract_node in article_node.findall(
        "./Abstract/AbstractText"
    ):
        label = abstract_node.attrib.get(
            "Label",
            "",
        )

        text = element_text(abstract_node)

        if not text:
            continue

        if label:
            abstract_parts.append(
                f"{label}: {text}"
            )
        else:
            abstract_parts.append(text)

    abstract = " ".join(abstract_parts)

    journal_title = element_text(
        article_node.find(
            "./Journal/Title"
        )
    )

    journal_abbreviation = element_text(
        medline.find(
            "MedlineJournalInfo/"
            "MedlineTA"
        )
    )

    publication_type_values = [
        element_text(node)
        for node in article_node.findall(
            "./PublicationTypeList/"
            "PublicationType"
        )
    ]

    publication_types = ";".join(
        sorted(
            {
                value
                for value in publication_type_values
                if value
            }
        )
    )

    authors: list[str] = []

    for author in article_node.findall(
        "./AuthorList/Author"
    ):
        collective = element_text(
            author.find("CollectiveName")
        )

        if collective:
            authors.append(collective)
            continue

        last_name = element_text(
            author.find("LastName")
        )

        initials = element_text(
            author.find("Initials")
        )

        name = clean_text(
            f"{last_name} {initials}"
        )

        if name:
            authors.append(name)

    mesh_terms: list[str] = []

    for heading in medline.findall(
        "./MeshHeadingList/"
        "MeshHeading"
    ):
        descriptor = element_text(
            heading.find("DescriptorName")
        )

        qualifiers = [
            element_text(node)
            for node in heading.findall(
                "QualifierName"
            )
        ]

        qualifiers = [
            value
            for value in qualifiers
            if value
        ]

        if descriptor and qualifiers:
            mesh_terms.append(
                f"{descriptor}/"
                + ",".join(qualifiers)
            )
        elif descriptor:
            mesh_terms.append(descriptor)

    keywords = [
        element_text(node)
        for node in medline.findall(
            "./KeywordList/Keyword"
        )
    ]

    chemicals = [
        element_text(node)
        for node in medline.findall(
            "./ChemicalList/Chemical/"
            "NameOfSubstance"
        )
    ]

    doi = ""

    if pubmed_data is not None:
        for article_id in pubmed_data.findall(
            "./ArticleIdList/ArticleId"
        ):
            if (
                article_id.attrib.get("IdType")
                == "doi"
            ):
                doi = element_text(article_id)
                break

    year = element_text(
        article_node.find(
            "./Journal/JournalIssue/"
            "PubDate/Year"
        )
    )

    if not year:
        medline_date = element_text(
            article_node.find(
                "./Journal/JournalIssue/"
                "PubDate/MedlineDate"
            )
        )

        match = re.search(
            r"\b(19|20)\d{2}\b",
            medline_date,
        )

        if match:
            year = match.group(0)

    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "publication_year": year,
        "journal": (
            journal_title
            or journal_abbreviation
        ),
        "authors": ";".join(authors),
        "publication_types": publication_types,
        "mesh_terms": ";".join(
            sorted(set(mesh_terms))
        ),
        "keywords": ";".join(
            sorted(
                {
                    value
                    for value in keywords
                    if value
                }
            )
        ),
        "chemicals": ";".join(
            sorted(
                {
                    value
                    for value in chemicals
                    if value
                }
            )
        ),
        "doi": doi,
        "pubmed_url": (
            f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            if pmid
            else ""
        ),
    }


def efetch(
    session: requests.Session,
    pmids: list[str],
    email: str,
    api_key: str | None,
) -> list[dict[str, Any]]:
    if not pmids:
        return []

    response = request_with_retry(
        session=session,
        endpoint="efetch.fcgi",
        params={
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "tool": TOOL_NAME,
            "email": email,
        },
        api_key=api_key,
    )

    root = ET.fromstring(
        response.content
    )

    records: list[dict[str, Any]] = []

    for article in root.findall(
        ".//PubmedArticle"
    ):
        record = parse_pubmed_article(
            article
        )

        if record:
            records.append(record)

    return records


def query_for_type(
    row: pd.Series,
    query_type: str,
) -> str:
    column = {
        "direct": "pubmed_direct_query",
        "pgx": "pubmed_pgx_query",
        "mechanism": "pubmed_mechanism_query",
    }[query_type]

    return str(row[column]).strip()


def cache_path(
    cache_dir: Path,
    rank: int,
    query_type: str,
) -> Path:
    return (
        cache_dir
        / f"rank_{rank:02d}_{query_type}.json"
    )


def keyword_presence(
    text: object,
    term: object,
) -> bool:
    return (
        str(term).lower()
        in str(text).lower()
    )


def main() -> None:
    args = parse_args()

    if (
        args.email
        == "replace-with-your-email@example.com"
    ):
        raise SystemExit(
            "Set your email first, for example:\n"
            "export NCBI_EMAIL='your-email@example.com'"
        )

    if args.rank_start > args.rank_end:
        raise SystemExit(
            "--rank-start must be <= --rank-end"
        )

    manifest = pd.read_csv(
        args.manifest,
        low_memory=False,
    )

    manifest = manifest.loc[
        manifest[
            "external_validation_rank"
        ].between(
            args.rank_start,
            args.rank_end,
        )
    ].sort_values(
        "external_validation_rank"
    )

    if manifest.empty:
        raise SystemExit(
            "No manifest rows matched the requested ranks."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_dir = (
        args.output_dir / "cache"
    )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": (
                f"{TOOL_NAME}/1.0 "
                f"({args.email})"
            )
        }
    )

    query_rows: list[dict[str, Any]] = []
    article_rows: list[dict[str, Any]] = []

    total_queries = (
        len(manifest)
        * len(args.query_types)
    )

    query_number = 0

    for _, pair in manifest.iterrows():
        rank = int(
            pair["external_validation_rank"]
        )

        drug_name = str(pair["drug_name"])
        gene_name = str(pair["gene_name"])

        for query_type in args.query_types:
            query_number += 1

            query = query_for_type(
                pair,
                query_type,
            )

            print(
                f"[{query_number}/{total_queries}] "
                f"rank={rank} "
                f"{drug_name}–{gene_name} "
                f"query={query_type}"
            )

            saved_cache = cache_path(
                cache_dir,
                rank,
                query_type,
            )

            if (
                saved_cache.exists()
                and not args.force
            ):
                payload = json.loads(
                    saved_cache.read_text()
                )
            else:
                search_result = esearch(
                    session=session,
                    query=query,
                    retmax=args.retmax,
                    email=args.email,
                    api_key=args.api_key,
                )

                records = efetch(
                    session=session,
                    pmids=search_result["pmids"],
                    email=args.email,
                    api_key=args.api_key,
                )

                payload = {
                    "rank": rank,
                    "drug_name": drug_name,
                    "gene_name": gene_name,
                    "query_type": query_type,
                    "query": query,
                    "total_pubmed_count": (
                        search_result["count"]
                    ),
                    "query_translation": (
                        search_result[
                            "query_translation"
                        ]
                    ),
                    "retrieved_pmids": (
                        search_result["pmids"]
                    ),
                    "records": records,
                }

                saved_cache.write_text(
                    json.dumps(
                        payload,
                        indent=2,
                        ensure_ascii=False,
                    )
                )

            query_rows.append(
                {
                    "external_validation_rank": rank,
                    "seed": pair["seed"],
                    "drug_id": pair["drug_id"],
                    "drug_name": drug_name,
                    "gene_id": pair["gene_id"],
                    "gene_name": gene_name,
                    "query_type": query_type,
                    "query": query,
                    "total_pubmed_count": payload[
                        "total_pubmed_count"
                    ],
                    "retrieved_records": len(
                        payload["records"]
                    ),
                    "query_translation": payload[
                        "query_translation"
                    ],
                }
            )

            for record_position, record in enumerate(
                payload["records"],
                start=1,
            ):
                searchable_text = " ".join(
                    [
                        record.get("title", ""),
                        record.get("abstract", ""),
                        record.get("mesh_terms", ""),
                        record.get("keywords", ""),
                        record.get("chemicals", ""),
                    ]
                )

                article_rows.append(
                    {
                        "external_validation_rank": rank,
                        "seed": pair["seed"],
                        "drug_id": pair["drug_id"],
                        "drug_name": drug_name,
                        "gene_id": pair["gene_id"],
                        "gene_name": gene_name,
                        "weighted_consensus_rank": pair[
                            "weighted_consensus_rank"
                        ],
                        "models_top_25": pair[
                            "models_top_25"
                        ],
                        "query_type": query_type,
                        "query_result_rank": (
                            record_position
                        ),
                        **record,
                        "drug_term_present": (
                            keyword_presence(
                                searchable_text,
                                drug_name,
                            )
                        ),
                        "gene_term_present": (
                            keyword_presence(
                                searchable_text,
                                gene_name,
                            )
                        ),
                        "both_terms_present": (
                            keyword_presence(
                                searchable_text,
                                drug_name,
                            )
                            and keyword_presence(
                                searchable_text,
                                gene_name,
                            )
                        ),
                        "manual_relevance": "",
                        "manual_evidence_class": "",
                        "manual_support_direction": "",
                        "manual_notes": "",
                    }
                )

    query_summary = pd.DataFrame(
        query_rows
    )

    article_results = pd.DataFrame(
        article_rows
    )

    query_output = (
        args.output_dir
        / (
            f"pubmed_query_summary_"
            f"ranks_{args.rank_start}_"
            f"{args.rank_end}.csv"
        )
    )

    article_output = (
        args.output_dir
        / (
            f"pubmed_article_results_"
            f"ranks_{args.rank_start}_"
            f"{args.rank_end}.csv"
        )
    )

    review_output = (
        args.output_dir
        / (
            f"pubmed_manual_review_"
            f"ranks_{args.rank_start}_"
            f"{args.rank_end}.csv"
        )
    )

    query_summary.to_csv(
        query_output,
        index=False,
    )

    article_results.to_csv(
        article_output,
        index=False,
    )

    review_columns = [
        "external_validation_rank",
        "drug_name",
        "gene_name",
        "query_type",
        "query_result_rank",
        "pmid",
        "publication_year",
        "title",
        "abstract",
        "journal",
        "publication_types",
        "mesh_terms",
        "drug_term_present",
        "gene_term_present",
        "both_terms_present",
        "manual_relevance",
        "manual_evidence_class",
        "manual_support_direction",
        "manual_notes",
        "pubmed_url",
    ]

    if article_results.empty:
        pd.DataFrame(
            columns=review_columns
        ).to_csv(
            review_output,
            index=False,
        )
    else:
        article_results[
            review_columns
        ].drop_duplicates(
            [
                "external_validation_rank",
                "pmid",
            ]
        ).sort_values(
            [
                "external_validation_rank",
                "both_terms_present",
                "query_result_rank",
            ],
            ascending=[True, False, True],
        ).to_csv(
            review_output,
            index=False,
        )

    print("\nPUBMED RETRIEVAL COMPLETE")
    print(
        "Candidate pairs:",
        manifest.shape[0],
    )
    print(
        "Queries executed/cached:",
        len(query_summary),
    )
    print(
        "Article-query records:",
        len(article_results),
    )
    print(
        "Unique PMIDs:",
        (
            article_results["pmid"].nunique()
            if not article_results.empty
            else 0
        ),
    )

    if not query_summary.empty:
        print("\nQUERY RESULT COUNTS")
        print(
            query_summary[
                [
                    "external_validation_rank",
                    "drug_name",
                    "gene_name",
                    "query_type",
                    "total_pubmed_count",
                    "retrieved_records",
                ]
            ].to_string(index=False)
        )

    print("\nSaved:")
    print(query_output)
    print(article_output)
    print(review_output)
    print("Cache:", cache_dir)


if __name__ == "__main__":
    main()
