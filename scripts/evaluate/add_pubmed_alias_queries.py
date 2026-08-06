#!/usr/bin/env python
"""Add alias-aware PubMed queries to the external validation manifest."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(
    "results/consensus_inference/molecular/"
    "strict_cold/summary"
)

MANIFEST_PATH = ROOT / "external_validation_manifest.csv"


GENE_ALIASES: dict[str, list[str]] = {
    "NR1I2": [
        "PXR",
        "pregnane X receptor",
        "steroid and xenobiotic receptor",
    ],
    "GSTM1": [
        "glutathione S-transferase mu 1",
        "GST mu 1",
    ],
    "OPRM1": [
        "mu opioid receptor",
        "mu-opioid receptor",
        "opioid receptor mu 1",
    ],
    "SLC22A6": [
        "OAT1",
        "organic anion transporter 1",
    ],
    "KCNH2": [
        "hERG",
        "human ether-a-go-go-related gene",
        "potassium voltage-gated channel subfamily H member 2",
    ],
    "COMT": [
        "catechol-O-methyltransferase",
        "catechol O-methyltransferase",
    ],
    "DRD2": [
        "dopamine D2 receptor",
        "D2 dopamine receptor",
    ],
    "FOLR1": [
        "folate receptor alpha",
        "folate receptor 1",
    ],
    "ABCC4": [
        "MRP4",
        "multidrug resistance-associated protein 4",
    ],
    "ABCC2": [
        "MRP2",
        "multidrug resistance-associated protein 2",
    ],
    "MT-RNR1": [
        "mitochondrial 12S rRNA",
        "12S ribosomal RNA",
    ],
    "GSTP1": [
        "glutathione S-transferase pi 1",
        "GST pi 1",
    ],
    "UGT2B7": [
        "UDP-glucuronosyltransferase 2B7",
        "UDP glucuronosyltransferase 2B7",
    ],
    "IFNL3": [
        "IL28B",
        "interferon lambda 3",
        "interleukin 28B",
    ],
    "CFTR": [
        "cystic fibrosis transmembrane conductance regulator",
    ],
}


DRUG_ALIASES: dict[str, list[str]] = {
    "escitalopram": [
        "S-citalopram",
    ],
    "acenocoumarol": [
        "nicoumalone",
    ],
    "chlorthalidone": [
        "chlortalidone",
    ],
    "ethinyl estradiol": [
        "ethinylestradiol",
    ],
}


def field_term(term: str) -> str:
    escaped = str(term).replace('"', "")
    return f'"{escaped}"[Title/Abstract]'


def grouped_terms(
    primary: str,
    aliases: list[str],
) -> str:
    terms: list[str] = []
    seen: set[str] = set()

    for term in [primary, *aliases]:
        normalised = str(term).strip().lower()

        if not normalised or normalised in seen:
            continue

        seen.add(normalised)
        terms.append(field_term(term))

    return "(" + " OR ".join(terms) + ")"


def build_queries(
    drug_name: str,
    gene_name: str,
) -> tuple[str, str, str]:
    drug_group = grouped_terms(
        drug_name,
        DRUG_ALIASES.get(
            drug_name.lower(),
            [],
        ),
    )

    gene_group = grouped_terms(
        gene_name,
        GENE_ALIASES.get(
            gene_name.upper(),
            [],
        ),
    )

    direct = f"{drug_group} AND {gene_group}"

    pgx_context = (
        "("
        '"pharmacogenomics"[Title/Abstract] OR '
        '"pharmacogenetics"[Title/Abstract] OR '
        '"genetic variant"[Title/Abstract] OR '
        '"polymorphism"[Title/Abstract] OR '
        '"drug response"[Title/Abstract] OR '
        '"treatment response"[Title/Abstract] OR '
        '"toxicity"[Title/Abstract]'
        ")"
    )

    mechanism_context = (
        "("
        '"metabolism"[Title/Abstract] OR '
        '"metabolized"[Title/Abstract] OR '
        '"transport"[Title/Abstract] OR '
        '"transporter"[Title/Abstract] OR '
        '"substrate"[Title/Abstract] OR '
        '"inhibition"[Title/Abstract] OR '
        '"inhibitor"[Title/Abstract] OR '
        '"induction"[Title/Abstract] OR '
        '"expression"[Title/Abstract] OR '
        '"binding"[Title/Abstract] OR '
        '"ion current"[Title/Abstract]'
        ")"
    )

    pgx = f"{direct} AND {pgx_context}"
    mechanism = f"{direct} AND {mechanism_context}"

    return direct, pgx, mechanism


def main() -> None:
    manifest = pd.read_csv(
        MANIFEST_PATH,
        low_memory=False,
    )

    for index, row in manifest.iterrows():
        direct, pgx, mechanism = build_queries(
            drug_name=str(row["drug_name"]),
            gene_name=str(row["gene_name"]),
        )

        manifest.loc[
            index,
            "pubmed_direct_query",
        ] = direct

        manifest.loc[
            index,
            "pubmed_pgx_query",
        ] = pgx

        manifest.loc[
            index,
            "pubmed_mechanism_query",
        ] = mechanism

    manifest.to_csv(
        MANIFEST_PATH,
        index=False,
    )

    print("ALIAS-AWARE PUBMED QUERIES COMPLETE")
    print("Pairs:", len(manifest))

    print("\nFIRST TEN DIRECT QUERIES")
    print(
        manifest[
            [
                "external_validation_rank",
                "drug_name",
                "gene_name",
                "pubmed_direct_query",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nSaved:", MANIFEST_PATH)


if __name__ == "__main__":
    main()
