from pathlib import Path
import re
import numpy as np
import pandas as pd

DATA = Path("data")


def clean_name(x):
    if pd.isna(x):
        return np.nan
    s = re.sub(r"\s+", " ", str(x).strip())
    if not s or s.lower() in {"nan", "none", "null"}:
        return np.nan
    return s


def canonical_type(raw_type):
    mapping = {
        "Chemical": "drug",
        "Gene": "gene",
        "Disease": "phenotype",
        "Variant": "variant",
    }
    return mapping.get(str(raw_type).strip(), str(raw_type).strip().lower())


def subtype(raw_type):
    mapping = {
        "Chemical": "chemical",
        "Gene": "gene",
        "Disease": "disease",
        "Variant": "sequence_variant",
    }
    return mapping.get(str(raw_type).strip(), str(raw_type).strip().lower())


def canonical_assoc(x):
    s = str(x).strip().lower()
    return s.replace("-", "_").replace(" ", "_")


def normalise_evidence(x):
    s = str(x).lower()

    if "guideline" in s or "cpic" in s:
        return "guideline"
    if "label" in s or "fda" in s:
        return "label"
    if "pathway" in s or "reactome" in s:
        return "pathway"
    if "clinical" in s or "annotation" in s:
        return "clinical"
    if "literature" in s or "pmid" in s:
        return "literature"

    return "other"


relationships_all = pd.read_csv(
    DATA / "relationships.tsv",
    sep="\t",
    dtype=str,
).fillna("")

haplotype_mask = (
    (relationships_all["Entity1_type"] == "Haplotype") |
    (relationships_all["Entity2_type"] == "Haplotype")
)

excluded_haplotype = relationships_all[haplotype_mask].copy()
relationships = relationships_all[~haplotype_mask].copy()

print("\n=== SCOPE FILTER ===")
print("Raw relationship rows:", len(relationships_all))
print("Rows involving Haplotype excluded:", len(excluded_haplotype))
print("Rows remaining in four-node scope:", len(relationships))

genes = pd.read_csv(
    DATA / "genes.tsv",
    sep="\t",
    dtype=str,
).fillna("")

chemicals = pd.read_csv(
    DATA / "chemicals.tsv",
    sep="\t",
    dtype=str,
).fillna("")

phenotypes = pd.read_csv(
    DATA / "phenotypes.tsv",
    sep="\t",
    dtype=str,
).fillna("")

variants = pd.read_csv(
    DATA / "variants.tsv",
    sep="\t",
    dtype=str,
).fillna("")


# ------------------------------------------------------------------
# 1. Build base nodes from dedicated entity tables
# ------------------------------------------------------------------

base_parts = []

base_specs = [
    (
        chemicals,
        "PharmGKB Accession Id",
        "Name",
        "drug",
        "chemical",
        "chemicals.tsv",
    ),
    (
        genes,
        "PharmGKB Accession Id",
        "Symbol",
        "gene",
        "gene",
        "genes.tsv",
    ),
    (
        phenotypes,
        "PharmGKB Accession Id",
        "Name",
        "phenotype",
        "disease",
        "phenotypes.tsv",
    ),
    (
        variants,
        "Variant ID",
        "Variant Name",
        "variant",
        "sequence_variant",
        "variants.tsv",
    ),
]

for df, id_col, name_col, node_type, node_subtype, source in base_specs:
    part = pd.DataFrame({
        "entity_id": df[id_col].str.strip(),
        "node_name": df[name_col].map(clean_name),
        "node_type": node_type,
        "node_subtype": node_subtype,
        "node_source": source,
    })

    part = part[
        (part["entity_id"] != "") |
        part["node_name"].notna()
    ].copy()

    base_parts.append(part)

base_nodes = pd.concat(base_parts, ignore_index=True)


# ------------------------------------------------------------------
# 2. Build unique relationship endpoint records
# ------------------------------------------------------------------

endpoint_parts = []

for side in ["1", "2"]:
    part = relationships[
        [
            f"Entity{side}_id",
            f"Entity{side}_name",
            f"Entity{side}_type",
        ]
    ].copy()

    part.columns = [
        "entity_id",
        "endpoint_name",
        "raw_type",
    ]

    endpoint_parts.append(part)

endpoints = (
    pd.concat(endpoint_parts, ignore_index=True)
    .drop_duplicates()
)

endpoints["entity_id"] = endpoints["entity_id"].str.strip()
endpoints["endpoint_name"] = endpoints["endpoint_name"].map(clean_name)
endpoints["node_type"] = endpoints["raw_type"].map(canonical_type)
endpoints["node_subtype"] = endpoints["raw_type"].map(subtype)


# ------------------------------------------------------------------
# 3. Create stable node keys
# ------------------------------------------------------------------

def make_key(node_type, entity_id, name):
    if entity_id:
        return f"{node_type}|id|{entity_id}"

    if pd.notna(name):
        return f"{node_type}|name|{str(name).casefold()}"

    return None


base_nodes["node_key"] = base_nodes.apply(
    lambda r: make_key(
        r["node_type"],
        r["entity_id"],
        r["node_name"],
    ),
    axis=1,
)

endpoints["node_key"] = endpoints.apply(
    lambda r: make_key(
        r["node_type"],
        r["entity_id"],
        r["endpoint_name"],
    ),
    axis=1,
)


# ------------------------------------------------------------------
# 4. Construct supplemental endpoint nodes
# ------------------------------------------------------------------

base_keys = set(base_nodes["node_key"].dropna())

supplemental = endpoints[
    ~endpoints["node_key"].isin(base_keys)
].copy()

supplemental_nodes = pd.DataFrame({
    "entity_id": supplemental["entity_id"],
    "node_name": supplemental["endpoint_name"],
    "node_type": supplemental["node_type"],
    "node_subtype": supplemental["node_subtype"],
    "node_source": "relationship_endpoint",
    "node_key": supplemental["node_key"],
})

supplemental_nodes = (
    supplemental_nodes
    .dropna(subset=["node_key"])
    .sort_values([
        "node_type",
        "node_subtype",
        "entity_id",
        "node_name",
    ])
    .drop_duplicates("node_key")
)

nodes = (
    pd.concat(
        [base_nodes, supplemental_nodes],
        ignore_index=True,
    )
    .dropna(subset=["node_key"])
    .sort_values([
        "node_type",
        "node_subtype",
        "entity_id",
        "node_name",
    ])
    .drop_duplicates("node_key")
    .reset_index(drop=True)
)

nodes["node_index"] = nodes.index

node_key_to_idx = dict(
    zip(nodes["node_key"], nodes["node_index"])
)


# ------------------------------------------------------------------
# 5. Build relationship endpoint keys and map edges
# ------------------------------------------------------------------

rel = relationships.copy()

for side in ["1", "2"]:
    rel[f"type{side}"] = rel[f"Entity{side}_type"].map(canonical_type)
    rel[f"subtype{side}"] = rel[f"Entity{side}_type"].map(subtype)
    rel[f"name{side}"] = rel[f"Entity{side}_name"].map(clean_name)
    rel[f"id{side}"] = rel[f"Entity{side}_id"].str.strip()

    rel[f"key{side}"] = rel.apply(
        lambda r: make_key(
            r[f"type{side}"],
            r[f"id{side}"],
            r[f"name{side}"],
        ),
        axis=1,
    )

    rel[f"idx{side}"] = rel[f"key{side}"].map(node_key_to_idx)

rel["evidence_canonical"] = rel["Evidence"].map(normalise_evidence)
rel["association_canonical"] = rel["Association"].map(canonical_assoc)

rel["relation_type"] = (
    rel["type1"] + "_" +
    rel["type2"] + "_" +
    rel["evidence_canonical"] + "_" +
    rel["association_canonical"]
)


# ------------------------------------------------------------------
# 6. Audit results
# ------------------------------------------------------------------

print("\n=== NODE SUMMARY ===")
print("Base nodes before key deduplication:", len(base_nodes))
print("Supplemental nodes:", len(supplemental_nodes))
print("Final nodes:", len(nodes))

print("\nFinal nodes by model type:")
print(nodes["node_type"].value_counts().sort_index().to_string())

print("\nFinal nodes by subtype:")
print(nodes["node_subtype"].value_counts().sort_index().to_string())

print("\nSupplemental nodes by subtype:")
print(
    supplemental_nodes["node_subtype"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\n=== EMPTY-ID ENDPOINTS ===")
blank_id = endpoints[endpoints["entity_id"] == ""]
print("Unique endpoint records with blank ID:", len(blank_id))
print(
    blank_id["raw_type"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nBlank-ID examples:")
print(
    blank_id[
        ["raw_type", "endpoint_name"]
    ]
    .head(30)
    .to_string(index=False)
)

print("\n=== EDGE RETENTION ===")
print("Raw relationship rows:", len(rel))
print("Unmapped sources:", rel["idx1"].isna().sum())
print("Unmapped targets:", rel["idx2"].isna().sum())

unmapped_either = rel[
    rel["idx1"].isna() |
    rel["idx2"].isna()
]

print("Rows with either endpoint unmapped:", len(unmapped_either))
print("Mapped rows:", len(rel) - len(unmapped_either))
print(
    "Retention percentage:",
    round(
        100 * (len(rel) - len(unmapped_either)) / len(rel),
        4,
    ),
)

mapped = rel.dropna(
    subset=["idx1", "idx2"]
).copy()

mapped["idx1"] = mapped["idx1"].astype(int)
mapped["idx2"] = mapped["idx2"].astype(int)

edge_key = [
    "idx1",
    "idx2",
    "type1",
    "type2",
    "evidence_canonical",
    "association_canonical",
]

print("\n=== EDGE DEDUPLICATION EFFECT ===")
print(
    "Exact duplicate mapped edge rows:",
    mapped.duplicated(edge_key).sum(),
)

edge_groups = mapped.groupby(edge_key).size()

print(
    "Unique edge keys repeated:",
    int((edge_groups > 1).sum()),
)

print("Mapped rows before deduplication:", len(mapped))
print("Mapped unique edge keys:", len(edge_groups))

print("\n=== RELATION TYPES ===")
print("Canonical relation types:", mapped["relation_type"].nunique())

print(
    mapped["relation_type"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\n=== ID ALIAS COLLAPSE ===")

alias_counts = (
    endpoints[
        endpoints["entity_id"] != ""
    ]
    .groupby(["node_type", "entity_id"])["endpoint_name"]
    .nunique()
)

print(
    "IDs with multiple relationship names:",
    int((alias_counts > 1).sum()),
)

print(
    "Maximum names linked to one ID:",
    int(alias_counts.max()),
)

print("\n=== UNMAPPED EXAMPLES ===")
if len(unmapped_either):
    print(
        unmapped_either[
            [
                "Entity1_id",
                "Entity1_name",
                "Entity1_type",
                "Entity2_id",
                "Entity2_name",
                "Entity2_type",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )
else:
    print("None")
