from pathlib import Path
import pandas as pd

import re
import numpy as np

CONFIG = {
    "data": {
        "files": {
            "relationships": "relationships",
            "genes": "genes",
            "chemicals": "chemicals",
            "phenotypes": "phenotypes",
            "variants": "variants",
        }
    }
}

def clean_columns(df):
    df.columns = [
        re.sub(r"\\s+", " ", str(c).replace("\\ufeff", "").strip())
        for c in df.columns
    ]
    return df

def find_file(data_dir, stem):
    for ext in [".tsv", ".csv", ".txt"]:
        matches = list(data_dir.rglob(f"*{stem}*{ext}"))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"Could not find {stem} in {data_dir}. Make sure files are present."
    )

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

def canonical_type(x):
    x = str(x).strip().lower()
    if x in ["chemical", "chemicals", "drug", "compound"]:
        return "drug"
    if x in ["gene", "genes"]:
        return "gene"
    if x in [
        "phenotype", "phenotypes", "disease", "diseases",
        "clinical phenotype"
    ]:
        return "phenotype"
    if x in ["variant", "variants", "haplotype", "haplotypes"]:
        return "variant"
    return x

def pick_col(df, candidates, label):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    raise ValueError(
        f"No suitable {label} column found. Tried: {candidates}. "
        f"Available: {df.columns.tolist()}"
    )

def clean_node_name(x):
    x = str(x).strip()
    x = re.sub(r"\\s+", " ", x)
    if x.lower() in ["nan", "none", "", "null"]:
        return np.nan
    return x

def read_table(file_path):
    sep = "\\t" if file_path.suffix == ".tsv" else ","
    return clean_columns(pd.read_csv(file_path, sep=sep))

DATA_DIR = Path("data")
files_cfg = CONFIG["data"]["files"]

relationships_file = find_file(DATA_DIR, files_cfg["relationships"])
genes_file = find_file(DATA_DIR, files_cfg["genes"])
chemicals_file = find_file(DATA_DIR, files_cfg["chemicals"])
phenotypes_file = find_file(DATA_DIR, files_cfg["phenotypes"])
variants_file = find_file(DATA_DIR, files_cfg["variants"])

relationships = read_table(relationships_file)
genes_df = read_table(genes_file)
chemicals_df = read_table(chemicals_file)
phenotypes_df = read_table(phenotypes_file)
variants_df = read_table(variants_file)

def build_node_subset(df, type_name, name_candidates):
    name_col = pick_col(df, name_candidates, f"{type_name} name")
    raw = df[name_col]
    cleaned = raw.map(clean_node_name)

    return pd.DataFrame({
        "node_type": type_name,
        "raw_name": raw,
        "node_name": cleaned,
    })

node_parts = [
    build_node_subset(chemicals_df, "drug", ["Name", "name", "chemical"]),
    build_node_subset(genes_df, "gene", ["Symbol", "symbol", "name"]),
    build_node_subset(phenotypes_df, "phenotype", ["Name", "name"]),
    build_node_subset(
        variants_df,
        "variant",
        ["Variant Name", "Name", "name"],
    ),
]

print("\n=== SOURCE FILES ===")
for name, path in {
    "relationships": relationships_file,
    "genes": genes_file,
    "chemicals": chemicals_file,
    "phenotypes": phenotypes_file,
    "variants": variants_file,
}.items():
    print(f"{name}: {path.name}")

print("\n=== NODE CONSTRUCTION ===")
clean_node_parts = []

for part in node_parts:
    node_type = part["node_type"].iloc[0]
    n_raw = len(part)
    n_missing_after_clean = part["node_name"].isna().sum()

    valid = part.dropna(subset=["node_name"]).copy()
    n_valid = len(valid)
    n_duplicate_names = valid.duplicated("node_name").sum()
    n_unique = valid["node_name"].nunique()

    print(
        f"{node_type}: raw={n_raw}, "
        f"missing_after_clean={n_missing_after_clean}, "
        f"duplicate_clean_names={n_duplicate_names}, "
        f"unique_nodes={n_unique}"
    )

    duplicate_examples = (
        valid[valid.duplicated("node_name", keep=False)]
        .sort_values("node_name")
        .head(20)
    )
    if len(duplicate_examples):
        print("  Duplicate examples:")
        print(
            duplicate_examples[
                ["raw_name", "node_name"]
            ].to_string(index=False)
        )

    clean_node_parts.append(
        valid[["node_type", "node_name"]]
        .drop_duplicates("node_name")
    )

nodes = pd.concat(clean_node_parts, ignore_index=True)
nodes = nodes.drop_duplicates(["node_type", "node_name"]).reset_index(drop=True)
nodes["node_index"] = nodes.index

print("\nFinal nodes:", len(nodes))
print(nodes["node_type"].value_counts().sort_index().to_string())

src_col = pick_col(
    relationships,
    ["Entity1_name", "source", "Entity 1 Name"],
    "source",
)
tgt_col = pick_col(
    relationships,
    ["Entity2_name", "target", "Entity 2 Name"],
    "target",
)
src_t_col = pick_col(
    relationships,
    ["Entity1_type", "source_type", "Entity 1 Type"],
    "source_type",
)
tgt_t_col = pick_col(
    relationships,
    ["Entity2_type", "target_type", "Entity 2 Type"],
    "target_type",
)
assoc_col = pick_col(
    relationships,
    ["Association", "association"],
    "association",
)
evid_col = pick_col(
    relationships,
    ["Evidence", "evidence", "PK"],
    "evidence",
)

rel = relationships[
    [src_col, tgt_col, src_t_col, tgt_t_col, assoc_col, evid_col]
].copy()

rel.columns = [
    "source_name_raw",
    "target_name_raw",
    "source_type_raw",
    "target_type_raw",
    "association_raw",
    "evidence_raw",
]

print("\n=== RAW RELATIONSHIPS ===")
print("Raw relationship rows:", len(rel))

rel["source_name"] = rel["source_name_raw"].map(clean_node_name)
rel["target_name"] = rel["target_name_raw"].map(clean_node_name)
rel["source_type"] = rel["source_type_raw"].map(canonical_type)
rel["target_type"] = rel["target_type_raw"].map(canonical_type)
rel["evidence"] = rel["evidence_raw"].map(normalise_evidence)
rel["association"] = (
    rel["association_raw"]
    .astype(str)
    .str.lower()
    .str.strip()
)

print("\n=== TYPE NORMALISATION ===")
print("Raw source types:")
print(rel["source_type_raw"].value_counts(dropna=False).to_string())
print("\nCanonical source types:")
print(rel["source_type"].value_counts(dropna=False).to_string())
print("\nRaw target types:")
print(rel["target_type_raw"].value_counts(dropna=False).to_string())
print("\nCanonical target types:")
print(rel["target_type"].value_counts(dropna=False).to_string())

missing_names = rel[
    rel["source_name"].isna() |
    rel["target_name"].isna()
]

print("\nRows removed because endpoint name is missing:", len(missing_names))

rel_nonmissing = rel.dropna(
    subset=["source_name", "target_name"]
).copy()

name_idx = nodes.set_index(
    ["node_type", "node_name"]
)["node_index"].to_dict()

rel_nonmissing["source_idx"] = rel_nonmissing.apply(
    lambda r: name_idx.get(
        (r["source_type"], r["source_name"])
    ),
    axis=1,
)

rel_nonmissing["target_idx"] = rel_nonmissing.apply(
    lambda r: name_idx.get(
        (r["target_type"], r["target_name"])
    ),
    axis=1,
)

source_unmapped = rel_nonmissing[
    rel_nonmissing["source_idx"].isna()
]

target_unmapped = rel_nonmissing[
    rel_nonmissing["target_idx"].isna()
]

either_unmapped = rel_nonmissing[
    rel_nonmissing["source_idx"].isna() |
    rel_nonmissing["target_idx"].isna()
]

print("\n=== MAPPING LOSSES ===")
print("Unmapped source rows:", len(source_unmapped))
print("Unmapped target rows:", len(target_unmapped))
print("Rows with either endpoint unmapped:", len(either_unmapped))

if len(source_unmapped):
    print("\nTop unmapped source names:")
    print(
        source_unmapped.groupby(
            ["source_type_raw", "source_type", "source_name"]
        )
        .size()
        .sort_values(ascending=False)
        .head(30)
        .to_string()
    )

if len(target_unmapped):
    print("\nTop unmapped target names:")
    print(
        target_unmapped.groupby(
            ["target_type_raw", "target_type", "target_name"]
        )
        .size()
        .sort_values(ascending=False)
        .head(30)
        .to_string()
    )

mapped = rel_nonmissing.dropna(
    subset=["source_idx", "target_idx"]
).copy()

mapped["source_idx"] = mapped["source_idx"].astype(int)
mapped["target_idx"] = mapped["target_idx"].astype(int)

mapped["association_canonical"] = (
    mapped["association"]
    .str.replace("-", "_", regex=False)
    .str.replace(" ", "_", regex=False)
)

mapped["relation_type_current"] = (
    mapped["source_type"] + "_" +
    mapped["target_type"] + "_" +
    mapped["evidence"] + "_" +
    mapped["association"]
)

mapped["relation_type_canonical"] = (
    mapped["source_type"] + "_" +
    mapped["target_type"] + "_" +
    mapped["evidence"] + "_" +
    mapped["association_canonical"]
)

print("\n=== ASSOCIATION LABELS ===")
print("Current labels:")
print(mapped["association"].value_counts(dropna=False).to_string())
print("\nCanonical labels:")
print(
    mapped["association_canonical"]
    .value_counts(dropna=False)
    .to_string()
)

changed = mapped[
    mapped["association"] !=
    mapped["association_canonical"]
]

print("\nRows whose label changes under canonicalisation:", len(changed))

print("\n=== DUPLICATE EDGE AUDIT ===")
structural_key = [
    "source_idx",
    "target_idx",
    "source_type",
    "target_type",
    "evidence",
    "association_canonical",
]

print(
    "Exact duplicate mapped relationships:",
    mapped.duplicated(structural_key).sum(),
)

duplicate_groups = (
    mapped.groupby(structural_key)
    .size()
    .sort_values(ascending=False)
)

print(
    "Unique edge keys represented more than once:",
    (duplicate_groups > 1).sum(),
)

print("\nLargest duplicate groups:")
print(duplicate_groups[duplicate_groups > 1].head(30).to_string())

print("\n=== CONTRADICTORY LABEL AUDIT ===")
pair_key = [
    "source_idx",
    "target_idx",
    "source_type",
    "target_type",
    "evidence",
]

pair_labels = (
    mapped.groupby(pair_key)["association_canonical"]
    .agg(lambda x: sorted(set(x)))
)

contradictory = pair_labels[
    pair_labels.map(len) > 1
]

print(
    "Endpoint/evidence groups with multiple association labels:",
    len(contradictory),
)

print(contradictory.head(30).to_string())

print("\n=== FINAL RETENTION ===")
print("Raw rows:", len(rel))
print("Mapped rows retained:", len(mapped))
print(
    "Retention percentage:",
    round(100 * len(mapped) / len(rel), 2),
)

print(
    "Current relation types:",
    mapped["relation_type_current"].nunique(),
)

print(
    "Canonicalised relation types:",
    mapped["relation_type_canonical"].nunique(),
)

print("\n=== MAPPING LOSS BY TYPE PAIR ===")
loss_by_pair = (
    either_unmapped
    .groupby([
        "source_type_raw",
        "target_type_raw",
    ])
    .size()
    .sort_values(ascending=False)
)
print(loss_by_pair.to_string())

print("\n=== UNIQUE UNMAPPED ENTITIES ===")

source_missing_unique = (
    source_unmapped[
        ["source_type_raw", "source_type", "source_name"]
    ]
    .drop_duplicates()
)

target_missing_unique = (
    target_unmapped[
        ["target_type_raw", "target_type", "target_name"]
    ]
    .drop_duplicates()
)

print("\nUnique unmapped sources by type:")
print(
    source_missing_unique["source_type_raw"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nUnique unmapped targets by type:")
print(
    target_missing_unique["target_type_raw"]
    .value_counts(dropna=False)
    .to_string()
)

print("\n=== POTENTIAL NODE EXPANSION ===")

missing_entities = pd.concat([
    source_missing_unique.rename(columns={
        "source_type": "node_type",
        "source_name": "node_name",
    })[["node_type", "node_name"]],
    target_missing_unique.rename(columns={
        "target_type": "node_type",
        "target_name": "node_name",
    })[["node_type", "node_name"]],
]).drop_duplicates()

print("Additional unique nodes required:", len(missing_entities))

print(
    missing_entities["node_type"]
    .value_counts()
    .sort_index()
    .to_string()
)

expanded_nodes = pd.concat([
    nodes[["node_type", "node_name"]],
    missing_entities,
]).drop_duplicates()

print("Current nodes:", len(nodes))
print("Potential expanded nodes:", len(expanded_nodes))
