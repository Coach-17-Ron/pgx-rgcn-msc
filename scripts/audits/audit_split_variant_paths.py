from pathlib import Path
import pandas as pd

split_files = sorted(
    Path("results").glob(
        "*beta0p0_seed*_reg-default/artifacts/split_assignments.parquet"
    )
)

print("Split files selected:", len(split_files))
for p in split_files:
    print(p)

for path in split_files:
    df = pd.read_parquet(path)

    # Training graph only
    train = df[df["split"] == "train"].copy()

    # Test drug-gene rows in either direction, normalised as drug → gene
    test_dg = df[
        (df["split"] == "test") &
        (
            (
                (df["source_type"] == "drug") &
                (df["target_type"] == "gene")
            ) |
            (
                (df["source_type"] == "gene") &
                (df["target_type"] == "drug")
            )
        )
    ].copy()

    def normalise_pair(row):
        if row["source_type"] == "drug":
            return row["source_idx"], row["target_idx"]
        return row["target_idx"], row["source_idx"]

    if len(test_dg):
        test_dg[["drug_idx", "gene_idx"]] = pd.DataFrame(
            test_dg.apply(normalise_pair, axis=1).tolist(),
            index=test_dg.index,
        )

    # Unique test pairs rather than counting forward and reverse twice
    test_pairs = (
        test_dg[
            [
                "drug_idx",
                "gene_idx",
                "association_norm",
                "evidence",
            ]
        ]
        .drop_duplicates()
    )

    # Direct test-pair rows that accidentally remain in training,
    # including the reverse orientation
    train_dg = train[
        (
            (train["source_type"] == "drug") &
            (train["target_type"] == "gene")
        ) |
        (
            (train["source_type"] == "gene") &
            (train["target_type"] == "drug")
        )
    ].copy()

    if len(train_dg):
        train_dg[["drug_idx", "gene_idx"]] = pd.DataFrame(
            train_dg.apply(normalise_pair, axis=1).tolist(),
            index=train_dg.index,
        )

    train_direct_pairs = set(
        train_dg[["drug_idx", "gene_idx"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    test_pairs["direct_pair_in_train"] = [
        (d, g) in train_direct_pairs
        for d, g in test_pairs[
            ["drug_idx", "gene_idx"]
        ].itertuples(index=False, name=None)
    ]

    # Training drug → variant edges, normalised
    train_dv = train[
        (
            (train["source_type"] == "drug") &
            (train["target_type"] == "variant")
        ) |
        (
            (train["source_type"] == "variant") &
            (train["target_type"] == "drug")
        )
    ].copy()

    def normalise_dv(row):
        if row["source_type"] == "drug":
            return row["source_idx"], row["target_idx"]
        return row["target_idx"], row["source_idx"]

    if len(train_dv):
        train_dv[["drug_idx", "variant_idx"]] = pd.DataFrame(
            train_dv.apply(normalise_dv, axis=1).tolist(),
            index=train_dv.index,
        )

    # Training variant → gene edges, normalised
    train_vg = train[
        (
            (train["source_type"] == "variant") &
            (train["target_type"] == "gene")
        ) |
        (
            (train["source_type"] == "gene") &
            (train["target_type"] == "variant")
        )
    ].copy()

    def normalise_vg(row):
        if row["source_type"] == "variant":
            return row["source_idx"], row["target_idx"]
        return row["target_idx"], row["source_idx"]

    if len(train_vg):
        train_vg[["variant_idx", "gene_idx"]] = pd.DataFrame(
            train_vg.apply(normalise_vg, axis=1).tolist(),
            index=train_vg.index,
        )

    paths = (
        train_dv[
            [
                "drug_idx",
                "variant_idx",
                "evidence",
                "association_norm",
            ]
        ]
        .drop_duplicates()
        .merge(
            train_vg[
                [
                    "variant_idx",
                    "gene_idx",
                    "evidence",
                    "association_norm",
                ]
            ].drop_duplicates(),
            on="variant_idx",
            suffixes=("_dv", "_vg"),
        )
    )

    path_pairs = (
        paths.groupby(["drug_idx", "gene_idx"])
        .agg(
            variant_count=("variant_idx", "nunique"),
            evidence_values=(
                "evidence_dv",
                lambda x: set(x),
            ),
            association_values=(
                "association_norm_dv",
                lambda x: set(x),
            ),
        )
        .reset_index()
    )

    audited = test_pairs.merge(
        path_pairs,
        on=["drug_idx", "gene_idx"],
        how="left",
    )

    audited["has_train_variant_path"] = (
        audited["variant_count"].fillna(0) > 0
    )

    audited["same_evidence_path"] = audited.apply(
        lambda r:
            r["has_train_variant_path"] and
            r["evidence"] in r["evidence_values"],
        axis=1,
    )

    audited["same_association_path"] = audited.apply(
        lambda r:
            r["has_train_variant_path"] and
            r["association_norm"] in r["association_values"],
        axis=1,
    )

    print("\n" + "=" * 90)
    print("FILE:", path)
    print("Rows:", len(df))
    print("Train rows:", len(train))
    print("Unique test drug-gene records:", len(test_pairs))

    print(
        "Test pairs whose direct pair remains in train:",
        int(audited["direct_pair_in_train"].sum()),
    )

    print(
        "Test pairs with training drug-variant-gene path:",
        int(audited["has_train_variant_path"].sum()),
    )

    print(
        "Variant-path percentage:",
        round(
            100 * audited["has_train_variant_path"].mean(),
            2,
        ),
    )

    print(
        "Test pairs with same-evidence variant path:",
        int(audited["same_evidence_path"].sum()),
    )

    print(
        "Test pairs with same-association variant path:",
        int(audited["same_association_path"].sum()),
    )

    print("\nBy test association:")
    summary = (
        audited.groupby("association_norm")
        .agg(
            test_pairs=("drug_idx", "size"),
            with_variant_path=(
                "has_train_variant_path",
                "sum",
            ),
            direct_in_train=(
                "direct_pair_in_train",
                "sum",
            ),
        )
    )

    summary["variant_path_pct"] = (
        100 *
        summary["with_variant_path"] /
        summary["test_pairs"]
    ).round(2)

    print(summary.to_string())

    print("\nTop exposed test pairs:")
    print(
        audited[
            audited["has_train_variant_path"]
        ]
        .sort_values("variant_count", ascending=False)
        .head(20)[
            [
                "drug_idx",
                "gene_idx",
                "association_norm",
                "evidence",
                "variant_count",
                "same_evidence_path",
                "same_association_path",
                "direct_pair_in_train",
            ]
        ]
        .to_string(index=False)
    )
