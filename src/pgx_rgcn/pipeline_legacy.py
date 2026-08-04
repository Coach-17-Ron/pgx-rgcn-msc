"""
=================================================================
PHARMACOGENOMICS R-GCN PIPELINE  -  COMBINED RUN SCRIPT
=================================================================
Single-file runnable copy of the 7-script pipeline. Built for the
ilifu HPC run before refactoring back into modules for GitHub.

PROJECT:  MSc Computational Health Informatics mini-dissertation
AUTHOR:   Kgabe Ronald Molepo (MLPKGA007)
SUPERVISORS:  Hocine Bendou, Khuthala Mnika
INSTITUTION:  University of Cape Town

=================================================================
PIPELINE STAGES (run sequentially or selectively)
=================================================================
  1. Build heterogeneous KG from ClinPGx TSVs
  2. Cold-drug 70/15/15 split + hub-penalised R-GCN training
  3. Cold-drug test evaluation with bootstrap CIs + novel predictions
  4. Intrinsic + post-hoc graph-based interpretation
  5. Pathway ORA + real PubMed evidence + integrated scoring
  6. Phenotype propagation (Objective 6 extension)

=================================================================
USAGE
=================================================================
  # Full pipeline:
  python pgx_pipeline_combined.py --data_dir ./data --stages all

  # Specific stages (e.g., re-run interpretation only):
  python pgx_pipeline_combined.py --data_dir ./data --run_id <ID> --stages 4

  # Skip live PubMed (offline / sandbox):
  python pgx_pipeline_combined.py --data_dir ./data --stages all --skip_literature

  # β=0 ablation run:
  python pgx_pipeline_combined.py --data_dir ./data --stages all --ablation_beta_zero

=================================================================
REQUIRED BEFORE RUN
=================================================================
  1. Edit PUBMED_EMAIL in the CONFIG section below to your real email
     (NCBI requires this for accountability — no fake emails).
  2. Place ClinPGx TSV files in <data_dir>/ :
        relationships*.tsv  genes*.tsv  chemicals*.tsv
        phenotypes*.tsv     variants*.tsv
  3. (Optional) Place GMT pathway files in <data_dir>/pathways/ :
        ReactomePathways.gmt
        c2.cp.kegg.v*.symbols.gmt   (from MSigDB)
     Without GMT files, stage 5 will skip pathway ORA but still run
     literature contextualisation.

=================================================================
NEW IN THIS BUILD vs original notebook
=================================================================
  * 70/15/15 cold-drug split (was 90/10 train/test)
  * Validation-MRR early stopping (was train-loss-based)
  * β=0.005 hub penalty on negative-sample gene defaults
  * Negative sampler excludes ALL curated positives across splits
  * Real PubMed evidence via NCBI E-utilities (ESearch+ESummary+EFetch)
  * Gene-level ORA with protein-coding-genes universe (GENCODE v44)
  * Phenotype propagation (1-hop and 2-hop via variant)
  * Negative gene-phenotype evidence flagged as is_negative_evidence

=================================================================
ILIFU SLURM EXAMPLE
=================================================================
  #!/bin/bash
  #SBATCH --job-name=pgx_rgcn
  #SBATCH --time=04:00:00
  #SBATCH --mem=32G
  #SBATCH --cpus-per-task=4
  #SBATCH --output=logs/%j.out
  source ~/venvs/pgx/bin/activate
  python pgx_pipeline_combined.py --data_dir /scratch/$USER/clinpgx --stages all

  # For GPU:
  #SBATCH --partition=GPU
  #SBATCH --gres=gpu:1
=================================================================
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# torch_geometric is heavy; import only if needed for training/eval/interpret
from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv

import networkx as nx
from scipy.stats import hypergeom


# ============================================================================
# CONFIG  (edit PUBMED_EMAIL before running stage 5)
# ============================================================================
CONFIG = {
    "seed": 42,
    "run_id": None,           # None = auto-timestamp; pin a string to reproduce
    "force_cpu": False,        # set True to disable CUDA detection

    "paths": {
        "data_dir": "./data",
        "results_dir": "./results",
        "artifacts_subdir": "artifacts",
        "figures_subdir": "figures",
        "logs_subdir": "logs",
    },

    "data": {
        "drug_fingerprints_path": "./data/drug_fingerprints.parquet",
        "files": {
            "relationships": "relationships",
            "genes": "genes",
            "chemicals": "chemicals",
            "phenotypes": "phenotypes",
            "variants": "variants",
        },
        "sample_drug_count": 25,
        "sample_seed": 42,
    },

    "split": {
        "split_mode": "dg_context",  # "dg_context" (main) or "strict_cold_drug" (sensitivity)
        "train_drug_fraction": 0.70,
        "val_drug_fraction": 0.15,
        "test_drug_fraction": 0.15,
    },

    "model": {
        "embedding_dim": 64,
        "hidden_dim": 64,
        "output_dim": 64,
        "num_bases": 8,
        "dropout": 0.25,
    },

    "training": {
        "epochs": 100,
        "batch_size": 512,
        "num_negatives": 8,
        "hub_penalty_beta": 0.005,
        "margin": 1.0,
        "learning_rate": 0.003,
        "weight_decay": 1.0e-4,
        "grad_clip_norm": 2.0,
        "patience": 10,
        "min_delta": 1.0e-4,
        "validate_every": 5,
    },

    "evaluation": {
        "hits_at_k": [1, 3, 5, 10],
        "bootstrap_samples": 1000,
        "bootstrap_seed": 12345,
    },

    "interpretation": {
        "top_n_predictions": 50,
        "explanation_hops": [2, 3, 4],
        "gnnexplainer_epochs": 200,
    },

    "contextualisation": {
        "literature": {
            "max_pairs_to_query": 50,
            "articles_per_pair": 10,
            "date_from": 2020,
            "date_to": 2026,
            "request_sleep_seconds": 0.34,
            # >>> EDIT THIS BEFORE RUNNING STAGE 5 <<<
            "pubmed_email": "mlpkga007@myuct.ac.za",
        },
        "pathways": {
            "background_source": "protein_coding_genes",
            "protein_coding_count": 19969,
            "top_n_per_list": 50,
            "fdr_method": "fdr_bh",
            "fdr_threshold": 0.05,
            "reactome_endpoint": "https://reactome.org/ContentService/data/mapping/UniProt",
            "kegg_endpoint": "https://rest.kegg.jp",
            "request_sleep_seconds": 0.25,
        },
        "integrated_score_weights": {
            "hub_adjusted": 0.40,
            "literature": 0.30,
            "pathway": 0.30,
        },
        "integrated_top_n": 50,
    },
}


def load_config(cfg=None):
    """Return the inline CONFIG (or a passed-in dict), with validation."""
    if cfg is None:
        cfg = CONFIG
    s = cfg["split"]
    total = s["train_drug_fraction"] + s["val_drug_fraction"] + s["test_drug_fraction"]
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split fractions sum to {total}, must equal 1.0")
    w = cfg["contextualisation"]["integrated_score_weights"]
    if abs(sum(w.values()) - 1.0) > 1e-6:
        raise ValueError(
            f"Integrated score weights sum to {sum(w.values())}, must equal 1.0"
        )
    return cfg

# ============================================================================
# SHARED UTILITIES  (paths, device, seed, logger, versions)
# ============================================================================

def setup_paths(cfg: dict, run_id: str = None) -> dict:
    p = cfg["paths"]
    if run_id is None:
        run_id = cfg.get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(".").resolve()
    data_dir = (root / p["data_dir"]).resolve()
    results_dir = (root / p["results_dir"] / run_id).resolve()
    artifacts_dir = results_dir / p["artifacts_subdir"]
    figures_dir = results_dir / p["figures_subdir"]
    logs_dir = results_dir / p["logs_subdir"]
    for d in [data_dir, results_dir, artifacts_dir, figures_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return {
        "run_id": run_id,
        "root": root,
        "data_dir": data_dir,
        "results_dir": results_dir,
        "artifacts_dir": artifacts_dir,
        "figures_dir": figures_dir,
        "logs_dir": logs_dir,
    }


def select_device(cfg: dict) -> torch.device:
    if cfg.get("force_cpu", False):
        print("Device: FORCE_CPU set -> using CPU.")
        return torch.device("cpu")
    if not torch.cuda.is_available():
        print("Device: no CUDA device visible -> using CPU.")
        return torch.device("cpu")
    try:
        t = torch.randn(8, 8, device="cuda")
        _ = (t @ t).sum().item()
        torch.cuda.synchronize()
        name = torch.cuda.get_device_name(0)
        print(f"Device: GPU compute test passed -> using CUDA ({name}).")
        return torch.device("cuda")
    except Exception as e:
        print(f"Device: GPU visible but cannot run kernels ({type(e).__name__}). "
              f"Falling back to CPU. Detail: {e}")
        return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN settings — slower but reproducible
    # (the warnings about non-deterministic ops can be ignored in our case;
    # RGCNConv has no documented non-deterministic kernels at our scale)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass  # not all torch builds expose cudnn


def setup_logger(log_path: Path, name: str = "pipeline") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def log_versions(logger: logging.Logger) -> None:
    import torch_geometric
    logger.info(f"Python: {sys.version.split()[0]}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"PyG: {torch_geometric.__version__}")
    logger.info(f"NumPy: {np.__version__}")


# ============================================================================
# STAGE 1: BUILD HETEROGENEOUS KNOWLEDGE GRAPH FROM CLINPGX
# ============================================================================





# ====================== Helpers (preserved verbatim) ======================
# These helpers are reproduced from Block 3 of the original code so the
# data-standardisation logic is identical.

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [
        re.sub(r"\s+", " ", str(c).replace("\ufeff", "").strip())
        for c in df.columns
    ]
    return df


def find_file(data_dir: Path, stem: str) -> Path:
    for ext in [".tsv", ".csv", ".txt"]:
        matches = list(data_dir.rglob(f"*{stem}*{ext}"))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"Could not find {stem} in {data_dir}. Make sure files are present."
    )


def normalise_evidence(x) -> str:
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


def canonical_type(x) -> str:
    x = str(x).strip().lower()
    if x in ["chemical", "chemicals", "drug", "compound"]:
        return "drug"
    if x in ["gene", "genes"]:
        return "gene"
    if x in ["phenotype", "phenotypes", "disease", "diseases", "clinical phenotype"]:
        return "phenotype"
    if x in ["variant", "variants", "haplotype", "haplotypes"]:
        return "variant"
    return x


def pick_col(df: pd.DataFrame, candidates, label: str) -> str:
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
    x = re.sub(r"\s+", " ", x)
    if x.lower() in ["nan", "none", "", "null"]:
        return np.nan
    return x


def read_table(file_path: Path) -> pd.DataFrame:
    return clean_columns(
        pd.read_csv(file_path, sep="\t" if file_path.suffix == ".tsv" else ",")
    )


# ====================== Main =========================================
def run_stage_01_build_graph(cfg, paths, logger, args):


    logger.info(f"RUN_ID: {paths['run_id']}")
    logger.info(f"DATA_DIR: {paths['data_dir']}")
    logger.info(f"RESULTS_DIR: {paths['results_dir']}")

    # --- Locate raw files ---
    logger.info("Locating raw ClinPGx files...")
    data_dir = paths["data_dir"]
    files_cfg = cfg["data"]["files"]
    relationships_file = find_file(data_dir, files_cfg["relationships"])
    genes_file = find_file(data_dir, files_cfg["genes"])
    chemicals_file = find_file(data_dir, files_cfg["chemicals"])
    phenotypes_file = find_file(data_dir, files_cfg["phenotypes"])
    variants_file = find_file(data_dir, files_cfg["variants"])

    logger.info(f"  relationships: {relationships_file.name}")
    logger.info(f"  genes:         {genes_file.name}")
    logger.info(f"  chemicals:     {chemicals_file.name}")
    logger.info(f"  phenotypes:    {phenotypes_file.name}")
    logger.info(f"  variants:      {variants_file.name}")

    # --- Load tables ---
    relationships = read_table(relationships_file)
    genes_df = read_table(genes_file)
    chemicals_df = read_table(chemicals_file)
    phenotypes_df = read_table(phenotypes_file)
    variants_df = read_table(variants_file)

    # --- Build canonical node table ---
    # Each node type is unioned into a single nodes table with a unique
    # node_index (the integer identifier used by all downstream code).
    logger.info("Building canonical node table...")

    def build_node_subset(df, type_name, name_candidates):
        name_col = pick_col(df, name_candidates, f"{type_name} name")
        sub = pd.DataFrame({
            "node_type": type_name,
            "node_name": df[name_col].map(clean_node_name)
        })
        return sub.dropna(subset=["node_name"]).drop_duplicates("node_name")

    drugs_sub = build_node_subset(chemicals_df, "drug", ["Name", "name", "chemical"])
    genes_sub = build_node_subset(genes_df, "gene", ["Symbol", "symbol", "name"])
    phen_sub = build_node_subset(phenotypes_df, "phenotype", ["Name", "name"])
    var_sub = build_node_subset(variants_df, "variant", ["Variant Name", "Name", "name"])

    nodes = pd.concat([drugs_sub, genes_sub, phen_sub, var_sub], ignore_index=True)
    nodes = nodes.drop_duplicates(["node_type", "node_name"]).reset_index(drop=True)
    nodes["node_index"] = nodes.index
    logger.info(f"Total nodes: {len(nodes)}")
    for nt, n in nodes["node_type"].value_counts().items():
        logger.info(f"  {nt:>9}: {n}")

    # --- Build edge table from relationships ---
    logger.info("Building canonical edge table...")
    src_col = pick_col(relationships, ["Entity1_name", "source", "Entity 1 Name"], "source")
    tgt_col = pick_col(relationships, ["Entity2_name", "target", "Entity 2 Name"], "target")
    src_t_col = pick_col(relationships, ["Entity1_type", "source_type", "Entity 1 Type"], "source_type")
    tgt_t_col = pick_col(relationships, ["Entity2_type", "target_type", "Entity 2 Type"], "target_type")
    assoc_col = pick_col(relationships, ["Association", "association"], "association")
    evid_col = pick_col(relationships, ["Evidence", "evidence", "PK"], "evidence")

    rel = relationships[[src_col, tgt_col, src_t_col, tgt_t_col, assoc_col, evid_col]].copy()
    rel.columns = ["source_name", "target_name", "source_type", "target_type",
                   "association", "evidence_raw"]
    rel["source_name"] = rel["source_name"].map(clean_node_name)
    rel["target_name"] = rel["target_name"].map(clean_node_name)
    rel = rel.dropna(subset=["source_name", "target_name"])
    rel["source_type"] = rel["source_type"].map(canonical_type)
    rel["target_type"] = rel["target_type"].map(canonical_type)
    rel["evidence"] = rel["evidence_raw"].map(normalise_evidence)
    rel["association"] = rel["association"].astype(str).str.lower().str.strip()

    # Map names -> node_index
    name_idx = nodes.set_index(["node_type", "node_name"])["node_index"].to_dict()
    rel["source_idx"] = rel.apply(
        lambda r: name_idx.get((r["source_type"], r["source_name"])), axis=1
    )
    rel["target_idx"] = rel.apply(
        lambda r: name_idx.get((r["target_type"], r["target_name"])), axis=1
    )
    rel = rel.dropna(subset=["source_idx", "target_idx"]).copy()
    rel["source_idx"] = rel["source_idx"].astype(int)
    rel["target_idx"] = rel["target_idx"].astype(int)

    # Compose a relation_type that captures direction + evidence + association
    rel["relation_type"] = (
        rel["source_type"] + "_" + rel["target_type"] + "_" +
        rel["evidence"] + "_" + rel["association"]
    )

    edges = rel[[
        "source_idx", "target_idx", "source_type", "target_type",
        "relation_type", "evidence", "association"
    ]].reset_index(drop=True)

    logger.info(f"Total edges: {len(edges)}")
    logger.info(f"Unique relation types: {edges['relation_type'].nunique()}")

    # --- Save artifacts ---
    nodes_path = paths["artifacts_dir"] / "nodes.parquet"
    edges_path = paths["artifacts_dir"] / "edges.parquet"
    nodes.to_parquet(nodes_path, index=False)
    edges.to_parquet(edges_path, index=False)

    metadata = {
        "run_id": paths["run_id"],
        "n_nodes": int(len(nodes)),
        "n_edges": int(len(edges)),
        "node_type_counts": nodes["node_type"].value_counts().to_dict(),
        "relation_type_count": int(edges["relation_type"].nunique()),
        "relation_types": sorted(edges["relation_type"].unique().tolist()),
        "source_files": {
            "relationships": str(relationships_file.name),
            "genes": str(genes_file.name),
            "chemicals": str(chemicals_file.name),
            "phenotypes": str(phenotypes_file.name),
            "variants": str(variants_file.name),
        },
        "config_snapshot": cfg,
    }
    meta_path = paths["artifacts_dir"] / "graph_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"Saved: {nodes_path}")
    logger.info(f"Saved: {edges_path}")
    logger.info(f"Saved: {meta_path}")
    logger.info("Script 01 complete.")



# ============================================================================
# STAGE 2: COLD-DRUG SPLIT + HUB-PENALISED R-GCN TRAINING
# ============================================================================




# ====================== Model definitions (verbatim from original) ====
class TrueRGCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_relations, num_bases=8, dropout=0.25):
        super().__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels,
                              num_relations=num_relations, num_bases=num_bases)
        self.conv2 = RGCNConv(hidden_channels, out_channels,
                              num_relations=num_relations, num_bases=num_bases)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        return x


class DistMultDecoder(nn.Module):
    def __init__(self, num_relations, hidden_channels):
        super().__init__()
        self.rel_emb = nn.Embedding(num_relations, hidden_channels)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, head_emb, rel_idx, tail_emb):
        r_emb = self.rel_emb(rel_idx)
        return (head_emb * r_emb * tail_emb).sum(dim=-1)


class PGx_RGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_edge_types, num_decoder_relations,
                 num_bases=8, dropout=0.25,
                 drug_fp_dim=None, n_drugs_for_fp=0):
        """Optional Round 3 args:
        - drug_fp_dim: dimension of drug input fingerprints (e.g. 2048).
          If None, drug nodes use random Xavier features (legacy behaviour).
        - n_drugs_for_fp: number of drug nodes in the graph (needed to
          pre-register buffers of correct shape).
        """
        super().__init__()
        self.drug_fp_dim = drug_fp_dim
        if drug_fp_dim is not None and n_drugs_for_fp > 0:
            self.drug_fp_proj = nn.Linear(drug_fp_dim, in_channels)
            self.drug_missing_emb = nn.Parameter(torch.empty(in_channels))
            nn.init.xavier_uniform_(self.drug_missing_emb.unsqueeze(0))
            # Buffers (persistent=True so checkpoint is self-contained)
            self.register_buffer(
                "drug_fp_buffer",
                torch.zeros(n_drugs_for_fp, drug_fp_dim),
                persistent=True,
            )
            self.register_buffer(
                "drug_has_fp_buffer",
                torch.zeros(n_drugs_for_fp, dtype=torch.bool),
                persistent=True,
            )
            self.drug_offset = None  # set via attach_drug_fingerprints()
        self.encoder = TrueRGCNEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_relations=num_edge_types,
            num_bases=min(num_bases, num_edge_types),
            dropout=dropout,
        )
        self.decoder = DistMultDecoder(
            num_relations=num_decoder_relations,
            hidden_channels=out_channels,
        )

    def attach_drug_fingerprints(self, drug_fp, has_fp, drug_offset):
        """Populate the pre-registered fingerprint buffers. Called once
        per run after model construction, by stage 2 (train) and stage 3
        (eval). Idempotent; safe to call repeatedly."""
        if self.drug_fp_dim is None:
            raise RuntimeError(
                "Cannot attach drug fingerprints to a model constructed "
                "with drug_fp_dim=None. Re-construct with drug_fp_dim set."
            )
        assert drug_fp.shape == self.drug_fp_buffer.shape, (
            f"drug_fp shape {drug_fp.shape} != buffer shape "
            f"{self.drug_fp_buffer.shape}")
        self.drug_fp_buffer.data.copy_(drug_fp.float())
        self.drug_has_fp_buffer.data.copy_(has_fp.bool())
        self.drug_offset = int(drug_offset)

    def encode(self, x, edge_index, edge_type):
        """Wrapper around encoder that injects drug fingerprints if available.

        For drugs with a Morgan fingerprint, the random Xavier row in x is
        replaced by drug_fp_proj(fingerprint). For drugs without one, the
        row is replaced by a single shared learnable vector
        (drug_missing_emb). Gradients flow through both. If the model was
        constructed without drug_fp_dim, this falls back to a plain
        encoder forward (legacy / backward-compat)."""
        if self.drug_fp_dim is None or self.drug_offset is None:
            return self.encoder(x, edge_index, edge_type)
        n_drugs = self.drug_fp_buffer.size(0)
        proj = self.drug_fp_proj(self.drug_fp_buffer)  # (n_drugs, in_channels)
        missing = self.drug_missing_emb.unsqueeze(0).expand(n_drugs, -1)
        drug_rows = torch.where(
            self.drug_has_fp_buffer.unsqueeze(-1), proj, missing,
        )
        x_mut = x.clone()
        x_mut[self.drug_offset:self.drug_offset + n_drugs] = drug_rows
        return self.encoder(x_mut, edge_index, edge_type)

    def forward(self, x, edge_index, edge_type, head_idx, rel_idx, tail_idx):
        z = self.encode(x, edge_index, edge_type)
        return self.decoder(z[head_idx], rel_idx, z[tail_idx])




# ============================================================================
# Round 3: drug-fingerprint loader
# ============================================================================
def load_drug_fingerprints(fp_path, nodes, logger, fp_dim=2048):
    """Load drug Morgan fingerprints from a parquet and align with the drug
    node indices in `nodes`.

    Returns
    -------
    fp_tensor : torch.Tensor of shape (n_drugs, fp_dim) float32, or None
    has_fp    : torch.Tensor of shape (n_drugs,)        bool,    or None
    n_with    : int  count of drugs with a non-zero fingerprint
    """
    from pathlib import Path as _P
    fp_path = _P(fp_path)
    if not fp_path.exists():
        logger.info(f"No drug-fingerprint parquet at {fp_path}; "
                    f"drug features will be random Xavier (legacy).")
        return None, None, 0

    fp_df = pd.read_parquet(fp_path)
    logger.info(f"Loaded {len(fp_df)} fingerprint rows from {fp_path}")

    # Join by lowercased name (canonical key)
    fp_df = fp_df.copy()
    fp_df["_join_key"] = fp_df["drug_name"].astype(str).str.strip().str.lower()
    fp_map = {}
    for _, r in fp_df.iterrows():
        if r["has_smiles"] and r["fingerprint"] is not None:
            fp_map[r["_join_key"]] = np.asarray(r["fingerprint"], dtype=np.float32)

    drug_nodes = nodes[nodes["node_type"] == "drug"].reset_index(drop=True)
    n_drugs = len(drug_nodes)
    fp_arr = np.zeros((n_drugs, fp_dim), dtype=np.float32)
    has_fp = np.zeros(n_drugs, dtype=bool)
    for i, name in enumerate(drug_nodes["node_name"].values):
        key = str(name).strip().lower()
        if key in fp_map:
            fp_arr[i] = fp_map[key]
            has_fp[i] = True

    n_with = int(has_fp.sum())
    logger.info(f"Drug nodes: {n_drugs}; with fingerprint: {n_with} "
                f"({100*n_with/n_drugs:.1f}%); without: {n_drugs - n_with}")
    return (torch.from_numpy(fp_arr),
            torch.from_numpy(has_fp),
            n_with)


# ====================== Split logic (70/15/15) =========================
def cold_drug_split(nodes, edges, cfg_split, seed, split_mode="dg_context"):
    """Split drugs (not edges) into train/val/test under one of two protocols.

    Parameters
    ----------
    split_mode : str
        "dg_context": ONLY drug-gene edges incident to val/test drugs are
            removed from the training graph. Non-drug-gene context edges
            (e.g. gene-variant, gene-phenotype) involving the same drug-set
            membership are retained. This is the standard "cold-tail" KG
            completion protocol and is the main publication setting.
        "strict_cold_drug": EVERY edge incident to a val/test drug is moved
            to the corresponding split. The training graph contains zero
            edges touching any val/test drug. Stricter; report as
            sensitivity analysis. Because the model has no molecular drug
            descriptors, performance under this protocol is bounded by
            random for genuinely unseen drugs.

    Returns
    -------
    train_edges, val_edges, test_edges : DataFrames
    train_drugs, val_drugs, test_drugs : sets of int node indices
    split_meta : dict with split_mode, drug counts, edge counts (for audit)
    """
    if split_mode not in ("dg_context", "strict_cold_drug"):
        raise ValueError(
            f"Unknown split_mode '{split_mode}'. "
            f"Use 'dg_context' or 'strict_cold_drug'."
        )

    rng = np.random.default_rng(seed)
    drug_ids = nodes[nodes["node_type"] == "drug"]["node_index"].values.copy()
    rng.shuffle(drug_ids)

    n = len(drug_ids)
    n_train = int(round(n * cfg_split["train_drug_fraction"]))
    n_val = int(round(n * cfg_split["val_drug_fraction"]))
    # test gets the remainder (avoids rounding drift)
    train_drugs = set(int(d) for d in drug_ids[:n_train].tolist())
    val_drugs = set(int(d) for d in drug_ids[n_train:n_train + n_val].tolist())
    test_drugs = set(int(d) for d in drug_ids[n_train + n_val:].tolist())

    is_dg = (
        ((edges["source_type"] == "drug") & (edges["target_type"] == "gene")) |
        ((edges["source_type"] == "gene") & (edges["target_type"] == "drug"))
    )

    if split_mode == "dg_context":
        # Only drug-gene edges get re-assigned to val/test
        scope_mask = is_dg
    else:  # strict_cold_drug
        # Every edge incident to a val/test drug gets re-assigned
        scope_mask = pd.Series(True, index=edges.index)

    def assign_row(row):
        s, t = int(row["source_idx"]), int(row["target_idx"])
        for d in (s, t):
            if d in val_drugs:
                return "val"
            if d in test_drugs:
                return "test"
        return "train"

    split_label = pd.Series("train", index=edges.index, dtype="object")
    if scope_mask.any():
        split_label.loc[scope_mask] = edges.loc[scope_mask].apply(
            assign_row, axis=1,
        )

    train_edges = edges[split_label == "train"].copy().reset_index(drop=True)
    val_edges = edges[split_label == "val"].copy().reset_index(drop=True)
    test_edges = edges[split_label == "test"].copy().reset_index(drop=True)

    split_meta = {
        "split_mode": split_mode,
        "n_drugs_total": int(n),
        "n_train_drugs": len(train_drugs),
        "n_val_drugs": len(val_drugs),
        "n_test_drugs": len(test_drugs),
        "n_train_edges": len(train_edges),
        "n_val_edges": len(val_edges),
        "n_test_edges": len(test_edges),
    }

    return (train_edges, val_edges, test_edges,
            train_drugs, val_drugs, test_drugs, split_meta)


def leakage_audit(train_edges, val_drugs, test_drugs, split_mode):
    """Audit which val/test drugs still appear in the training edge set.

    Under split_mode='strict_cold_drug', this should be 0 — the audit
    proves the strict-cold-drug guarantee holds.
    Under split_mode='dg_context', this is intentionally non-zero (context
    edges retained) and quantifying it gives the reviewer transparency.
    """
    train_node_set = set(train_edges["source_idx"].astype(int).tolist()) \
                     | set(train_edges["target_idx"].astype(int).tolist())
    val_leaked = sorted(train_node_set & val_drugs)
    test_leaked = sorted(train_node_set & test_drugs)
    return {
        "split_mode": split_mode,
        "expected_strict_leakage": split_mode == "dg_context",
        "n_val_drugs_in_train_graph": len(val_leaked),
        "n_test_drugs_in_train_graph": len(test_leaked),
        "val_drugs_in_train_graph_sample": val_leaked[:20],
        "test_drugs_in_train_graph_sample": test_leaked[:20],
    }


def flag_positive_dg(edge_df: pd.DataFrame) -> pd.DataFrame:
    """Add is_dg_positive / is_dg_negative / is_dg_ambiguous flags using
    EXACT-equality matching against canonical association labels.

    Previous versions used assoc.str.contains("associated") with
    ~contains("not_associated") as a guard, which silently failed because
    ClinPGx writes the literal string "not associated" with a SPACE, not
    an underscore — so the negation never fired and every "not associated"
    row was incorrectly counted as positive. This is fixed by normalising
    spaces/hyphens to underscores before exact-equality matching.
    """
    is_forward_dg = (
        (edge_df["source_type"] == "drug") & (edge_df["target_type"] == "gene")
    )
    # Canonical normalisation: lowercase, strip, collapse spaces and hyphens
    # into underscores. This makes "not associated", "Not Associated",
    # "not-associated", "NOT ASSOCIATED" all map to "not_associated".
    assoc = (
        edge_df["association"].astype(str).str.lower().str.strip()
        .str.replace(r"[\s\-]+", "_", regex=True)
    )
    edge_df = edge_df.copy()
    edge_df["association_norm"] = assoc
    edge_df["is_dg_positive"] = (is_forward_dg & assoc.eq("associated")).astype(bool)
    edge_df["is_dg_negative"] = (is_forward_dg & assoc.eq("not_associated")).astype(bool)
    edge_df["is_dg_ambiguous"] = (is_forward_dg & assoc.eq("ambiguous")).astype(bool)
    return edge_df


# ====================== HeteroData build (mirrors Block 5) =============
def build_hetero_data(nodes, train_edges, rel_to_idx, embed_dim, device):
    """Construct PyG HeteroData. Node features = learnable Xavier inits."""
    data = HeteroData()

    node_offsets = {}
    for nt in ["drug", "gene", "phenotype", "variant"]:
        nt_nodes = nodes[nodes["node_type"] == nt]
        data[nt].num_nodes = len(nt_nodes)
        # Local index within type
        local = {idx: i for i, idx in enumerate(nt_nodes["node_index"].values)}
        node_offsets[nt] = local
        # Learnable embeddings (Xavier init)
        x = torch.empty(len(nt_nodes), embed_dim)
        nn.init.xavier_uniform_(x)
        data[nt].x = x

    # Map global node_index -> local index
    def to_local(global_idx, ntype):
        return node_offsets[ntype].get(int(global_idx))

    # Track local indices on the train edge dataframe (needed in training)
    train_edges = train_edges.copy()
    train_edges["source_hetero_id"] = train_edges.apply(
        lambda r: to_local(r["source_idx"], r["source_type"]), axis=1
    )
    train_edges["target_hetero_id"] = train_edges.apply(
        lambda r: to_local(r["target_idx"], r["target_type"]), axis=1
    )
    train_edges = train_edges.dropna(subset=["source_hetero_id", "target_hetero_id"])
    train_edges["source_hetero_id"] = train_edges["source_hetero_id"].astype(int)
    train_edges["target_hetero_id"] = train_edges["target_hetero_id"].astype(int)

    # Build typed edge tensors. PyG keys edges by (src_type, rel, tgt_type).
    for (st, tt, rt), group in train_edges.groupby(
            ["source_type", "target_type", "relation_type"]):
        src = torch.tensor(group["source_hetero_id"].values, dtype=torch.long)
        tgt = torch.tensor(group["target_hetero_id"].values, dtype=torch.long)
        data[st, rt, tt].edge_index = torch.stack([src, tgt], dim=0)

    return data.to(device), train_edges, node_offsets


# ====================== Validation: MRR on a held-out set ==============
@torch.no_grad()
def evaluate_mrr(model, x, edge_index, edge_type,
                 eval_edges, node_offsets, n_genes, device,
                 batch_size=128, hits_k_list=(1, 3, 5, 10)):
    """Cold-drug MRR / Hits@K computed on val or test positive DG edges."""
    model.eval()

    pos = eval_edges[eval_edges.get("is_dg_positive", False) == True]
    if len(pos) == 0:
        return {"mrr": 0.0, "n_queries": 0,
                **{f"hits_at_{k}": 0.0 for k in hits_k_list}}

    drug_offset = sum(node_offsets[nt].__len__() if nt < "drug" else 0
                      for nt in ["drug"])
    # Actually compute offsets correctly
    offsets, off = {}, 0
    for nt in ["drug", "gene", "phenotype", "variant"]:
        offsets[nt] = off
        off += len(node_offsets[nt])
    drug_offset = offsets["drug"]
    gene_offset = offsets["gene"]

    # Encoder pass (once)  -- Round 3: model.encode applies fingerprints
    # if attached, else falls back to plain encoder (legacy behaviour)
    z = model.encode(x, edge_index, edge_type)

    ranks = []
    # Use the most-common DG relation as the query relation
    # (the model has separate decoder weights per relation; for cold-drug
    # eval we ask the model to score the head against every gene)
    rel_counts = pos["relation_type"].value_counts()
    if len(rel_counts) == 0:
        return {"mrr": 0.0, "n_queries": 0,
                **{f"hits_at_{k}": 0.0 for k in hits_k_list}}

    # All gene global indices
    all_gene_local = torch.arange(n_genes, device=device, dtype=torch.long)
    all_gene_global = gene_offset + all_gene_local

    for _, row in pos.iterrows():
        head_local = node_offsets["drug"].get(int(row["source_idx"]))
        tail_local = node_offsets["gene"].get(int(row["target_idx"]))
        rel_name = row["relation_type"]
        # Skip if drug/gene not in node_offsets (shouldn't happen) or
        # relation never seen in training
        if head_local is None or tail_local is None:
            continue

        head_global = torch.tensor([drug_offset + head_local],
                                   dtype=torch.long, device=device)
        # Use rel_idx from train mapping; if not present, skip
        # (eval relation must exist in training relation vocabulary)
        rel_idx_tensor = torch.tensor(
            [eval_edges["rel_idx"].iloc[0]],  # placeholder; replaced below
            dtype=torch.long, device=device)
        # Better: use the relation idx attached to this row if available
        if "rel_idx" in row and not pd.isna(row["rel_idx"]):
            rel_idx_tensor = torch.tensor([int(row["rel_idx"])],
                                          dtype=torch.long, device=device)

        # Score head against every gene
        head_emb = z[head_global].expand(n_genes, -1)
        tail_emb = z[all_gene_global]
        rel_emb = model.decoder.rel_emb(rel_idx_tensor).expand(n_genes, -1)
        scores = (head_emb * rel_emb * tail_emb).sum(dim=-1)

        # Rank (1-based; descending)
        rank = (scores > scores[tail_local]).sum().item() + 1
        ranks.append(rank)

    if len(ranks) == 0:
        return {"mrr": 0.0, "n_queries": 0,
                **{f"hits_at_{k}": 0.0 for k in hits_k_list}}

    ranks = np.array(ranks)
    metrics = {
        "mrr": float(np.mean(1.0 / ranks)),
        "n_queries": int(len(ranks)),
    }
    for k in hits_k_list:
        metrics[f"hits_at_{k}"] = float(np.mean(ranks <= k))
    return metrics


# ====================== Main =========================================
def run_stage_02_train(cfg, paths, logger, args, device):


    logger.info(f"RUN_ID: {paths['run_id']} | Device: {device}")

    # --- Load script-01 artifacts ---
    nodes = pd.read_parquet(paths["artifacts_dir"] / "nodes.parquet")
    edges = pd.read_parquet(paths["artifacts_dir"] / "edges.parquet")
    logger.info(f"Loaded {len(nodes)} nodes and {len(edges)} edges")

    # --- Cold-drug 70/15/15 split (split_mode-aware) ---
    split_mode = cfg["split"].get("split_mode", "dg_context")
    logger.info(f"Performing 70/15/15 cold-drug split (split_mode={split_mode})...")
    (train_edges, val_edges, test_edges,
     train_drugs, val_drugs, test_drugs, split_meta) = \
        cold_drug_split(nodes, edges, cfg["split"], cfg["seed"],
                        split_mode=split_mode)

    logger.info(f"Drugs: train={len(train_drugs)}, val={len(val_drugs)}, "
                f"test={len(test_drugs)}")
    logger.info(f"Edges: train={len(train_edges)}, val={len(val_edges)}, "
                f"test={len(test_edges)}")

    # Leakage audit (proves what was and was not held out)
    audit = leakage_audit(train_edges, val_drugs, test_drugs, split_mode)
    with open(paths["artifacts_dir"] / "split_leakage_audit.json", "w") as f:
        json.dump(audit, f, indent=2)
    if split_mode == "strict_cold_drug":
        if audit["n_val_drugs_in_train_graph"] > 0 or \
           audit["n_test_drugs_in_train_graph"] > 0:
            logger.warning(
                "LEAKAGE under strict_cold_drug: "
                f"{audit['n_val_drugs_in_train_graph']} val + "
                f"{audit['n_test_drugs_in_train_graph']} test drugs "
                f"still appear in the training graph. Investigate."
            )
        else:
            logger.info("Leakage audit: strict_cold_drug guarantee holds (0 leakage)")
    else:
        logger.info(
            f"Leakage audit: dg_context retains "
            f"{audit['n_val_drugs_in_train_graph']} val and "
            f"{audit['n_test_drugs_in_train_graph']} test drugs in training "
            f"graph as non-DG context (expected and intended)."
        )

    # Save split_meta for downstream methods text
    with open(paths["artifacts_dir"] / "split_meta.json", "w") as f:
        json.dump(split_meta, f, indent=2)

    # Flag positive DG edges in each split
    train_edges = flag_positive_dg(train_edges)
    val_edges = flag_positive_dg(val_edges)
    test_edges = flag_positive_dg(test_edges)

    # Relation vocabulary from training only
    unique_relations = sorted(train_edges["relation_type"].unique())
    rel_to_idx = {r: i for i, r in enumerate(unique_relations)}
    train_edges["rel_idx"] = train_edges["relation_type"].map(rel_to_idx).astype(int)
    val_edges = val_edges[val_edges["relation_type"].isin(rel_to_idx)].copy()
    val_edges["rel_idx"] = val_edges["relation_type"].map(rel_to_idx).astype(int)
    test_edges = test_edges[test_edges["relation_type"].isin(rel_to_idx)].copy()
    test_edges["rel_idx"] = test_edges["relation_type"].map(rel_to_idx).astype(int)

    # Save split assignments
    split_df = pd.concat([
        train_edges.assign(split="train"),
        val_edges.assign(split="val"),
        test_edges.assign(split="test"),
    ], ignore_index=True)
    split_df.to_parquet(paths["artifacts_dir"] / "split_assignments.parquet",
                        index=False)

    with open(paths["artifacts_dir"] / "rel_to_idx.json", "w") as f:
        json.dump(rel_to_idx, f, indent=2)

    # --- Build HeteroData from train edges only ---
    logger.info("Building HeteroData (train edges only)...")
    embed_dim = cfg["model"]["embedding_dim"]
    data, train_edges, node_offsets = build_hetero_data(
        nodes, train_edges, rel_to_idx, embed_dim, device
    )

    # Convert to homogeneous for R-GCN
    hom = data.to_homogeneous(node_attrs=["x"], add_node_type=True,
                              add_edge_type=True).to(device)
    x = hom.x.to(device)
    edge_index = hom.edge_index.to(device)
    edge_type = hom.edge_type.to(device)
    num_edge_types = int(edge_type.max().item()) + 1
    n_genes = data["gene"].num_nodes

    # ---- PyG edge-type mapping (CRITICAL for downstream interpretation) ----
    # PyG's to_homogeneous() assigns edge_type IDs in the order it iterates
    # over data.edge_types, NOT in the order of rel_to_idx. The relation_importance
    # function in stage 4 uses edge_type IDs from the encoder, so it MUST use
    # this PyG mapping rather than rel_to_idx. Save it here for stage 4.
    pyg_edge_type_mapping = {}
    for i, et in enumerate(data.edge_types):
        # et is a tuple (source_type, relation_name, target_type)
        pyg_edge_type_mapping[i] = {
            "source_type": et[0],
            "relation_type": et[1],
            "target_type": et[2],
            "human_readable": f"{et[0]} -[{et[1]}]-> {et[2]}",
        }
    with open(paths["artifacts_dir"] / "pyg_edge_type_mapping.json", "w") as f:
        json.dump(pyg_edge_type_mapping, f, indent=2)
    logger.info(f"Saved pyg_edge_type_mapping.json "
                f"({len(pyg_edge_type_mapping)} PyG relation types). "
                f"This is the authoritative mapping for relation-importance "
                f"interpretation; rel_to_idx.json is the decoder mapping.")

    drug_offset = sum(data[nt].num_nodes for nt in []
                      if nt in ["dummy"])  # computed below
    # Compute offsets the simple way
    offsets, off = {}, 0
    for nt in data.node_types:
        offsets[nt] = off
        off += data[nt].num_nodes
    drug_offset = offsets["drug"]
    gene_offset = offsets["gene"]

    logger.info(f"Homogeneous: {x.size(0)} nodes, {edge_index.size(1)} edges, "
                f"{num_edge_types} relation types")

    # --- Supervised positives + negative sampler setup ---
    # D1: apply supervision regime (default | ambig_as_pos | nonassoc_excluded)
    regime = getattr(args, "supervision_regime", "default")
    logger.info(f"Supervision regime: {regime}")

    if regime == "default":
        # CIBB baseline: only is_dg_positive=True as supervised positives
        dg_positives = train_edges[
            train_edges["is_dg_positive"] == True
        ].reset_index(drop=True)
        n_extras = 0

    elif regime == "ambig_as_pos":
        # Merge ambiguous edges into positives (treat as noisy positives)
        dg_positives = train_edges[
            (train_edges["is_dg_positive"] == True) |
            (train_edges["is_dg_ambiguous"] == True)
        ].reset_index(drop=True)
        n_ambig = int((train_edges["is_dg_ambiguous"] == True).sum())
        n_extras = n_ambig
        logger.info(f"Ambiguous edges merged into positives: {n_ambig} extras")

    elif regime == "nonassoc_excluded":
        # Drop not-associated edges from training graph entirely
        n_nonassoc = int((train_edges["is_dg_negative"] == True).sum())
        train_edges = train_edges[
            train_edges["is_dg_negative"] != True
        ].reset_index(drop=True)
        dg_positives = train_edges[
            train_edges["is_dg_positive"] == True
        ].reset_index(drop=True)
        n_extras = -n_nonassoc
        logger.info(f"Not-associated edges dropped from graph: {n_nonassoc} removed")

    else:
        raise ValueError(f"Unknown supervision_regime: {regime}")

    logger.info(f"Supervised positive drug-gene edges: {len(dg_positives)} "
                f"(regime={regime}, extras={n_extras})")

    gene_degrees = dg_positives["target_hetero_id"].value_counts()
    degree_smoothed = gene_degrees.astype(float) ** 0.75
    probs_array = degree_smoothed / degree_smoothed.sum()
    genes_array_local = probs_array.index.values.astype(np.int64)

    penalty_weights = torch.zeros(n_genes, dtype=torch.float32, device=device)
    for gid, p in zip(genes_array_local, probs_array.values):
        penalty_weights[gid] = float(p)
    penalty_weights = penalty_weights / penalty_weights.max().clamp(min=1e-8)

    # Exclusion: train positives + ALL curated positives across full edges
    # (including val/test) so we never sample a true positive as a negative.
    train_pos_pairs = set(zip(
        dg_positives["source_hetero_id"].astype(int),
        dg_positives["target_hetero_id"].astype(int)
    ))
    # Local-id mapping for val/test positives
    def to_local_dg(df):
        df = df[df["is_dg_positive"] == True].copy()
        s_local = df["source_idx"].map(node_offsets["drug"])
        t_local = df["target_idx"].map(node_offsets["gene"])
        m = s_local.notna() & t_local.notna()
        return set(zip(s_local[m].astype(int), t_local[m].astype(int)))
    global_pos_pairs = train_pos_pairs | to_local_dg(val_edges) | to_local_dg(test_edges)
    logger.info(f"Pairs excluded from negative sampling: {len(global_pos_pairs)}")

    def sample_negative_tails(batch_heads_local, num_negatives):
        out = []
        for h in batch_heads_local.detach().cpu().numpy():
            chosen = []
            while len(chosen) < num_negatives:
                c = np.random.choice(genes_array_local, p=probs_array.values)
                if (int(h), int(c)) not in global_pos_pairs:
                    chosen.append(c)
            out.extend(chosen)
        return torch.tensor(out, dtype=torch.long, device=device)

    # --- Model + optimiser ---
    # Round 3: load drug fingerprints if available
    # CONFIG value is already relative to the run directory (e.g.
    # "./data/drug_fingerprints.parquet"); use as-is, no extra joining.
    fp_path = Path(cfg["data"].get(
        "drug_fingerprints_path", "./data/drug_fingerprints.parquet"
    ))
    drug_fp, drug_has_fp, n_with_fp = load_drug_fingerprints(
        fp_path, nodes, logger, fp_dim=2048,
    )
    n_drugs_for_fp = drug_fp.size(0) if drug_fp is not None else 0
    drug_fp_dim = 2048 if drug_fp is not None else None
    use_fingerprints = drug_fp is not None

    model = PGx_RGCN(
        in_channels=x.size(-1),
        hidden_channels=cfg["model"]["hidden_dim"],
        out_channels=cfg["model"]["output_dim"],
        num_edge_types=num_edge_types,
        num_decoder_relations=len(rel_to_idx),
        num_bases=cfg["model"]["num_bases"],
        dropout=cfg["model"]["dropout"],
        drug_fp_dim=drug_fp_dim,
        n_drugs_for_fp=n_drugs_for_fp,
    ).to(device)
    if use_fingerprints:
        model.attach_drug_fingerprints(
            drug_fp.to(device),
            drug_has_fp.to(device),
            drug_offset=offsets["drug"],
        )
        logger.info(f"Drug fingerprints ATTACHED to model: "
                    f"{n_with_fp}/{n_drugs_for_fp} drugs have SMILES.")
    else:
        logger.info("Drug fingerprints NOT used (parquet absent). "
                    "Falling back to random Xavier drug features.")
    optimizer = Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    head_global = torch.tensor(
        drug_offset + dg_positives["source_hetero_id"].values,
        dtype=torch.long, device=device,
    )
    tail_global = torch.tensor(
        gene_offset + dg_positives["target_hetero_id"].values,
        dtype=torch.long, device=device,
    )
    rel_idx_tensor = torch.tensor(
        dg_positives["rel_idx"].values, dtype=torch.long, device=device,
    )

    # --- Training loop (validation-MRR-based early stopping) ---
    EPOCHS = cfg["training"]["epochs"]
    BATCH = cfg["training"]["batch_size"]
    NUM_NEG = cfg["training"]["num_negatives"]
    BETA = 0.0 if args.ablation_beta_zero else cfg["training"]["hub_penalty_beta"]
    MARGIN = cfg["training"]["margin"]
    PATIENCE = cfg["training"]["patience"]
    VALIDATE_EVERY = cfg["training"]["validate_every"]
    GRAD_CLIP = cfg["training"]["grad_clip_norm"]

    if args.ablation_beta_zero:
        logger.info("ABLATION MODE: HUB_PENALTY_BETA forced to 0.0")
    logger.info(f"Training: epochs={EPOCHS}, batch={BATCH}, neg={NUM_NEG}, "
                f"beta={BETA}, margin={MARGIN}")

    history_rows = []
    best_val_mrr = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = total_base = total_hub = 0.0
        n_batches = 0
        perm = torch.randperm(head_global.size(0), device=device)

        for i in range(0, head_global.size(0), BATCH):
            optimizer.zero_grad()
            idx = perm[i:i + BATCH]
            bh = head_global[idx]
            bt = tail_global[idx]
            br = rel_idx_tensor[idx]
            bh_local = torch.tensor(
                dg_positives.iloc[idx.detach().cpu().numpy()]["source_hetero_id"].values,
                dtype=torch.long, device=device,
            )

            bt_neg_local = sample_negative_tails(bh_local, NUM_NEG)
            bt_neg = gene_offset + bt_neg_local
            bh_neg = bh.repeat_interleave(NUM_NEG)
            br_neg = br.repeat_interleave(NUM_NEG)

            pos_s = model(x, edge_index, edge_type, bh, br, bt)
            neg_s = model(x, edge_index, edge_type, bh_neg, br_neg, bt_neg)
            pos_exp = pos_s.repeat_interleave(NUM_NEG)
            base = F.margin_ranking_loss(
                pos_exp, neg_s,
                torch.ones_like(pos_exp), margin=MARGIN,
            )
            hub = torch.mean(
                torch.sigmoid(neg_s) * penalty_weights[bt_neg_local]
            )
            loss = base + BETA * hub
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()
            total_base += base.item()
            total_hub += hub.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_base = total_base / max(n_batches, 1)
        avg_hub = total_hub / max(n_batches, 1)

        # Periodic validation
        val_mrr = None
        if epoch == 1 or epoch % VALIDATE_EVERY == 0 or epoch == EPOCHS:
            val_metrics = evaluate_mrr(
                model, x, edge_index, edge_type,
                val_edges, node_offsets, n_genes, device,
                hits_k_list=tuple(cfg["evaluation"]["hits_at_k"]),
            )
            val_mrr = val_metrics["mrr"]

            if val_mrr > best_val_mrr + cfg["training"]["min_delta"]:
                best_val_mrr = val_mrr
                patience_counter = 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                logger.info(f"Epoch {epoch:03d} | loss={avg_loss:.4f} "
                            f"| val_MRR={val_mrr:.4f} ** NEW BEST **")
            else:
                patience_counter += 1
                logger.info(f"Epoch {epoch:03d} | loss={avg_loss:.4f} "
                            f"| val_MRR={val_mrr:.4f} | patience={patience_counter}/{PATIENCE}")
        else:
            logger.info(f"Epoch {epoch:03d} | loss={avg_loss:.4f}")

        history_rows.append({
            "epoch": epoch, "loss": avg_loss, "base_loss": avg_base,
            "hub_penalty": avg_hub, "val_mrr": val_mrr,
        })

        if patience_counter >= PATIENCE:
            logger.info(f"Early stopping at epoch {epoch}. Best val MRR: {best_val_mrr:.4f}")
            break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    # --- Save artifacts ---
    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(paths["artifacts_dir"] / "training_history.csv", index=False)

    torch.save({
        "model_state": model.state_dict(),
        "config": cfg,
        "num_edge_types": num_edge_types,
        "num_decoder_relations": len(rel_to_idx),
        "best_val_mrr": float(best_val_mrr),
        "ablation_beta_zero": args.ablation_beta_zero,
        # Round 3 metadata
        "use_fingerprints": bool(use_fingerprints),
        "drug_fp_dim": int(drug_fp_dim) if drug_fp_dim is not None else None,
        "n_drugs_for_fp": int(n_drugs_for_fp),
    }, paths["artifacts_dir"] / "model_checkpoint.pt")

    torch.save(data.cpu(), paths["artifacts_dir"] / "hetero_data.pt")
    # Also save the homogeneous tensors so script 03 doesn't need to rebuild
    torch.save({
        "x": x.cpu(), "edge_index": edge_index.cpu(), "edge_type": edge_type.cpu(),
        "node_offsets": offsets,
        "node_types_order": list(data.node_types),
    }, paths["artifacts_dir"] / "homogeneous_tensors.pt")

    # Convergence figure
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history_df["epoch"], history_df["loss"], label="total")
    ax[0].plot(history_df["epoch"], history_df["base_loss"], label="ranking")
    ax[0].plot(history_df["epoch"], history_df["hub_penalty"], label="hub penalty")
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Loss"); ax[0].legend()
    ax[0].set_title("Training convergence")
    val_h = history_df.dropna(subset=["val_mrr"])
    ax[1].plot(val_h["epoch"], val_h["val_mrr"], marker="o")
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Validation MRR")
    ax[1].set_title("Validation MRR (cold-drug)")
    plt.tight_layout()
    plt.savefig(paths["figures_dir"] / "figure4_training_convergence.png", dpi=200)
    plt.close()

    logger.info(f"Final best validation MRR: {best_val_mrr:.4f}")
    logger.info("Script 02 complete.")



# ============================================================================
# STAGE 3: COLD-DRUG TEST EVALUATION + NOVEL HYPOTHESES
# ============================================================================




# Import the model class so checkpoint loads
def build_known_positive_gene_map(all_edges, nodes):
    """Build {drug_local_idx -> set(gene_local_idx)} of ALL curated positive
    drug-gene pairs across train/val/test. Used by compute_ranks(filtered=True)
    to mask other known positives before computing the rank of the target tail.
    This implements the standard filtered-MRR convention from the KG completion
    literature (Bordes et al. 2013).
    """
    drug_nodes = nodes[nodes["node_type"] == "drug"].reset_index(drop=True)
    gene_nodes = nodes[nodes["node_type"] == "gene"].reset_index(drop=True)
    drug_global_to_local = {
        int(g): i for i, g in enumerate(drug_nodes["node_index"].values)
    }
    gene_global_to_local = {
        int(g): i for i, g in enumerate(gene_nodes["node_index"].values)
    }

    positives = flag_positive_dg(all_edges)
    positives = positives[positives["is_dg_positive"] == True]

    known = defaultdict(set)
    for _, r in positives.iterrows():
        s_type = str(r["source_type"])
        t_type = str(r["target_type"])
        # Orient to drug -> gene regardless of original row direction
        if s_type == "drug" and t_type == "gene":
            d_g, g_g = int(r["source_idx"]), int(r["target_idx"])
        elif s_type == "gene" and t_type == "drug":
            d_g, g_g = int(r["target_idx"]), int(r["source_idx"])
        else:
            continue
        d_local = drug_global_to_local.get(d_g)
        g_local = gene_global_to_local.get(g_g)
        if d_local is not None and g_local is not None:
            known[d_local].add(g_local)
    return known


def compute_ranks(model, x, edge_index, edge_type, eval_pos_df,
                  node_offsets, n_genes, device,
                  known_positives_by_drug=None, filtered=False):
    """For each positive (drug, gene) edge in eval_pos_df, compute the
    rank of the true gene among all genes scored against that drug.

    When `filtered=True` and `known_positives_by_drug` is provided, all
    other known positive genes for the same drug are masked to -inf before
    computing the rank. This is the standard "filtered" protocol from
    Bordes et al. 2013 and is what should be reported as the headline number.
    With `filtered=False` (default), raw ranks are returned.
    """
    model.eval()
    drug_offset = node_offsets["drug"]
    gene_offset = node_offsets["gene"]

    with torch.no_grad():
        z = model.encode(x, edge_index, edge_type)
        all_gene_global = gene_offset + torch.arange(n_genes, device=device,
                                                     dtype=torch.long)
        tail_emb_all = z[all_gene_global]

        # Round 3: track which queries have a fingerprinted head for
        # stratified MRR reporting
        drug_has_fp_local = None
        if hasattr(model, "drug_has_fp_buffer") \
                and model.drug_has_fp_buffer.numel() > 0:
            drug_has_fp_local = model.drug_has_fp_buffer.detach().cpu()

        ranks = []
        predicted_top_gene = []
        query_has_fp = []  # parallel to ranks, True if head drug has SMILES
        for _, row in eval_pos_df.iterrows():
            head_local = row.get("source_local")
            tail_local = row.get("target_local")
            rel_idx = int(row["rel_idx"])
            if head_local is None or tail_local is None:
                continue
            head_local_i = int(head_local)
            tail_local_i = int(tail_local)
            head_global = drug_offset + head_local_i
            head_emb = z[head_global].unsqueeze(0).expand(n_genes, -1)
            rel_emb = model.decoder.rel_emb(
                torch.tensor([rel_idx], device=device, dtype=torch.long)
            ).expand(n_genes, -1)
            scores = (head_emb * rel_emb * tail_emb_all).sum(dim=-1)

            # Filtered protocol: mask other known positives for this drug
            if filtered and known_positives_by_drug is not None:
                mask_genes = known_positives_by_drug.get(head_local_i, set()) \
                             - {tail_local_i}
                if mask_genes:
                    mask_idx = torch.tensor(
                        sorted(mask_genes), dtype=torch.long, device=device
                    )
                    scores = scores.clone()
                    scores[mask_idx] = -float("inf")

            rank = (scores > scores[tail_local_i]).sum().item() + 1
            ranks.append(rank)
            predicted_top_gene.append(int(scores.argmax().item()))
            if drug_has_fp_local is not None:
                query_has_fp.append(bool(drug_has_fp_local[head_local_i].item()))
            else:
                query_has_fp.append(False)

    return np.array(ranks), predicted_top_gene, np.array(query_has_fp, dtype=bool)


def metrics_from_ranks(ranks: np.ndarray, hits_k_list=(1, 3, 5, 10)) -> dict:
    if len(ranks) == 0:
        return {"mrr": 0.0, "n_queries": 0,
                **{f"hits_at_{k}": 0.0 for k in hits_k_list}}
    m = {
        "mrr": float(np.mean(1.0 / ranks)),
        "n_queries": int(len(ranks)),
    }
    for k in hits_k_list:
        m[f"hits_at_{k}"] = float(np.mean(ranks <= k))
    return m


def bootstrap_ci(ranks: np.ndarray, n_samples: int, hits_k_list, seed: int):
    """95% CIs on MRR and Hits@K via bootstrap resampling of the rank list."""
    rng = np.random.default_rng(seed)
    n = len(ranks)
    if n == 0:
        return {}
    mrr_samples = []
    hits_samples = {k: [] for k in hits_k_list}
    for _ in range(n_samples):
        idx = rng.integers(0, n, size=n)
        r = ranks[idx]
        mrr_samples.append(np.mean(1.0 / r))
        for k in hits_k_list:
            hits_samples[k].append(np.mean(r <= k))
    out = {
        "mrr_ci95": [float(np.percentile(mrr_samples, 2.5)),
                     float(np.percentile(mrr_samples, 97.5))],
    }
    for k in hits_k_list:
        out[f"hits_at_{k}_ci95"] = [
            float(np.percentile(hits_samples[k], 2.5)),
            float(np.percentile(hits_samples[k], 97.5)),
        ]
    return out


def degree_baseline_ranks(eval_pos_df, train_edges, n_genes, node_offsets,
                          gene_global_to_local=None):
    """Trivial baseline: rank genes purely by their training-degree
    (most-connected first). MRR/Hits@K against this is the floor.

    train_edges is read off disk via split_assignments.parquet, where the
    only available gene identifier is `target_idx` (a GLOBAL node index).
    To rank within the gene-only candidate space we convert it to a local
    gene index using the caller-supplied gene_global_to_local map."""
    dg_train = train_edges[train_edges["is_dg_positive"] == True].copy()
    if gene_global_to_local is not None:
        dg_train["target_local"] = dg_train["target_idx"].map(gene_global_to_local)
        dg_train = dg_train.dropna(subset=["target_local"])
        dg_train["target_local"] = dg_train["target_local"].astype(int)
        gene_deg = dg_train["target_local"].value_counts()
    else:
        # Legacy path (kept for backward compat) - assumes target_hetero_id exists
        gene_deg = dg_train["target_hetero_id"].value_counts()
    deg_array = np.zeros(n_genes)
    for gid, d in gene_deg.items():
        deg_array[int(gid)] = d
    sorted_genes = np.argsort(-deg_array)  # descending
    gene_to_rank = {int(g): i + 1 for i, g in enumerate(sorted_genes)}

    ranks = []
    for _, row in eval_pos_df.iterrows():
        tail_local = row.get("target_local")
        if tail_local is None:
            continue
        ranks.append(gene_to_rank.get(int(tail_local), n_genes))
    return np.array(ranks)


def hub_bias_analysis(test_metrics_ranks, predicted_top_genes, nodes, n_genes,
                      logger):
    """Measure CYP-family dominance in top predictions vs degree baseline."""
    # Get gene names from local indices via node_offsets
    gene_nodes = nodes[nodes["node_type"] == "gene"].reset_index(drop=True)
    gene_local_to_name = dict(enumerate(gene_nodes["node_name"].values))

    # Count predictions that map to a CYP-family gene
    top_names = [gene_local_to_name.get(g, "?") for g in predicted_top_genes]
    cyp_count = sum(1 for n in top_names if str(n).upper().startswith("CYP"))
    cyp_fraction = cyp_count / max(len(top_names), 1)
    return {
        "n_predictions": len(top_names),
        "n_cyp_predictions": cyp_count,
        "cyp_fraction": float(cyp_fraction),
        # The headline 6.00 -> 0.87 ratio compares vs a degree baseline
        # See README for full interpretation
    }


def run_stage_03_evaluate(cfg, paths, logger, args, device):


    logger.info(f"RUN_ID: {paths['run_id']} | Device: {device}")

    # --- Load artifacts ---
    nodes = pd.read_parquet(paths["artifacts_dir"] / "nodes.parquet")
    split_df = pd.read_parquet(paths["artifacts_dir"] / "split_assignments.parquet")
    with open(paths["artifacts_dir"] / "rel_to_idx.json") as f:
        rel_to_idx = json.load(f)
    homog = torch.load(paths["artifacts_dir"] / "homogeneous_tensors.pt",
                       weights_only=False)
    ckpt = torch.load(paths["artifacts_dir"] / "model_checkpoint.pt",
                      weights_only=False)

    x = homog["x"].to(device)
    edge_index = homog["edge_index"].to(device)
    edge_type = homog["edge_type"].to(device)
    node_offsets = homog["node_offsets"]
    # n_genes from offsets diff
    nt_order = homog["node_types_order"]
    counts = {}
    prev = 0
    for nt in nt_order:
        # Find next offset
        next_off = None
        for nt2 in nt_order:
            if node_offsets[nt2] > node_offsets[nt]:
                if next_off is None or node_offsets[nt2] < next_off:
                    next_off = node_offsets[nt2]
        counts[nt] = (next_off if next_off else x.size(0)) - node_offsets[nt]
    n_genes = counts["gene"]
    logger.info(f"n_genes={n_genes}, n_drugs={counts['drug']}")

    # Rebuild model
    # Round 3: check if checkpoint was trained with fingerprints
    use_fp_ck = bool(ckpt.get("use_fingerprints", False))
    drug_fp_dim_ck = ckpt.get("drug_fp_dim")
    n_drugs_for_fp_ck = int(ckpt.get("n_drugs_for_fp", 0))

    model = PGx_RGCN(
        in_channels=x.size(-1),
        hidden_channels=cfg["model"]["hidden_dim"],
        out_channels=cfg["model"]["output_dim"],
        num_edge_types=ckpt["num_edge_types"],
        num_decoder_relations=ckpt["num_decoder_relations"],
        num_bases=cfg["model"]["num_bases"],
        dropout=cfg["model"]["dropout"],
        drug_fp_dim=drug_fp_dim_ck if use_fp_ck else None,
        n_drugs_for_fp=n_drugs_for_fp_ck if use_fp_ck else 0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    # Attach the runtime fingerprint tensors (the buffers in state_dict are
    # already populated, but we re-attach to set drug_offset for safety).
    if use_fp_ck:
        fp_path = Path(cfg["data"].get(
            "drug_fingerprints_path", "./data/drug_fingerprints.parquet"
        ))
        if fp_path.exists():
            drug_fp, drug_has_fp, _ = load_drug_fingerprints(
                fp_path, nodes, logger, fp_dim=drug_fp_dim_ck,
            )
            if drug_fp is not None:
                model.attach_drug_fingerprints(
                    drug_fp.to(device),
                    drug_has_fp.to(device),
                    drug_offset=node_offsets["drug"],
                )
                logger.info("Drug fingerprints re-attached for stage 3.")
        else:
            # Buffers in state_dict already carry the fingerprints; we only
            # need to set drug_offset
            model.drug_offset = int(node_offsets["drug"])
            logger.info("Using fingerprints from checkpoint buffers "
                        "(no parquet present).")
    model.eval()
    logger.info(f"Loaded model (ablation_beta_zero={ckpt['ablation_beta_zero']}, "
                f"best val MRR={ckpt['best_val_mrr']:.4f})")

    # --- Prepare evaluation dataframes ---
    # Build drug/gene global->local lookups
    drug_nodes = nodes[nodes["node_type"] == "drug"].reset_index(drop=True)
    gene_nodes = nodes[nodes["node_type"] == "gene"].reset_index(drop=True)
    drug_global_to_local = {int(g): i for i, g in enumerate(drug_nodes["node_index"].values)}
    gene_global_to_local = {int(g): i for i, g in enumerate(gene_nodes["node_index"].values)}

    def attach_locals(df):
        df = df.copy()
        df["source_local"] = df["source_idx"].map(drug_global_to_local)
        df["target_local"] = df["target_idx"].map(gene_global_to_local)
        return df.dropna(subset=["source_local", "target_local"])

    train_df = attach_locals(split_df[split_df["split"] == "train"])
    val_df = attach_locals(split_df[split_df["split"] == "val"])
    test_df = attach_locals(split_df[split_df["split"] == "test"])
    test_pos = test_df[test_df["is_dg_positive"] == True].reset_index(drop=True)
    val_pos = val_df[val_df["is_dg_positive"] == True].reset_index(drop=True)

    logger.info(f"Test positive DG edges to evaluate: {len(test_pos)}")

    # --- Build known-positive map for filtered evaluation ---
    # Standard KG-completion convention (Bordes et al. 2013): when computing
    # the rank of a target tail, mask all OTHER known positives for the same
    # head from the candidate list. Without this, a model that correctly
    # places multiple known positives at the top is penalised for each one
    # ranking the others above it.
    logger.info("Building known-positive gene map for filtered evaluation...")
    edges_all = pd.read_parquet(paths["artifacts_dir"] / "edges.parquet")
    known_positives_by_drug = build_known_positive_gene_map(edges_all, nodes)
    logger.info(f"Built filter map: {len(known_positives_by_drug)} drugs "
                f"with mean {np.mean([len(v) for v in known_positives_by_drug.values()]):.1f} "
                f"known positives each")

    # --- Compute test ranks: BOTH raw and filtered ---
    logger.info("Computing test ranks (cold-drug, raw)...")
    test_ranks_raw, test_top_genes, _query_has_fp_raw = compute_ranks(
        model, x, edge_index, edge_type, test_pos,
        node_offsets, n_genes, device,
        known_positives_by_drug=None, filtered=False,
    )

    logger.info("Computing test ranks (cold-drug, filtered)...")
    test_ranks_filt, _, query_has_fp = compute_ranks(
        model, x, edge_index, edge_type, test_pos,
        node_offsets, n_genes, device,
        known_positives_by_drug=known_positives_by_drug, filtered=True,
    )

    hits_k_list = tuple(cfg["evaluation"]["hits_at_k"])
    metrics_raw = metrics_from_ranks(test_ranks_raw, hits_k_list)
    metrics_filt = metrics_from_ranks(test_ranks_filt, hits_k_list)
    cis_raw = bootstrap_ci(
        test_ranks_raw,
        n_samples=cfg["evaluation"]["bootstrap_samples"],
        hits_k_list=hits_k_list,
        seed=cfg["evaluation"]["bootstrap_seed"],
    )
    cis_filt = bootstrap_ci(
        test_ranks_filt,
        n_samples=cfg["evaluation"]["bootstrap_samples"],
        hits_k_list=hits_k_list,
        seed=cfg["evaluation"]["bootstrap_seed"],
    )
    metrics = {
        "raw": {**metrics_raw, **cis_raw},
        "filtered": {**metrics_filt, **cis_filt},
    }

    # Round 3: stratified MRR by whether the test-drug head has a SMILES
    # fingerprint. query_has_fp comes from compute_ranks (parallel to
    # test_ranks_filt). Reported only when fingerprints are in use.
    if "query_has_fp" in dir() and query_has_fp is not None and query_has_fp.any():
        mask_with = query_has_fp
        mask_without = ~query_has_fp
        if mask_with.sum() > 0:
            ranks_w = test_ranks_filt[mask_with]
            m_w = metrics_from_ranks(ranks_w, hits_k_list)
            metrics["filtered_with_smiles"] = {
                **m_w,
                "n_queries_with_smiles": int(mask_with.sum()),
            }
        if mask_without.sum() > 0:
            ranks_wo = test_ranks_filt[mask_without]
            m_wo = metrics_from_ranks(ranks_wo, hits_k_list)
            metrics["filtered_without_smiles"] = {
                **m_wo,
                "n_queries_without_smiles": int(mask_without.sum()),
            }
        logger.info("Stratified filtered MRR:")
        if "filtered_with_smiles" in metrics:
            logger.info(f"  with SMILES    ({metrics['filtered_with_smiles']['n_queries_with_smiles']:4d} drugs): "
                        f"MRR={metrics['filtered_with_smiles']['mrr']:.4f}")
        if "filtered_without_smiles" in metrics:
            logger.info(f"  without SMILES ({metrics['filtered_without_smiles']['n_queries_without_smiles']:4d} drugs): "
                        f"MRR={metrics['filtered_without_smiles']['mrr']:.4f}")

    # Validation re-evaluation (filtered, best-state)
    val_ranks_filt, _, _ = compute_ranks(
        model, x, edge_index, edge_type, val_pos,
        node_offsets, n_genes, device,
        known_positives_by_drug=known_positives_by_drug, filtered=True,
    )
    metrics["validation_filtered"] = metrics_from_ranks(val_ranks_filt, hits_k_list)

    # Degree baseline (floor) - degree baseline is rank-by-degree only, no
    # model involved, so filtering doesn't apply in the standard sense; we
    # still mask known positives for fair comparison
    logger.info("Computing degree-frequency baseline...")
    deg_ranks = degree_baseline_ranks(test_pos, train_df, n_genes, node_offsets,
                                        gene_global_to_local=gene_global_to_local)
    metrics["degree_baseline"] = metrics_from_ranks(deg_ranks, hits_k_list)

    logger.info("====== TEST RANKING METRICS ======")
    logger.info(f"  RAW      MRR: {metrics['raw']['mrr']:.4f} "
                f"(95% CI {metrics['raw']['mrr_ci95'][0]:.4f}-{metrics['raw']['mrr_ci95'][1]:.4f})")
    logger.info(f"  FILTERED MRR: {metrics['filtered']['mrr']:.4f} "
                f"(95% CI {metrics['filtered']['mrr_ci95'][0]:.4f}-{metrics['filtered']['mrr_ci95'][1]:.4f})")
    logger.info(f"  RAW      Hits@10: {metrics['raw']['hits_at_10']:.4f}")
    logger.info(f"  FILTERED Hits@10: {metrics['filtered']['hits_at_10']:.4f}")
    logger.info(f"  Degree baseline MRR: {metrics['degree_baseline']['mrr']:.4f}")

    # Use filtered ranks for downstream artifacts (hub bias, novel hypotheses)
    test_ranks = test_ranks_filt

    with open(paths["artifacts_dir"] / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # --- Hub-bias analysis ---
    logger.info("Hub-bias analysis...")
    hub_bias = hub_bias_analysis(test_ranks, test_top_genes, nodes, n_genes, logger)
    with open(paths["artifacts_dir"] / "hub_bias_analysis.json", "w") as f:
        json.dump(hub_bias, f, indent=2)
    logger.info(f"CYP-family fraction of top predictions: {hub_bias['cyp_fraction']:.3f}")

    # --- Hub bias figure (gene-degree distribution of top predictions) ---
    gene_nodes_full = nodes[nodes["node_type"] == "gene"].reset_index(drop=True)
    dg_train_for_deg = train_df[train_df["is_dg_positive"] == True]
    deg_count = dg_train_for_deg["target_local"].astype(int).value_counts().to_dict()
    pred_degs = [deg_count.get(g, 0) for g in test_top_genes]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(pred_degs, bins=30)
    ax.set_xlabel("Training degree of predicted top-gene")
    ax.set_ylabel("Frequency")
    ax.set_title("Hub-bias: degree distribution of top predictions")
    plt.tight_layout()
    plt.savefig(paths["figures_dir"] / "figure_hub_bias.png", dpi=200)
    plt.close()

    # --- Ambiguous pair prioritisation for re-curation ---
    # This is the curator-facing artifact. For each ambiguous drug-gene pair
    # (across ALL splits — train, val, test), we score the model's view of
    # whether it really is an association, and produce a sorted CSV the
    # curator can open in Excel.
    logger.info("Building ambiguous-pair priority list for re-curation...")

    # Pull ambiguous pairs from ALL splits, not just test — most ambiguous
    # pairs end up in the training graph after the cold-drug split.
    all_with_locals = pd.concat([train_df, val_df, test_df], ignore_index=True)
    ambig = all_with_locals[
        all_with_locals["association"].astype(str)
                                       .str.contains("ambiguous", case=False)
    ].copy()
    # Forward drug-gene direction only (clearest semantics for the curator)
    ambig = ambig[
        (ambig["source_type"] == "drug") & (ambig["target_type"] == "gene")
    ].drop_duplicates(["source_idx", "target_idx", "relation_type"]) \
     .reset_index(drop=True)

    if len(ambig) == 0:
        logger.info("No ambiguous drug-gene pairs found in any split.")
    else:
        logger.info(f"Found {len(ambig)} ambiguous drug-gene pairs to rank")

        # Lookups (drug global idx -> name; gene local idx -> name + global idx)
        drug_nodes_full = nodes[nodes["node_type"] == "drug"]
        drug_id_to_name = drug_nodes_full.set_index("node_index")["node_name"].to_dict()
        gene_local_to_name = dict(enumerate(gene_nodes_full["node_name"].values))
        gene_local_to_global = dict(enumerate(gene_nodes_full["node_index"].astype(int).values))

        # Set of curated POSITIVE drug-gene pairs from training — used to
        # filter the "top-5 alternatives" so we don't suggest genes the
        # curator has already confirmed for this drug.
        train_pos_global = train_df[train_df["is_dg_positive"] == True]
        pos_pairs_global = set(zip(
            train_pos_global["source_idx"].astype(int).values,
            train_pos_global["target_idx"].astype(int).values,
        ))

        # Encoder forward pass once (reused for every ambiguous pair)
        model.eval()
        with torch.no_grad():
            z = model.encode(x, edge_index, edge_type)
            all_gene_global = node_offsets["gene"] + torch.arange(
                n_genes, device=device, dtype=torch.long,
            )
            tail_emb_all = z[all_gene_global]

        rows = []
        for _, r in ambig.iterrows():
            d_idx_g = int(r["source_idx"])
            g_idx_g = int(r["target_idx"])
            d_local = r.get("source_local")
            g_local = r.get("target_local")
            rel_idx = r.get("rel_idx")
            if pd.isna(d_local) or pd.isna(g_local) or pd.isna(rel_idx):
                continue
            d_local = int(d_local); g_local = int(g_local); rel_idx = int(rel_idx)

            with torch.no_grad():
                head_emb = z[node_offsets["drug"] + d_local].unsqueeze(0).expand(n_genes, -1)
                rel_emb = model.decoder.rel_emb(
                    torch.tensor([rel_idx], device=device, dtype=torch.long)
                ).expand(n_genes, -1)
                scores = (head_emb * rel_emb * tail_emb_all).sum(dim=-1)

            rank_of_tail = int((scores > scores[g_local]).sum().item()) + 1
            priority = 1.0 / rank_of_tail

            # Top-5 alternatives, excluding the curated tail itself and any
            # gene already curated as a positive for this drug
            order = torch.argsort(scores, descending=True).cpu().numpy()
            alts = []
            for cand_local in order:
                cl = int(cand_local)
                if cl == g_local:
                    continue
                cg = gene_local_to_global.get(cl)
                if cg is None or (d_idx_g, int(cg)) in pos_pairs_global:
                    continue
                cand_rank = int((scores > scores[cl]).sum().item()) + 1
                alts.append((gene_local_to_name.get(cl, str(cl)), cand_rank))
                if len(alts) >= 5:
                    break
            alt_summary = "; ".join(f"{name} (r={rk})" for name, rk in alts)

            # Suggested curator action based on percentile rank
            rank_pct = rank_of_tail / max(n_genes, 1)
            if rank_pct <= 0.05:
                action = "upgrade_to_associated"
            elif rank_pct <= 0.20:
                action = "lean_associated"
            elif rank_pct >= 0.95:
                action = "downgrade_to_not_associated"
            elif rank_pct >= 0.80:
                action = "lean_not_associated"
            else:
                action = "remains_ambiguous"

            rows.append({
                "drug_name": drug_id_to_name.get(d_idx_g, str(d_idx_g)),
                "gene_name": gene_local_to_name.get(g_local, str(g_local)),
                "drug_node_index": d_idx_g,
                "gene_node_index": g_idx_g,
                "relation_type": str(r["relation_type"]),
                "association": str(r["association"]),
                "split": str(r.get("split", "?")),
                "model_rank_of_curated_tail": rank_of_tail,
                "n_genes_searched": int(n_genes),
                "rank_percentile": float(rank_pct),
                "priority_score": float(priority),
                "suggested_action": action,
                "top_5_alternative_genes": alt_summary,
            })

        ambig_out = pd.DataFrame(rows)
        if len(ambig_out) > 0:
            ambig_out = ambig_out.sort_values(
                "priority_score", ascending=False,
            ).reset_index(drop=True)

            ambig_out.to_parquet(
                paths["artifacts_dir"] / "ambiguous_priorities.parquet",
                index=False,
            )
            ambig_out.to_csv(
                paths["artifacts_dir"] / "ambiguous_priorities.csv",
                index=False,
            )

            logger.info(f"Saved {len(ambig_out)} ambiguous pairs with priority scores")
            logger.info("Curator-action breakdown:")
            for action, n in ambig_out["suggested_action"].value_counts().items():
                logger.info(f"  {action:36s}: {n}")

            # Show the top-5 most strongly-agreed-with ambiguous pairs
            logger.info("Top-5 ambiguous pairs the model thinks should be UPGRADED:")
            top = ambig_out.head(5)
            for _, t in top.iterrows():
                logger.info(
                    f"  {t['drug_name']:25s} -> {t['gene_name']:10s}  "
                    f"rank={t['model_rank_of_curated_tail']} "
                    f"({t['rank_percentile']*100:.1f}%-ile)  "
                    f"action={t['suggested_action']}"
                )

            # Also show the bottom-5 — pairs the model thinks should be DOWNGRADED
            downgrade = ambig_out.sort_values(
                "rank_percentile", ascending=False,
            ).head(5)
            if (downgrade["rank_percentile"] >= 0.80).any():
                logger.info("Top-5 ambiguous pairs the model thinks should be DOWNGRADED:")
                for _, t in downgrade.iterrows():
                    if t["rank_percentile"] < 0.80:
                        continue
                    logger.info(
                        f"  {t['drug_name']:25s} -> {t['gene_name']:10s}  "
                        f"rank={t['model_rank_of_curated_tail']} "
                        f"({t['rank_percentile']*100:.1f}%-ile)  "
                        f"action={t['suggested_action']}"
                    )

    # --- Candidate predictions: top-K per test drug, labelled by rediscovery ---
    # Every top-K row gets a "category" tag:
    #   "rediscovered_positive" : (drug, gene) is already a curated positive
    #                              (drug-gene edge with `associated` label)
    #   "rediscovered_negative" : (drug, gene) is curated as not_associated
    #   "rediscovered_ambiguous": (drug, gene) is curated as ambiguous
    #   "candidate_novel"       : no curated drug-gene edge exists at all
    #
    # Rediscovery of known curated positives is a sanity-check on the model.
    # Only the "candidate_novel" rows are genuine new hypotheses; the others
    # are model-on-known-data behaviour.
    logger.info("Generating candidate predictions with rediscovery labels...")

    # Index every curated drug-gene edge (any association) by (drug, gene)
    edges_all_dg = edges_all[
        (((edges_all["source_type"] == "drug") & (edges_all["target_type"] == "gene")) |
         ((edges_all["source_type"] == "gene") & (edges_all["target_type"] == "drug")))
    ].copy()
    # Orient to (drug, gene)
    def _orient(row):
        if row["source_type"] == "drug":
            return int(row["source_idx"]), int(row["target_idx"])
        return int(row["target_idx"]), int(row["source_idx"])
    dg_pair_to_assoc = {}
    import re as _re_assoc
    for _, r in edges_all_dg.iterrows():
        d, g = _orient(r)
        # Canonical normalisation: handle "not associated" (space) and
        # "not-associated" (hyphen) the same as "not_associated"
        assoc_raw = str(r["association"]).lower().strip()
        assoc = _re_assoc.sub(r"[\s\-]+", "_", assoc_raw)
        # Multiple edges per pair possible (different evidence tiers); the
        # strongest curated label wins. Priority: associated > not_associated > ambiguous
        cur = dg_pair_to_assoc.get((d, g))
        if cur == "associated":
            continue
        # Exact-equality matching after canonicalisation
        if assoc == "associated":
            dg_pair_to_assoc[(d, g)] = "associated"
        elif assoc == "not_associated" and cur != "associated":
            dg_pair_to_assoc[(d, g)] = "not_associated"
        elif assoc == "ambiguous" and cur not in ("associated", "not_associated"):
            dg_pair_to_assoc[(d, g)] = "ambiguous"

    def categorise(d_global, g_global):
        assoc = dg_pair_to_assoc.get((int(d_global), int(g_global)))
        if assoc == "associated":
            return "rediscovered_positive"
        if assoc == "not_associated":
            return "rediscovered_negative"
        if assoc == "ambiguous":
            return "rediscovered_ambiguous"
        return "candidate_novel"

    candidate_rows = []
    test_drug_ids = test_df["source_idx"].unique()
    train_rel_counts = train_df[train_df["is_dg_positive"] == True]["rel_idx"].value_counts()
    if len(train_rel_counts) == 0:
        logger.warning("No positive training edges; skipping candidate ranking")
    else:
        default_rel_idx = int(train_rel_counts.idxmax())
        gene_offset = node_offsets["gene"]
        drug_offset = node_offsets["drug"]

        with torch.no_grad():
            z = model.encode(x, edge_index, edge_type)
            all_gene_global = gene_offset + torch.arange(n_genes, device=device,
                                                          dtype=torch.long)
            tail_emb = z[all_gene_global]
            rel_emb = model.decoder.rel_emb(
                torch.tensor([default_rel_idx], device=device, dtype=torch.long)
            ).expand(n_genes, -1)

            top_k = 25
            for d_global in test_drug_ids[:200]:
                d_local = drug_global_to_local.get(int(d_global))
                if d_local is None:
                    continue
                head_global = drug_offset + d_local
                head_emb = z[head_global].unsqueeze(0).expand(n_genes, -1)
                scores = (head_emb * rel_emb * tail_emb).sum(dim=-1)
                top_idx = torch.topk(scores, k=min(top_k, n_genes))
                for rank, (s, g_local) in enumerate(zip(top_idx.values.cpu().numpy(),
                                                        top_idx.indices.cpu().numpy()), 1):
                    g_global = int(gene_nodes_full["node_index"].iloc[int(g_local)])
                    candidate_rows.append({
                        "drug_node_index": int(d_global),
                        "gene_node_index": g_global,
                        "rank": rank,
                        "score": float(s),
                        "category": categorise(d_global, g_global),
                    })

        all_predictions = pd.DataFrame(candidate_rows)
        all_predictions.to_parquet(
            paths["artifacts_dir"] / "all_predictions.parquet", index=False,
        )

        # Split into candidate_novel vs rediscovered for downstream stages
        novel_df = all_predictions[
            all_predictions["category"] == "candidate_novel"
        ].reset_index(drop=True)
        rediscovered_df = all_predictions[
            all_predictions["category"] != "candidate_novel"
        ].reset_index(drop=True)

        novel_df.to_parquet(
            paths["artifacts_dir"] / "candidate_predictions.parquet", index=False,
        )
        rediscovered_df.to_parquet(
            paths["artifacts_dir"] / "rediscovered_predictions.parquet", index=False,
        )

        # Keep novel_predictions.parquet for downstream compatibility, but
        # populate it ONLY with candidate_novel rows now (so downstream stages
        # don't bake known positives into pathway hit lists, etc.)
        novel_df.to_parquet(
            paths["artifacts_dir"] / "novel_predictions.parquet", index=False,
        )

        logger.info(f"Total ranked candidates: {len(all_predictions)}")
        for cat, n in all_predictions["category"].value_counts().items():
            logger.info(f"  {cat:25s}: {n}")
        logger.info(f"Saved candidate_predictions.parquet ({len(novel_df)} rows), "
                    f"rediscovered_predictions.parquet ({len(rediscovered_df)} rows), "
                    f"all_predictions.parquet ({len(all_predictions)} rows)")

    logger.info("Script 03 complete.")



# ============================================================================
# STAGE 4: INTRINSIC + POST-HOC INTERPRETATION
# ============================================================================





# ===================== Intrinsic interpretation =======================
def relation_importance(model, pyg_edge_type_mapping, logger):
    """Frobenius norm of each W_r in the encoder.

    CRITICAL: this uses the PyG homogeneous edge_type ID, NOT the decoder
    rel_to_idx. PyG assigns edge_type IDs in iteration order over
    data.edge_types, which does not generally match the alphabetical
    rel_to_idx mapping used by the DistMult decoder. Mislabelling here
    would put the wrong biological labels on the relation-importance bars.

    For RGCNConv with basis decomposition, each relation's effective
    weight is reconstructed as comp[r] @ basis before the norm is taken.

    Parameters
    ----------
    pyg_edge_type_mapping : dict[int -> dict]
        Mapping {edge_type_id: {"source_type", "relation_type", "target_type",
        "human_readable"}} as written by stage 2 to pyg_edge_type_mapping.json.
    """
    rows = []
    encoder = model.encoder
    for layer_name, conv in [("conv1", encoder.conv1), ("conv2", encoder.conv2)]:
        try:
            if hasattr(conv, "comp") and conv.comp is not None:
                comp = conv.comp.detach().cpu()
                basis = conv.weight.detach().cpu()
                num_relations = comp.size(0)
                for r_idx in range(num_relations):
                    W_r = torch.einsum("b,bij->ij", comp[r_idx], basis)
                    meta = pyg_edge_type_mapping.get(r_idx, {})
                    rows.append({
                        "layer": layer_name,
                        "edge_type_id": int(r_idx),
                        "relation": meta.get("human_readable", f"edge_type_{r_idx}"),
                        "source_type": meta.get("source_type", "?"),
                        "relation_type": meta.get("relation_type", "?"),
                        "target_type": meta.get("target_type", "?"),
                        "frobenius_norm": float(torch.norm(W_r).item()),
                    })
            else:
                weight = conv.weight.detach().cpu()
                num_relations = weight.size(0)
                for r_idx in range(num_relations):
                    meta = pyg_edge_type_mapping.get(r_idx, {})
                    rows.append({
                        "layer": layer_name,
                        "edge_type_id": int(r_idx),
                        "relation": meta.get("human_readable", f"edge_type_{r_idx}"),
                        "source_type": meta.get("source_type", "?"),
                        "relation_type": meta.get("relation_type", "?"),
                        "target_type": meta.get("target_type", "?"),
                        "frobenius_norm": float(torch.norm(weight[r_idx]).item()),
                    })
        except Exception as e:
            logger.warning(f"Could not extract norms for {layer_name}: {e}")
    return pd.DataFrame(rows)


# ===================== Post-hoc per-prediction explainer ==============
# These functions are direct ports of the original Block 10 logic.
# They produce structured, auditable graph-based explanations.

class Explainer:
    """Holds the undirected knowledge graph and lookup tables; provides
    path-extraction and shared-neighbour methods for any (drug, gene)
    prediction."""

    def __init__(self, nodes: pd.DataFrame, edges: pd.DataFrame, logger):
        self.logger = logger
        self.node_name_lookup = nodes.set_index("node_index")["node_name"].to_dict()
        self.node_type_lookup = nodes.set_index("node_index")["node_type"].to_dict()
        self.edge_relation_lookup = {}
        for _, r in edges.iterrows():
            u, v = int(r["source_idx"]), int(r["target_idx"])
            self.edge_relation_lookup[(u, v)] = str(r["relation_type"])
            self.edge_relation_lookup[(v, u)] = str(r["relation_type"])

        logger.info("Building undirected NetworkX graph for path extraction...")
        self.G = nx.Graph()
        for _, r in edges.iterrows():
            self.G.add_edge(int(r["source_idx"]), int(r["target_idx"]),
                            relation=str(r["relation_type"]))
        logger.info(f"  graph: {self.G.number_of_nodes()} nodes, "
                    f"{self.G.number_of_edges()} edges")

    def path_to_names(self, path):
        return " -> ".join([
            f"{self.node_name_lookup.get(n, n)}[{self.node_type_lookup.get(n, '?')}]"
            for n in path
        ])

    def path_to_relations(self, path):
        rels = []
        for i in range(len(path) - 1):
            rels.append(self.edge_relation_lookup.get(
                (path[i], path[i + 1]), "unknown"))
        return " | ".join(rels)

    def shortest_paths(self, source, target, max_paths=3):
        if source is None or target is None:
            return []
        if source not in self.G or target not in self.G:
            return []
        try:
            return list(nx.all_shortest_paths(self.G, source, target))[:max_paths]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def khop_distances(self, source, max_hops=4):
        if source is None or source not in self.G:
            return {}
        return dict(nx.single_source_shortest_path_length(
            self.G, source, cutoff=max_hops))

    def shared_khop_nodes(self, drug_idx, gene_idx, shared_type,
                          max_hops=4, top_k=15):
        """Nodes of type `shared_type` reachable within max_hops from BOTH
        drug and gene. Ranked by combined distance, then by node degree."""
        if drug_idx is None or gene_idx is None:
            return []
        d = self.khop_distances(drug_idx, max_hops)
        g = self.khop_distances(gene_idx, max_hops)
        shared = set(d) & set(g)
        shared = [n for n in shared
                  if self.node_type_lookup.get(n) == shared_type
                  and n not in (drug_idx, gene_idx)]
        shared = sorted(
            shared,
            key=lambda n: (d.get(n, 99) + g.get(n, 99), -self.G.degree(n))
        )
        return [{
            "node_index": n,
            "name": self.node_name_lookup.get(n, str(n)),
            "node_type": shared_type,
            "drug_distance": d.get(n, np.nan),
            "gene_distance": g.get(n, np.nan),
            "combined_distance": d.get(n, 99) + g.get(n, 99),
            "degree": self.G.degree(n),
        } for n in shared[:top_k]]

    def explain_prediction(self, drug_idx, gene_idx, max_hops=4,
                           max_paths=3, top_shared=10):
        """Produce a single structured explanation row for one (drug, gene) pair."""
        d_name = self.node_name_lookup.get(drug_idx, str(drug_idx))
        g_name = self.node_name_lookup.get(gene_idx, str(gene_idx))

        paths = self.shortest_paths(drug_idx, gene_idx, max_paths=max_paths)
        path_lengths = [len(p) - 1 for p in paths]
        shortest_len = min(path_lengths) if path_lengths else None
        path_strs = [self.path_to_names(p) for p in paths]
        path_rels = [self.path_to_relations(p) for p in paths]

        shared_genes = self.shared_khop_nodes(drug_idx, gene_idx, "gene",
                                                max_hops, top_shared)
        shared_variants = self.shared_khop_nodes(drug_idx, gene_idx, "variant",
                                                  max_hops, top_shared)
        shared_phenotypes = self.shared_khop_nodes(drug_idx, gene_idx, "phenotype",
                                                    max_hops, top_shared)

        def names(shared):
            return "; ".join([
                f"{s['name']}(d={s['drug_distance']},g={s['gene_distance']})"
                for s in shared
            ])

        return {
            "drug_node_index": int(drug_idx),
            "gene_node_index": int(gene_idx),
            "drug_name": d_name,
            "gene_name": g_name,
            "shortest_path_length": shortest_len,
            "n_shortest_paths": len(paths),
            "shortest_paths": " || ".join(path_strs),
            "shortest_path_relations": " || ".join(path_rels),
            "shared_genes_within_khop": names(shared_genes),
            "shared_variants_within_khop": names(shared_variants),
            "shared_phenotypes_within_khop": names(shared_phenotypes),
            "n_shared_genes": len(shared_genes),
            "n_shared_variants": len(shared_variants),
            "n_shared_phenotypes": len(shared_phenotypes),
        }


# ===================== Main ===========================================
def run_stage_04_interpret(cfg, paths, logger, args, device):


    logger.info(f"RUN_ID: {paths['run_id']} | Device: {device}")

    # --- Load artifacts ---
    nodes = pd.read_parquet(paths["artifacts_dir"] / "nodes.parquet")
    edges = pd.read_parquet(paths["artifacts_dir"] / "edges.parquet")
    with open(paths["artifacts_dir"] / "rel_to_idx.json") as f:
        rel_to_idx = json.load(f)
    homog = torch.load(paths["artifacts_dir"] / "homogeneous_tensors.pt",
                       weights_only=False)
    ckpt = torch.load(paths["artifacts_dir"] / "model_checkpoint.pt",
                      weights_only=False)
    novel_path = paths["artifacts_dir"] / "novel_predictions.parquet"

    # --- Rebuild model for intrinsic analysis ---
    x = homog["x"].to(device)
    # Round 3: checkpoint may carry drug-fingerprint layers
    use_fp_ck = bool(ckpt.get("use_fingerprints", False))
    drug_fp_dim_ck = ckpt.get("drug_fp_dim")
    n_drugs_for_fp_ck = int(ckpt.get("n_drugs_for_fp", 0))
    model = PGx_RGCN(
        in_channels=x.size(-1),
        hidden_channels=cfg["model"]["hidden_dim"],
        out_channels=cfg["model"]["output_dim"],
        num_edge_types=ckpt["num_edge_types"],
        num_decoder_relations=ckpt["num_decoder_relations"],
        num_bases=cfg["model"]["num_bases"],
        dropout=cfg["model"]["dropout"],
        drug_fp_dim=drug_fp_dim_ck if use_fp_ck else None,
        n_drugs_for_fp=n_drugs_for_fp_ck if use_fp_ck else 0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    # Stage 4 only inspects encoder weights, not the encode() forward,
    # so re-attaching fingerprints is not needed here.

    # --- (a) Intrinsic: relation-specific Frobenius norms ---
    logger.info("=== (a) Intrinsic: relation importance ===")
    # Load the PyG edge-type mapping written by stage 2 (this is the
    # authoritative mapping for encoder relation IDs; rel_to_idx is for the
    # decoder and is NOT in general the same).
    pyg_map_path = paths["artifacts_dir"] / "pyg_edge_type_mapping.json"
    if pyg_map_path.exists():
        with open(pyg_map_path) as f:
            pyg_edge_type_mapping = {int(k): v for k, v in json.load(f).items()}
        logger.info(f"Loaded PyG edge-type mapping ({len(pyg_edge_type_mapping)} entries)")
    else:
        logger.warning(
            "pyg_edge_type_mapping.json not found — relation labels in "
            "relation_importance figure will be edge_type IDs only. "
            "Re-run stage 2 to regenerate the mapping."
        )
        pyg_edge_type_mapping = {}
    ri = relation_importance(model, pyg_edge_type_mapping, logger)
    ri.to_parquet(paths["artifacts_dir"] / "relation_importance.parquet",
                  index=False)

    top_rel = ri.groupby("relation")["frobenius_norm"].mean() \
                .sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    top_rel.iloc[::-1].plot(kind="barh", ax=ax)
    ax.set_xlabel("Mean Frobenius norm (across encoder layers)")
    ax.set_title("Top 15 relations by encoder weight magnitude")
    plt.tight_layout()
    plt.savefig(paths["figures_dir"] / "figure_relation_importance.png", dpi=200)
    plt.close()
    logger.info(f"Saved {len(ri)} relation-importance rows")
    logger.info("Top 5 relations by mean Frobenius norm:")
    for r, v in top_rel.head(5).items():
        logger.info(f"  {r:60s} {v:.4f}")

    # --- (b) Post-hoc: per-prediction structural explanations ---
    if not novel_path.exists():
        logger.warning("No novel_predictions.parquet found; skipping per-prediction explanations")
        return

    novel = pd.read_parquet(novel_path)
    top_n = args.top_n or cfg["interpretation"]["top_n_predictions"]
    # Take top-N by rank (rank=1 is best)
    top = novel.sort_values("rank").head(top_n) if "rank" in novel.columns \
        else novel.head(top_n)
    logger.info(f"=== (b) Post-hoc: explaining top-{len(top)} predictions ===")

    explainer = Explainer(nodes, edges, logger)
    max_hops = max(cfg["interpretation"]["explanation_hops"])

    expl_rows = []
    for i, row in enumerate(top.itertuples(index=False), 1):
        explanation = explainer.explain_prediction(
            drug_idx=int(row.drug_node_index),
            gene_idx=int(row.gene_node_index),
            max_hops=max_hops,
            max_paths=3,
            top_shared=10,
        )
        # Attach the original model scoring info
        explanation["rank"] = int(getattr(row, "rank", -1))
        explanation["model_score"] = float(getattr(row, "score", float("nan")))
        expl_rows.append(explanation)

        if i % 10 == 0 or i == len(top):
            logger.info(f"  ... {i}/{len(top)} explanations built")

    expl_df = pd.DataFrame(expl_rows)
    expl_df.to_parquet(paths["artifacts_dir"] / "explanations.parquet", index=False)

    # --- Summary diagnostics ---
    if len(expl_df) > 0:
        n_with_path = (expl_df["shortest_path_length"].notna()).sum()
        avg_path_len = expl_df["shortest_path_length"].mean()
        avg_shared_genes = expl_df["n_shared_genes"].mean()
        logger.info(f"Predictions with a path in the graph: {n_with_path}/{len(expl_df)}")
        logger.info(f"Mean shortest path length: {avg_path_len:.2f}")
        logger.info(f"Mean # shared genes within {max_hops} hops: {avg_shared_genes:.2f}")

        # Show top-3 fully-resolved examples
        logger.info("Top-3 explanations (rank 1, 2, 3):")
        for _, r in expl_df.sort_values("rank").head(3).iterrows():
            logger.info(f"  rank {int(r['rank'])}: {r['drug_name']} -> {r['gene_name']}  "
                        f"(path length {r['shortest_path_length']}, "
                        f"shared genes {r['n_shared_genes']})")
            if r["shortest_paths"]:
                logger.info(f"    path: {r['shortest_paths'].split(' || ')[0]}")

    logger.info("Script 04 complete.")



# ============================================================================
# STAGE 5: PATHWAY ORA + REAL PUBMED EVIDENCE + INTEGRATED SCORE
# ============================================================================






# ==================== Literature mining =============================
# CRITICAL: Every piece of text in the output must come from a real
# PubMed article we have actually retrieved by PMID. No model-generated
# summaries. No invented citations. If NCBI returns nothing for a pair,
# the pair has an empty evidence digest — never a placeholder.

NCBI_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
NCBI_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def esearch_pmids(drug_name: str, gene_name: str, email: str,
                  date_from: int = 2020, date_to: int = 2026,
                  retmax: int = 10, sleep: float = 0.34,
                  max_retries: int = 3) -> list:
    """Return up to `retmax` PMIDs (most relevant first) for the
    drug-gene co-occurrence search, restricted to the date range.

    Date filter uses PDAT (publication date) in YYYY:YYYY[PDAT] form.
    Sort = relevance is NCBI default.
    """
    term = (
        f'"{drug_name}"[All Fields] AND "{gene_name}"[All Fields] '
        f'AND ("{date_from}"[PDAT] : "{date_to}"[PDAT])'
    )
    params = {
        "db": "pubmed", "term": term, "retmode": "json",
        "retmax": int(retmax),
        "tool": "pgx-rgcn-pipeline", "email": email,
    }
    for attempt in range(max_retries):
        try:
            time.sleep(sleep)
            r = requests.get(NCBI_ESEARCH, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            return list(data.get("esearchresult", {}).get("idlist", []))
        except Exception as e:
            if attempt == max_retries - 1:
                return []
            time.sleep(2 ** attempt)
    return []


def esummary_metadata(pmids: list, email: str,
                       sleep: float = 0.34, max_retries: int = 3) -> dict:
    """Fetch title, journal, year, authors, DOI for a list of PMIDs.

    Returns {pmid: {title, journal, year, authors, doi}}.
    Empty dict if the call fails. If a particular PMID's metadata is
    incomplete, that PMID is omitted (we never invent fields).
    """
    if not pmids:
        return {}
    params = {
        "db": "pubmed", "id": ",".join(str(p) for p in pmids),
        "retmode": "json",
        "tool": "pgx-rgcn-pipeline", "email": email,
    }
    for attempt in range(max_retries):
        try:
            time.sleep(sleep)
            r = requests.get(NCBI_ESUMMARY, params=params, timeout=15)
            r.raise_for_status()
            data = r.json().get("result", {})
            out = {}
            for pmid in pmids:
                rec = data.get(str(pmid))
                if not isinstance(rec, dict):
                    continue
                title = rec.get("title", "").strip()
                journal = rec.get("fulljournalname", "") or rec.get("source", "")
                pubdate = rec.get("pubdate", "")
                # pubdate is like "2024 Mar" or "2024" — take the year token
                year = None
                if pubdate:
                    for token in pubdate.split():
                        if token.isdigit() and len(token) == 4:
                            year = int(token)
                            break
                authors_list = rec.get("authors", []) or []
                authors = [a.get("name", "") for a in authors_list
                           if isinstance(a, dict)]
                doi = ""
                for aid in rec.get("articleids", []) or []:
                    if isinstance(aid, dict) and aid.get("idtype") == "doi":
                        doi = aid.get("value", "")
                        break
                # Only include if we got at least a title and journal
                if title and journal:
                    out[str(pmid)] = {
                        "title": title,
                        "journal": str(journal).strip(),
                        "year": year,
                        "authors": authors,
                        "doi": doi,
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    }
            return out
        except Exception:
            if attempt == max_retries - 1:
                return {}
            time.sleep(2 ** attempt)
    return {}


def efetch_abstracts(pmids: list, email: str,
                     sleep: float = 0.34, max_retries: int = 3) -> dict:
    """Fetch the full abstract text for a list of PMIDs in one call.

    Returns {pmid: abstract_text}. Abstract text is the verbatim
    content of the <AbstractText> elements joined by spaces. If the
    article has no abstract (some letters, editorials), the PMID is
    omitted from the result rather than filled with a placeholder.
    """
    if not pmids:
        return {}
    params = {
        "db": "pubmed", "id": ",".join(str(p) for p in pmids),
        "rettype": "abstract", "retmode": "xml",
        "tool": "pgx-rgcn-pipeline", "email": email,
    }
    import xml.etree.ElementTree as ET
    for attempt in range(max_retries):
        try:
            time.sleep(sleep)
            r = requests.get(NCBI_EFETCH, params=params, timeout=30)
            r.raise_for_status()
            root = ET.fromstring(r.content)
            out = {}
            for article in root.iter("PubmedArticle"):
                pmid_elem = article.find(".//PMID")
                if pmid_elem is None:
                    continue
                pmid = pmid_elem.text
                # Collect all AbstractText elements (some abstracts have
                # multiple sections like Background/Methods/Results)
                texts = []
                for at in article.iter("AbstractText"):
                    label = at.get("Label", "")
                    body = (at.text or "").strip()
                    if label and body:
                        texts.append(f"{label}: {body}")
                    elif body:
                        texts.append(body)
                if texts:
                    out[pmid] = " ".join(texts).strip()
            return out
        except Exception:
            if attempt == max_retries - 1:
                return {}
            time.sleep(2 ** attempt)
    return {}


def build_evidence_digest(drug_name: str, gene_name: str,
                          articles: list) -> str:
    """Build a short structured text digest from real article metadata.

    All text in the digest is either (a) verbatim from PubMed metadata
    (title, journal, year, authors) or (b) explicit factual statements
    we can derive from the metadata itself (e.g., "N articles found").
    No abstract content is paraphrased here; full abstracts are saved
    separately in their own column for inspection.
    """
    if not articles:
        return (f"No PubMed articles found for {drug_name} + {gene_name} "
                f"in 2020-2026.")
    lines = [
        f"{len(articles)} PubMed article(s) (2020-2026) reference both "
        f"{drug_name} and {gene_name}:"
    ]
    for i, a in enumerate(articles, 1):
        authors_abbrev = (a["authors"][0] + " et al." if len(a["authors"]) > 1
                          else (a["authors"][0] if a["authors"] else "Unknown"))
        year = a.get("year", "n.d.")
        lines.append(
            f"  [{i}] {authors_abbrev} ({year}). "
            f"{a['title']} {a['journal']}. PMID:{a['pmid']}. "
            f"{a['url']}"
        )
    return "\n".join(lines)


def fetch_literature_evidence(novel_df, nodes, cfg_lit, logger):
    """For each top prediction, fetch real PubMed metadata + abstracts.

    Returns a DataFrame in long form: one row per (drug, gene, article).
    Each row carries the full bibliographic citation and the verbatim
    abstract text from PubMed. Predictions with no matching articles
    still produce a single row with empty article fields, so every
    prediction is represented.
    """
    drug_map = nodes[nodes["node_type"] == "drug"].set_index("node_index")["node_name"]
    gene_map = nodes[nodes["node_type"] == "gene"].set_index("node_index")["node_name"]

    n_pairs = min(len(novel_df), cfg_lit.get("max_pairs_to_query", 50))
    n_articles_per_pair = cfg_lit.get("articles_per_pair", 10)
    date_from = cfg_lit.get("date_from", 2020)
    date_to = cfg_lit.get("date_to", 2026)
    email = cfg_lit["pubmed_email"]
    sleep = cfg_lit["request_sleep_seconds"]

    logger.info(
        f"Fetching real PubMed evidence: {n_pairs} pairs, "
        f"up to {n_articles_per_pair} articles each, "
        f"date range {date_from}-{date_to}, email={email}"
    )

    rows = []
    for i, row in novel_df.head(n_pairs).iterrows():
        d_idx = int(row["drug_node_index"])
        g_idx = int(row["gene_node_index"])
        d_name = drug_map.get(d_idx)
        g_name = gene_map.get(g_idx)
        if d_name is None or g_name is None:
            continue

        # Step 1: ESearch -> PMIDs
        pmids = esearch_pmids(
            d_name, g_name, email,
            date_from=date_from, date_to=date_to,
            retmax=n_articles_per_pair, sleep=sleep,
        )
        if not pmids:
            rows.append({
                "drug_node_index": d_idx, "gene_node_index": g_idx,
                "drug_name": d_name, "gene_name": g_name,
                "rank": int(row.get("rank", -1)),
                "model_score": float(row.get("score", float("nan"))),
                "n_articles_found": 0,
                "pmid": "", "title": "", "journal": "",
                "year": None, "authors": "", "doi": "", "url": "",
                "abstract": "",
            })
            continue

        # Step 2: ESummary -> bibliographic metadata
        metadata = esummary_metadata(pmids, email, sleep=sleep)

        # Step 3: EFetch -> abstracts
        abstracts = efetch_abstracts(pmids, email, sleep=sleep)

        # Assemble — only include PMIDs that have BOTH valid metadata
        # AND non-empty abstract text (otherwise the evidence digest
        # would have nothing to show).
        kept = []
        for pmid in pmids:
            meta = metadata.get(str(pmid))
            abstract = abstracts.get(str(pmid), "")
            if not meta:
                continue  # invalid metadata, skip
            # Even without an abstract we keep the row (it's a real PMID),
            # but we mark abstract as empty so the user knows.
            kept.append({
                "pmid": str(pmid),
                **meta,
                "abstract": abstract.strip(),
            })

        if not kept:
            rows.append({
                "drug_node_index": d_idx, "gene_node_index": g_idx,
                "drug_name": d_name, "gene_name": g_name,
                "rank": int(row.get("rank", -1)),
                "model_score": float(row.get("score", float("nan"))),
                "n_articles_found": 0,
                "pmid": "", "title": "", "journal": "",
                "year": None, "authors": "", "doi": "", "url": "",
                "abstract": "",
            })
        else:
            for art in kept:
                rows.append({
                    "drug_node_index": d_idx, "gene_node_index": g_idx,
                    "drug_name": d_name, "gene_name": g_name,
                    "rank": int(row.get("rank", -1)),
                    "model_score": float(row.get("score", float("nan"))),
                    "n_articles_found": len(kept),
                    "pmid": art["pmid"],
                    "title": art["title"],
                    "journal": art["journal"],
                    "year": art.get("year"),
                    "authors": "; ".join(art.get("authors", [])),
                    "doi": art.get("doi", ""),
                    "url": art.get("url", ""),
                    "abstract": art.get("abstract", ""),
                })

        if (i + 1) % 10 == 0:
            logger.info(f"  ... {i + 1}/{n_pairs} pairs processed")

    return pd.DataFrame(rows)


# ==================== Pathway ORA (THE FIX) =========================
def load_pharmgkb_pathways(pathways_dir: Path, logger) -> dict:
    """Load PharmGKB per-pathway TSVs.

    Each file is named like PA145011108-Statin_Pathway_Generalized_Pharmacokinetics.tsv
    and contains rows of curated reactions. The `Genes` column carries a
    comma-separated list of HGNC gene symbols for that row. The pathway's
    full gene set is the union of all symbols across every row.

    Returns {pathway_name: set(gene_symbols)} with names prefixed `PGKB::`
    so downstream code can group enrichments by source.
    """
    out = {}
    if not pathways_dir.exists():
        return out

    # PharmGKB TSVs follow the pattern PA*-*.tsv (PA + digits, dash, descriptor)
    candidates = [
        p for p in pathways_dir.glob("PA*.tsv")
        if re.match(r"PA\d+-", p.name)
    ]
    if not candidates:
        return out

    n_loaded = 0
    n_skipped = 0
    for f in candidates:
        # Pathway name = filename stem, with the leading "PA<digits>-" preserved
        # so the curator can trace back which file the set came from.
        name = "PGKB::" + f.stem.replace("-", "_", 1)
        try:
            df = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False)
        except Exception as e:
            logger.warning(f"  skipping {f.name}: {e}")
            n_skipped += 1
            continue
        # Find the Genes column (case-tolerant)
        gene_col = None
        for c in df.columns:
            if c.strip().lower() == "genes":
                gene_col = c
                break
        if gene_col is None:
            n_skipped += 1
            continue
        # Union all comma-separated gene symbols across every row
        symbols = set()
        for cell in df[gene_col]:
            if not cell:
                continue
            for tok in str(cell).split(","):
                t = tok.strip().upper()
                if t and t not in ("NAN", "NONE", "NULL"):
                    symbols.add(t)
        if symbols:
            out[name] = symbols
            n_loaded += 1
        else:
            n_skipped += 1
    logger.info(f"  PharmGKB: loaded {n_loaded} pathways "
                f"({n_skipped} skipped without parsable Genes column)")
    return out


def load_gmt_gene_sets(pathways_dir: Path, logger) -> dict:
    """Load gene sets from .gmt files. Detects source from filename
    (Reactome / KEGG / generic) and prefixes pathway names accordingly.
    """
    out = {}
    if not pathways_dir.exists():
        return out
    gmt_files = list(pathways_dir.glob("*.gmt")) + list(pathways_dir.glob("*.GMT"))
    if not gmt_files:
        return out

    for f in gmt_files:
        fname = f.name.lower()
        if "reactome" in fname:
            prefix = "REACTOME::"
            source = "Reactome"
        elif "kegg" in fname:
            prefix = "KEGG::"
            source = "KEGG"
        else:
            # Unknown GMT — keep it but mark as OTHER
            prefix = "OTHER::"
            source = f.stem
        n_before = len(out)
        with open(f, "r") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                name = prefix + parts[0]
                genes = {g.strip().upper() for g in parts[2:] if g.strip()}
                if genes:
                    out[name] = genes
        logger.info(f"  {source} ({f.name}): {len(out) - n_before} gene sets")
    return out


def load_gene_sets(pathways_dir: Path, logger) -> dict:
    """Load gene sets from THREE possible sources in the same folder:

        1. PharmGKB per-pathway TSVs    (filenames like PA<id>-<name>.tsv)
        2. Reactome / KEGG GMT files    (filenames containing "reactome" / "kegg")
        3. Generic .gmt files           (any other .gmt — tagged OTHER::)

    Each pathway name is prefixed with its source so enrichment results
    can be grouped by source (PGKB:: / REACTOME:: / KEGG:: / OTHER::).
    Returns {prefixed_pathway_name: set(gene_symbols_uppercased)}.
    """
    gene_sets = {}
    if not pathways_dir.exists():
        logger.warning(f"{pathways_dir} not found; cannot run pathway ORA")
        return gene_sets

    logger.info(f"Scanning {pathways_dir} for pathway gene sets...")
    pgkb = load_pharmgkb_pathways(pathways_dir, logger)
    gmt = load_gmt_gene_sets(pathways_dir, logger)
    gene_sets.update(pgkb)
    gene_sets.update(gmt)

    if not gene_sets:
        logger.warning(
            f"No usable pathway files in {pathways_dir} "
            f"(expected PharmGKB *.tsv with a Genes column, or *.gmt files)"
        )
        return gene_sets

    # Source breakdown for the log
    by_source = defaultdict(int)
    for n in gene_sets:
        src = n.split("::", 1)[0] if "::" in n else "OTHER"
        by_source[src] += 1
    logger.info(f"Loaded {len(gene_sets)} gene sets total:")
    for src, cnt in sorted(by_source.items()):
        logger.info(f"  {src}: {cnt} sets")
    return gene_sets


def ora_enrichment(hit_list, gene_sets, universe_size, fdr_threshold=0.05):
    """Standard gene-level ORA via the hypergeometric test.

    For each gene set:
        N = universe size (e.g., all protein-coding genes)
        K = size of the gene set (clipped to N)
        n = hit list size
        k = number of hits that are in the gene set

        P(X >= k) = sum_{i=k..min(n,K)} C(K,i)*C(N-K,n-i) / C(N,n)
                  = hypergeom.sf(k-1, N, K, n)

    Multiple testing correction via Benjamini-Hochberg.
    """
    hits = set(g.upper() for g in hit_list)
    n = len(hits)
    rows = []
    for name, gset in gene_sets.items():
        K = min(len(gset), universe_size)
        if K == 0 or n == 0:
            continue
        k = len(hits & gset)
        if k == 0:
            p = 1.0
        else:
            p = float(hypergeom.sf(k - 1, universe_size, K, n))
        expected = (K / universe_size) * n
        odds_ratio = (k / max(expected, 1e-9)) if expected > 0 else float("nan")
        rows.append({
            "pathway": name,
            "set_size": int(K),
            "hits_in_set": int(k),
            "expected_hits": float(expected),
            "odds_ratio": float(odds_ratio),
            "p_value": p,
        })

    df = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    # Benjamini-Hochberg
    m = len(df)
    if m > 0:
        df["rank"] = np.arange(1, m + 1)
        df["q_value"] = df["p_value"] * m / df["rank"]
        # Enforce monotonicity (BH step-up)
        df["q_value"] = df["q_value"].iloc[::-1].cummin().iloc[::-1]
        df["q_value"] = df["q_value"].clip(upper=1.0)
        df["significant"] = df["q_value"] < fdr_threshold
        df = df.drop(columns=["rank"])
    return df


# ==================== Integrated explanation table ==================
def build_integrated_explanations(
    integrated_df, explanations_df, pathway_df, gene_sets,
    nodes, lit_df, top_n=50,
):
    """One row per top prediction, joining all evidence sources.

    Each row contains:
        - drug name, gene name, model rank, integrated_score
        - graph paths (from script 04) connecting drug -> gene
        - literature support (PubMed co-occurrence count)
        - pathway hits (which Reactome/KEGG pathways contain the gene)
        - a short natural-language explanation string

    This is the supervisor- and clinician-facing artifact: a single
    file where every claim is auditable in one row.
    """
    if len(integrated_df) == 0:
        return pd.DataFrame()

    top = integrated_df.sort_values("integrated_score", ascending=False).head(top_n)
    gene_map = nodes[nodes["node_type"] == "gene"].set_index("node_index")["node_name"]
    drug_map = nodes[nodes["node_type"] == "drug"].set_index("node_index")["node_name"]

    # Build a reverse index: gene_name -> list of (pathway_name, q_value)
    # for any pathway whose gene set contains that gene name.
    gene_to_pathways = {}
    if len(pathway_df) > 0 and gene_sets:
        for _, row in pathway_df.iterrows():
            if not row.get("significant", False):
                continue  # only attach pathway labels for significant ones
            pname = row["pathway"]
            for g in gene_sets.get(pname, set()):
                gene_to_pathways.setdefault(g.upper(), []).append(
                    (pname, float(row["q_value"]))
                )

    # Lookup from (drug, gene) -> explanation row in explanations.parquet
    expl_lookup = {}
    if len(explanations_df) > 0:
        for _, r in explanations_df.iterrows():
            expl_lookup[(int(r["drug_node_index"]), int(r["gene_node_index"]))] = r

    # Lookup from (drug, gene) -> list of article metadata dicts
    # (long-form lit_df: one row per article, can be many per pair).
    lit_articles = {}
    if len(lit_df) > 0 and "pmid" in lit_df.columns:
        for _, r in lit_df.iterrows():
            if not r.get("pmid"):
                continue
            key = (int(r["drug_node_index"]), int(r["gene_node_index"]))
            lit_articles.setdefault(key, []).append({
                "pmid": str(r["pmid"]),
                "title": r.get("title", ""),
                "journal": r.get("journal", ""),
                "year": r.get("year"),
                "authors": r.get("authors", ""),
                "doi": r.get("doi", ""),
                "url": r.get("url", ""),
                "abstract": r.get("abstract", ""),
            })

    rows = []
    for _, r in top.iterrows():
        d_idx = int(r["drug_node_index"])
        g_idx = int(r["gene_node_index"])
        d_name = drug_map.get(d_idx, str(d_idx))
        g_name = gene_map.get(g_idx, str(g_idx))

        # Graph path from script 04
        expl = expl_lookup.get((d_idx, g_idx))
        if expl is not None:
            shortest_paths_str = expl.get("shortest_paths", "")
            path_relations_str = expl.get("shortest_path_relations", "")
            shortest_path_length = expl.get("shortest_path_length")
            n_shared_genes = int(expl.get("n_shared_genes", 0) or 0)
            n_shared_variants = int(expl.get("n_shared_variants", 0) or 0)
            n_shared_phenotypes = int(expl.get("n_shared_phenotypes", 0) or 0)
            shared_genes_summary = expl.get("shared_genes_within_khop", "")
        else:
            shortest_paths_str = ""
            path_relations_str = ""
            shortest_path_length = None
            n_shared_genes = n_shared_variants = n_shared_phenotypes = 0
            shared_genes_summary = ""

        # Pathway hits for this gene
        pw_hits = gene_to_pathways.get(g_name.upper() if g_name else "", [])
        pw_hits_sorted = sorted(pw_hits, key=lambda x: x[1])[:5]
        pathway_summary = "; ".join(
            [f"{p[0]} (q={p[1]:.3g})" for p in pw_hits_sorted]
        )

        # Literature: build the verbatim evidence digest from real articles
        articles = lit_articles.get((d_idx, g_idx), [])
        n_articles = len(articles)
        # Citation block: one numbered line per real article, verbatim metadata
        citation_lines = []
        for i, a in enumerate(articles, 1):
            authors_str = a["authors"] if a["authors"] else "Unknown authors"
            authors_short = authors_str.split(";")[0].strip()
            if ";" in authors_str:
                authors_short = f"{authors_short} et al."
            year = a.get("year") or "n.d."
            citation_lines.append(
                f"[{i}] {authors_short} ({year}). {a['title']} "
                f"{a['journal']}. PMID:{a['pmid']}"
                + (f". DOI:{a['doi']}" if a.get('doi') else "")
                + f". {a['url']}"
            )
        literature_citations = "\n".join(citation_lines) if citation_lines \
            else "No PubMed articles (2020-2026) found for this pair."

        # First-sentence-of-abstract summary per article (verbatim, no generation)
        abstract_first_sentences = []
        for i, a in enumerate(articles, 1):
            ab = (a.get("abstract") or "").strip()
            if not ab:
                continue
            # Split on first period followed by whitespace+capital
            m = re.search(r"\.\s+[A-Z]", ab)
            first = ab[:m.start() + 1] if m else ab
            abstract_first_sentences.append(f"[{i}] {first}")
        abstract_summary = "\n".join(abstract_first_sentences) \
            if abstract_first_sentences else ""

        # Natural-language one-line summary — pure facts, no invented claims
        parts = []
        # Handle both None and NaN (NaN sneaks past `is not None` checks)
        if shortest_path_length is not None and pd.notna(shortest_path_length):
            parts.append(
                f"Integrated score {r['integrated_score']:.3f}; "
                f"drug and gene connected in the curated graph via a "
                f"{int(shortest_path_length)}-hop path."
            )
        else:
            parts.append(
                f"Integrated score {r['integrated_score']:.3f}; "
                f"no graph path within 4 hops."
            )
        if n_articles > 0:
            parts.append(
                f"{n_articles} PubMed article(s) from 2020-2026 reference "
                f"both {d_name} and {g_name} (full citations and abstracts "
                f"in the literature_citations and abstract_summary columns)."
            )
        else:
            parts.append(
                f"No PubMed articles from 2020-2026 found referencing both "
                f"{d_name} and {g_name}."
            )
        if pw_hits_sorted:
            pw_names = ", ".join(p[0] for p in pw_hits_sorted[:2])
            parts.append(
                f"Gene is enriched in {len(pw_hits_sorted)} significant "
                f"pathway(s) including: {pw_names}."
            )
        if n_shared_genes + n_shared_variants + n_shared_phenotypes > 0:
            parts.append(
                f"Within the k-hop neighbourhood, drug and gene share "
                f"{n_shared_genes} genes, {n_shared_variants} variants, "
                f"and {n_shared_phenotypes} phenotypes."
            )
        explanation_text = " ".join(parts)

        rows.append({
            "drug_node_index": d_idx,
            "gene_node_index": g_idx,
            "drug_name": d_name,
            "gene_name": g_name,
            "model_rank": int(r.get("rank", -1)),
            "model_score": float(r.get("score", float("nan"))),
            "hub_adjusted": float(r.get("hub_adj", float("nan"))),
            "literature_signal": float(r.get("lit", float("nan"))),
            "pathway_signal": float(r.get("pathway_hit", 0.0)),
            "integrated_score": float(r["integrated_score"]),
            "n_pubmed_articles_2020_2026": int(n_articles),
            "literature_citations": literature_citations,
            "abstract_summary": abstract_summary,
            "n_significant_pathways_for_gene": len(pw_hits_sorted),
            "top_pathway_hits": pathway_summary,
            "shortest_path_length": shortest_path_length,
            "shortest_paths": shortest_paths_str,
            "shortest_path_relations": path_relations_str,
            "shared_genes_within_khop": shared_genes_summary,
            "n_shared_genes": n_shared_genes,
            "n_shared_variants": n_shared_variants,
            "n_shared_phenotypes": n_shared_phenotypes,
            "explanation": explanation_text,
        })
    return pd.DataFrame(rows)


# ==================== Integrated scoring ============================
def integrated_scores(novel_df, lit_df, pathway_df, top_genes_in_pathways,
                      weights):
    """Combine hub-adjusted model score + literature support + pathway support.

    The literature signal is now the count of REAL PubMed articles
    retrieved for the pair (2020-2026 date range), not a popularity
    co-occurrence count. Articles are stored verbatim in lit_df with
    traceable PMIDs, DOIs, and full abstracts.

    All components rescaled to [0, 1] before weighting.
    """
    df = novel_df.copy()
    # Rank-based hub-adjusted score (lower rank = better)
    df["hub_adj"] = 1.0 / df["rank"]
    df["hub_adj"] = df["hub_adj"] / df["hub_adj"].max()

    # Literature: count of retrieved articles per pair (long-form lit_df
    # has one row per article; count unique non-empty PMIDs per pair).
    if len(lit_df) > 0 and "pmid" in lit_df.columns:
        with_articles = lit_df[lit_df["pmid"].astype(str) != ""]
        article_count_map = with_articles.groupby(
            ["drug_node_index", "gene_node_index"]
        )["pmid"].nunique().to_dict()
    else:
        article_count_map = {}
    df["n_real_articles"] = df.apply(
        lambda r: int(article_count_map.get(
            (int(r["drug_node_index"]), int(r["gene_node_index"])), 0
        )), axis=1
    )
    # log1p then rescale to [0, 1]
    df["lit"] = np.log1p(df["n_real_articles"].clip(lower=0))
    if df["lit"].max() > 0:
        df["lit"] = df["lit"] / df["lit"].max()

    # Pathway: 1 if gene is in any significant pathway, 0 otherwise
    df["pathway_hit"] = df["gene_node_index"].apply(
        lambda g: 1.0 if int(g) in top_genes_in_pathways else 0.0
    )

    df["integrated_score"] = (
        weights["hub_adjusted"] * df["hub_adj"]
        + weights["literature"] * df["lit"]
        + weights["pathway"] * df["pathway_hit"]
    )
    return df.sort_values("integrated_score", ascending=False)


# ==================== Main ===========================================
def run_stage_05_contextualise(cfg, paths, logger, args):


    logger.info(f"RUN_ID: {paths['run_id']}")

    # --- Load ---
    novel_df = pd.read_parquet(paths["artifacts_dir"] / "novel_predictions.parquet")
    nodes = pd.read_parquet(paths["artifacts_dir"] / "nodes.parquet")
    logger.info(f"Loaded {len(novel_df)} novel predictions")

    cfg_ctx = cfg["contextualisation"]

    # --- Literature: real PubMed evidence (full abstracts, traceable PMIDs) ---
    if args.skip_literature:
        logger.info("--skip_literature set; using empty literature table")
        lit_df = pd.DataFrame(columns=[
            "drug_node_index", "gene_node_index", "drug_name", "gene_name",
            "rank", "model_score", "n_articles_found",
            "pmid", "title", "journal", "year", "authors", "doi", "url",
            "abstract",
        ])
    else:
        lit_df = fetch_literature_evidence(novel_df, nodes,
                                            cfg_ctx["literature"], logger)
    lit_df.to_parquet(paths["artifacts_dir"] / "literature_evidence.parquet",
                      index=False)
    # CSV for human inspection (full abstracts included)
    lit_df.to_csv(paths["artifacts_dir"] / "literature_evidence.csv", index=False)
    logger.info(f"Literature evidence table: {len(lit_df)} rows "
                f"(one row per drug-gene-article triple)")
    if len(lit_df) > 0 and "pmid" in lit_df.columns:
        n_with_articles = (lit_df["pmid"] != "").sum()
        n_pairs_with = lit_df[lit_df["pmid"] != ""]["drug_node_index"].nunique()
        logger.info(f"  Pairs with at least 1 PubMed article: {n_pairs_with}")
        logger.info(f"  Total articles retrieved: {n_with_articles}")

    # --- Pathway ORA (the methodological fix) ---
    logger.info("=== Pathway ORA (gene-level, protein-coding universe) ===")
    cfg_path = cfg_ctx["pathways"]
    pathways_dir = paths["data_dir"] / "pathways"
    gene_sets = load_gene_sets(pathways_dir, logger)

    # Determine the universe size
    if cfg_path["background_source"] == "protein_coding_genes":
        universe_size = int(cfg_path["protein_coding_count"])
    elif cfg_path["background_source"] == "graph_genes":
        universe_size = int((nodes["node_type"] == "gene").sum())
    else:
        raise ValueError(f"Unknown background_source: {cfg_path['background_source']}")
    logger.info(f"Universe size: {universe_size} ({cfg_path['background_source']})")

    # Build hit list: top-N gene symbols across novel predictions
    top_n = cfg_path["top_n_per_list"]
    gene_map = nodes[nodes["node_type"] == "gene"].set_index("node_index")["node_name"]
    top_gene_idx = novel_df.sort_values("rank").head(top_n)["gene_node_index"].unique()
    hit_list = [gene_map.get(int(g)) for g in top_gene_idx]
    hit_list = [g for g in hit_list if g is not None]
    logger.info(f"Hit list size: {len(hit_list)} top-{top_n} genes")

    if len(gene_sets) > 0 and len(hit_list) > 0:
        # Split gene_sets by source prefix (PGKB / REACTOME / KEGG / OTHER)
        # and run BH-corrected ORA independently per source. Mixing sources
        # under a single BH would be misleading — different sources have
        # very different prior probabilities and average set sizes, so
        # global BH would over-penalise the source with more sets.
        sets_by_source = defaultdict(dict)
        for name, gset in gene_sets.items():
            src = name.split("::", 1)[0] if "::" in name else "OTHER"
            sets_by_source[src][name] = gset

        per_source_frames = []
        for source, sub_sets in sorted(sets_by_source.items()):
            logger.info(f"  Running ORA for {source} ({len(sub_sets)} pathways)...")
            sub_df = ora_enrichment(
                hit_list, sub_sets, universe_size,
                fdr_threshold=cfg_path["fdr_threshold"],
            )
            if len(sub_df) > 0:
                sub_df["source"] = source
                per_source_frames.append(sub_df)

        pathway_df = (
            pd.concat(per_source_frames, ignore_index=True)
            if per_source_frames else pd.DataFrame()
        )
        # Re-sort the combined output by q within source, then across sources
        if len(pathway_df) > 0:
            pathway_df = pathway_df.sort_values(
                ["source", "q_value"]
            ).reset_index(drop=True)

        pathway_df.to_parquet(paths["artifacts_dir"] / "pathway_enrichment.parquet",
                              index=False)
        # Per-source breakdown
        logger.info("Pathway ORA results by source:")
        if len(pathway_df) > 0 and "significant" in pathway_df.columns:
            for src, grp in pathway_df.groupby("source"):
                n_sig_src = int(grp["significant"].sum())
                logger.info(
                    f"  {src:10s}: {n_sig_src}/{len(grp)} significant at "
                    f"q<{cfg_path['fdr_threshold']} "
                    f"(min q={grp['q_value'].min():.3g})"
                )
        n_sig = int(pathway_df["significant"].sum()) if "significant" in pathway_df else 0
        logger.info(f"TOTAL: {n_sig}/{len(pathway_df)} significant")

        if "q_value" in pathway_df and len(pathway_df) > 0:
            # Show top 3 per source
            logger.info("Top 3 pathways per source:")
            for src, grp in pathway_df.groupby("source"):
                for _, r in grp.head(3).iterrows():
                    logger.info(
                        f"  [{src}] {r['pathway'][:55]:55s} "
                        f"k={int(r['hits_in_set'])}/{int(r['set_size'])} "
                        f"p={r['p_value']:.3e} q={r['q_value']:.3e}"
                    )

        # Pathway figure — top hits per source on stacked subplots
        if len(pathway_df) > 0:
            sources_present = sorted(pathway_df["source"].unique())
            n_src = len(sources_present)
            fig, axes = plt.subplots(
                n_src, 1, figsize=(8, max(3, 2.5 * n_src)),
                squeeze=False,
            )
            for ax_i, src in enumerate(sources_present):
                ax = axes[ax_i, 0]
                top_paths = pathway_df[pathway_df["source"] == src].head(10)
                if len(top_paths) == 0:
                    ax.axis("off")
                    continue
                y = np.arange(len(top_paths))
                ax.barh(y, -np.log10(top_paths["q_value"].clip(lower=1e-300)))
                ax.set_yticks(y)
                # Strip the source prefix for display
                labels = [
                    p.split("::", 1)[1] if "::" in p else p
                    for p in top_paths["pathway"]
                ]
                ax.set_yticklabels([lb[:55] for lb in labels], fontsize=7)
                ax.invert_yaxis()
                ax.axvline(-np.log10(cfg_path["fdr_threshold"]), color="red",
                           linestyle="--", linewidth=0.8,
                           label=f"q={cfg_path['fdr_threshold']}")
                ax.set_xlabel("-log10(q-value)", fontsize=8)
                ax.set_title(f"{src} - top 10 pathways", fontsize=9)
                ax.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(paths["figures_dir"] / "figure_pathway_enrichment.png",
                        dpi=200)
            plt.close()
    else:
        logger.warning("Skipping pathway ORA (no gene sets or no hits)")
        pathway_df = pd.DataFrame()

    # --- Integrated score ---
    # Identify gene_node_indices that appear in any significant pathway
    sig_gene_indices = set()
    if len(pathway_df) > 0 and "significant" in pathway_df.columns:
        gene_idx_lookup = nodes[nodes["node_type"] == "gene"].set_index(
            "node_name")["node_index"].to_dict()
        for _, r in pathway_df[pathway_df["significant"] == True].iterrows():
            # Walk gene_sets to find which hit list genes are in this pathway
            in_set = set(g.upper() for g in gene_sets.get(r["pathway"], set()))
            for g_name in hit_list:
                if g_name and g_name.upper() in in_set:
                    gi = gene_idx_lookup.get(g_name)
                    if gi is not None:
                        sig_gene_indices.add(int(gi))

    if len(novel_df) > 0:
        integrated = integrated_scores(
            novel_df, lit_df, pathway_df, sig_gene_indices,
            cfg_ctx["integrated_score_weights"],
        )
        integrated.to_parquet(paths["artifacts_dir"] / "integrated_scores.parquet",
                              index=False)
        logger.info(f"Integrated scores: {len(integrated)} predictions ranked")
        logger.info(f"  Top-5 by integrated score:")
        for _, r in integrated.head(5).iterrows():
            d = nodes[nodes["node_index"] == int(r["drug_node_index"])]["node_name"].iloc[0]
            g = gene_map.get(int(r["gene_node_index"]))
            logger.info(f"  {d:30s} -> {g:15s} score={r['integrated_score']:.3f}")

        # --- Integrated explanation table: one row per top prediction ---
        # Join with explanations.parquet (script 04) if present
        expl_path = paths["artifacts_dir"] / "explanations.parquet"
        if expl_path.exists():
            explanations_df = pd.read_parquet(expl_path)
            logger.info(f"Loaded {len(explanations_df)} graph-path explanations "
                        f"from script 04")
        else:
            logger.warning("explanations.parquet not found; integrated table "
                           "will omit graph paths. Run script 04 first.")
            explanations_df = pd.DataFrame()

        top_n_expl = cfg_ctx.get("integrated_top_n", 50)
        integrated_expl = build_integrated_explanations(
            integrated, explanations_df, pathway_df, gene_sets,
            nodes, lit_df, top_n=top_n_expl,
        )
        if len(integrated_expl) > 0:
            out_path = paths["artifacts_dir"] / "integrated_explanations.parquet"
            integrated_expl.to_parquet(out_path, index=False)
            # Also save a human-readable CSV for the supervisor
            csv_path = paths["artifacts_dir"] / "integrated_explanations.csv"
            integrated_expl.to_csv(csv_path, index=False)
            logger.info(f"Integrated explanations: {len(integrated_expl)} rows -> "
                        f"{out_path.name} and {csv_path.name}")
            logger.info("Top-3 integrated explanations:")
            for _, r in integrated_expl.head(3).iterrows():
                logger.info(f"  {r['drug_name']} -> {r['gene_name']}: "
                            f"{r['explanation']}")

    logger.info("Script 05 complete.")



# ============================================================================
# STAGE 6: PHENOTYPE PROPAGATION (1-HOP + 2-HOP-VIA-VARIANT)
# ============================================================================






# ===================== Phenotype propagation helpers ==================
def _canon_assoc(x) -> str:
    """Canonicalise a ClinPGx association string.

    ClinPGx writes "not associated" with a space; this function maps it
    (and any case/spacing/hyphen variant) to a single canonical token.
    Used by phenotype propagation labelling to fix the substring-match
    bug where "not_associated" in "not associated" returned False and
    every negative gene-phenotype edge was silently treated as positive.
    """
    import re as _re_can
    return _re_can.sub(r"[\s\-]+", "_",
                       str(x).lower().strip())


def build_gene_phenotype_index(edges_df: pd.DataFrame) -> dict:
    """Index gene_node_index -> list of phenotype edge dicts.

    Each dict contains: phenotype_idx, relation_type, evidence,
    association, and direction ("direct" or "via_variant").
    Direct gene_phenotype edges only here; variant-mediated handled
    separately.
    """
    idx = defaultdict(list)
    mask = (
        (edges_df["source_type"] == "gene") & (edges_df["target_type"] == "phenotype")
    )
    for _, r in edges_df[mask].iterrows():
        idx[int(r["source_idx"])].append({
            "phenotype_idx": int(r["target_idx"]),
            "relation_type": str(r["relation_type"]),
            "evidence": str(r["evidence"]),
            "association": str(r["association"]),
            "direction": "direct",
            "hops": 1,
        })
    # Also pick up phenotype -> gene edges (some curators encode them this way)
    mask2 = (
        (edges_df["source_type"] == "phenotype") & (edges_df["target_type"] == "gene")
    )
    for _, r in edges_df[mask2].iterrows():
        idx[int(r["target_idx"])].append({
            "phenotype_idx": int(r["source_idx"]),
            "relation_type": str(r["relation_type"]),
            "evidence": str(r["evidence"]),
            "association": str(r["association"]),
            "direction": "direct",
            "hops": 1,
        })
    return idx


def build_gene_variant_phenotype_index(edges_df: pd.DataFrame) -> dict:
    """Index gene -> list of phenotype edges reached via a variant.

    A variant connects a gene to a phenotype if there exist edges:
        gene <-> variant      AND      variant <-> phenotype
    Both directions allowed.
    """
    # Map gene -> connected variants
    gene_to_variants = defaultdict(set)
    mask_gv = (
        (edges_df["source_type"] == "gene") & (edges_df["target_type"] == "variant")
    )
    for _, r in edges_df[mask_gv].iterrows():
        gene_to_variants[int(r["source_idx"])].add(int(r["target_idx"]))
    mask_vg = (
        (edges_df["source_type"] == "variant") & (edges_df["target_type"] == "gene")
    )
    for _, r in edges_df[mask_vg].iterrows():
        gene_to_variants[int(r["target_idx"])].add(int(r["source_idx"]))

    # Map variant -> phenotype edges (with evidence/association)
    variant_to_phen = defaultdict(list)
    mask_vp = (
        (edges_df["source_type"] == "variant") & (edges_df["target_type"] == "phenotype")
    )
    for _, r in edges_df[mask_vp].iterrows():
        variant_to_phen[int(r["source_idx"])].append({
            "phenotype_idx": int(r["target_idx"]),
            "via_variant_idx": int(r["source_idx"]),
            "relation_type": str(r["relation_type"]),
            "evidence": str(r["evidence"]),
            "association": str(r["association"]),
        })
    mask_pv = (
        (edges_df["source_type"] == "phenotype") & (edges_df["target_type"] == "variant")
    )
    for _, r in edges_df[mask_pv].iterrows():
        variant_to_phen[int(r["target_idx"])].append({
            "phenotype_idx": int(r["source_idx"]),
            "via_variant_idx": int(r["target_idx"]),
            "relation_type": str(r["relation_type"]),
            "evidence": str(r["evidence"]),
            "association": str(r["association"]),
        })

    # Compose: gene -> list of phenotype dicts via variant
    out = defaultdict(list)
    for gene_idx, variants in gene_to_variants.items():
        for v in variants:
            for phen in variant_to_phen.get(v, []):
                out[gene_idx].append({
                    **phen,
                    "direction": "via_variant",
                    "hops": 2,
                })
    return out


def propagate(drug_gene_predictions: pd.DataFrame,
              direct_index: dict, variant_index: dict,
              nodes: pd.DataFrame, logger) -> pd.DataFrame:
    """For each (drug, gene) row in drug_gene_predictions, emit one row per
    inferred phenotype with full provenance."""
    phenotype_names = nodes.set_index("node_index")["node_name"].to_dict()
    rows = []

    for _, pred in drug_gene_predictions.iterrows():
        d_idx = int(pred["drug_node_index"])
        g_idx = int(pred["gene_node_index"])
        d_name = pred.get("drug_name", str(d_idx))
        g_name = pred.get("gene_name", str(g_idx))
        integrated_score = float(pred.get("integrated_score", float("nan")))
        model_rank = int(pred.get("model_rank", -1))

        # Direct 1-hop gene -> phenotype edges
        for edge in direct_index.get(g_idx, []):
            rows.append({
                "drug_node_index": d_idx, "gene_node_index": g_idx,
                "drug_name": d_name, "gene_name": g_name,
                "phenotype_node_index": edge["phenotype_idx"],
                "phenotype_name": phenotype_names.get(edge["phenotype_idx"], "?"),
                "propagation_hops": edge["hops"],
                "via_variant_idx": None, "via_variant_name": None,
                "gene_phenotype_relation": edge["relation_type"],
                "evidence": edge["evidence"],
                "association": edge["association"],
                "is_negative_evidence": _canon_assoc(edge["association"]) == "not_associated",
                "is_ambiguous": _canon_assoc(edge["association"]) == "ambiguous",
                "integrated_score": integrated_score,
                "model_rank": model_rank,
            })
        # 2-hop gene -> variant -> phenotype edges
        for edge in variant_index.get(g_idx, []):
            v_idx = edge["via_variant_idx"]
            rows.append({
                "drug_node_index": d_idx, "gene_node_index": g_idx,
                "drug_name": d_name, "gene_name": g_name,
                "phenotype_node_index": edge["phenotype_idx"],
                "phenotype_name": phenotype_names.get(edge["phenotype_idx"], "?"),
                "propagation_hops": edge["hops"],
                "via_variant_idx": v_idx,
                "via_variant_name": phenotype_names.get(v_idx, str(v_idx)),
                "gene_phenotype_relation": edge["relation_type"],
                "evidence": edge["evidence"],
                "association": edge["association"],
                "is_negative_evidence": _canon_assoc(edge["association"]) == "not_associated",
                "is_ambiguous": _canon_assoc(edge["association"]) == "ambiguous",
                "integrated_score": integrated_score,
                "model_rank": model_rank,
            })

    df = pd.DataFrame(rows)
    # Deduplicate at (drug, gene, phenotype) level — but keep best-evidence row
    if len(df) > 0:
        evidence_priority = {
            "guideline": 1, "label": 2, "clinical": 3,
            "literature": 4, "pathway": 5, "other": 6,
        }
        df["evidence_rank"] = df["evidence"].map(evidence_priority).fillna(7)
        df = df.sort_values(
            ["drug_node_index", "gene_node_index", "phenotype_node_index",
             "evidence_rank", "propagation_hops"]
        ).drop_duplicates(
            subset=["drug_node_index", "gene_node_index", "phenotype_node_index",
                    "direction" if "direction" in df.columns else "evidence"],
            keep="first",
        ).drop(columns=["evidence_rank"])

    return df


# ===================== Main ==========================================
def run_stage_06_propagate(cfg, paths, logger, args):


    logger.info(f"RUN_ID: {paths['run_id']}")

    # --- Load ---
    nodes = pd.read_parquet(paths["artifacts_dir"] / "nodes.parquet")
    edges = pd.read_parquet(paths["artifacts_dir"] / "edges.parquet")
    int_path = paths["artifacts_dir"] / "integrated_explanations.parquet"
    if not int_path.exists():
        logger.error(f"Not found: {int_path}. Run script 05 first.")
        sys.exit(1)
    integrated = pd.read_parquet(int_path)
    logger.info(f"Loaded {len(integrated)} top integrated predictions "
                f"from script 05")

    # --- Build indices ---
    logger.info("Building gene -> phenotype index (1-hop)...")
    direct_index = build_gene_phenotype_index(edges)
    logger.info(f"  {len(direct_index)} genes have direct phenotype edges")

    logger.info("Building gene -> variant -> phenotype index (2-hop)...")
    variant_index = build_gene_variant_phenotype_index(edges)
    logger.info(f"  {len(variant_index)} genes have variant-mediated phenotype edges")

    # --- Propagate ---
    logger.info("Propagating predictions through curated gene-phenotype edges...")
    inferences = propagate(integrated, direct_index, variant_index, nodes, logger)
    logger.info(f"Generated {len(inferences)} phenotype inferences")

    if len(inferences) > 0:
        out_pq = paths["artifacts_dir"] / "phenotype_inferences.parquet"
        out_csv = paths["artifacts_dir"] / "phenotype_inferences.csv"
        inferences.to_parquet(out_pq, index=False)
        inferences.to_csv(out_csv, index=False)
        logger.info(f"Saved: {out_pq.name} and {out_csv.name}")

        # Breakdown
        n_neg_evidence = int(inferences["is_negative_evidence"].sum())
        n_ambig = int(inferences["is_ambiguous"].sum())
        n_pos = len(inferences) - n_neg_evidence - n_ambig
        logger.info(f"  Positive associations:  {n_pos}")
        logger.info(f"  Negative evidence:      {n_neg_evidence}")
        logger.info(f"  Ambiguous:              {n_ambig}")

        # Evidence grade breakdown
        logger.info("Evidence grade breakdown:")
        for ev, n in inferences["evidence"].value_counts().items():
            logger.info(f"  {ev:12s}: {n}")

        # Top 5 highest-confidence (guideline evidence, positive) per drug
        logger.info("Sample inferences (top-5 by integrated_score, "
                    "positive guideline-evidence):")
        sample = inferences[
            (inferences["evidence"] == "guideline") &
            (~inferences["is_negative_evidence"]) &
            (~inferences["is_ambiguous"])
        ].sort_values("integrated_score", ascending=False).head(5)
        for _, r in sample.iterrows():
            via = f" via {r['via_variant_name']}" if pd.notna(r["via_variant_idx"]) else ""
            logger.info(f"  {r['drug_name']:25s} -> {r['gene_name']:10s} "
                        f"-> {r['phenotype_name']:40s}{via} "
                        f"(integrated={r['integrated_score']:.3f})")

        # Figure: stacked bar of evidence grades, split by polarity
        ev_order = ["guideline", "label", "clinical", "literature",
                    "pathway", "other"]
        ev_data = []
        for ev in ev_order:
            sub = inferences[inferences["evidence"] == ev]
            ev_data.append({
                "evidence": ev,
                "positive": ((~sub["is_negative_evidence"]) &
                             (~sub["is_ambiguous"])).sum(),
                "negative_evidence": sub["is_negative_evidence"].sum(),
                "ambiguous": sub["is_ambiguous"].sum(),
            })
        ev_df = pd.DataFrame(ev_data).set_index("evidence")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ev_df.plot(kind="bar", stacked=True, ax=ax,
                   color=["#4c72b0", "#c44e52", "#dd8452"])
        ax.set_xlabel("Evidence grade")
        ax.set_ylabel("Number of phenotype inferences")
        ax.set_title("Phenotype inferences by evidence grade and polarity")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(paths["figures_dir"] / "figure_phenotype_evidence_breakdown.png",
                    dpi=200)
        plt.close()
    else:
        logger.warning("No phenotype inferences produced; check graph structure")

    logger.info("Script 06 complete.")



# ============================================================================
# ORCHESTRATOR
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Combined PGx R-GCN pipeline (stages 1-6).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir", default="./data",
                   help="Directory holding ClinPGx TSV files and pathways/")
    p.add_argument("--results_dir", default="./results",
                   help="Where to write per-RUN_ID output folders")
    p.add_argument("--run_id", default=None,
                   help="Pin a RUN_ID (default: timestamp). Reuse to re-run a stage.")
    p.add_argument("--stages", default="all",
                   help='Stages to run: "all" or comma list e.g. "1,2,3"')
    p.add_argument("--force_cpu", action="store_true",
                   help="Force CPU even if CUDA is available")
    p.add_argument("--ablation_beta_zero", action="store_true",
                   help="Override HUB_PENALTY_BETA to 0 (ablation run)")
    p.add_argument("--hub_penalty_beta", type=float, default=None,
                   help="Override CONFIG.training.hub_penalty_beta. "
                        "Use to sweep ablations (0, 0.001, 0.005, 0.01).")
    p.add_argument("--supervision_regime",
                   default="default",
                   choices=["default", "ambig_as_pos", "nonassoc_excluded"],
                   help="How to handle ambiguous and not-associated drug-gene "
                        "pairs in supervised loss. "
                        "'default': CIBB baseline. "
                        "'ambig_as_pos': ambiguous merged into positives. "
                        "'nonassoc_excluded': not-associated dropped from graph.")
    p.add_argument("--split_mode", default=None,
                   choices=["dg_context", "strict_cold_drug"],
                   help="Override CONFIG.split.split_mode. "
                        "'dg_context' is the publication main; "
                        "'strict_cold_drug' is the sensitivity analysis.")
    p.add_argument("--seed", type=int, default=None,
                   help="Override CONFIG.seed. Use to run multi-seed sweeps "
                        "(suggested: 42, 123, 2026).")
    p.add_argument("--skip_literature", action="store_true",
                   help="Skip live PubMed queries in stage 5")
    p.add_argument("--top_n", type=int, default=None,
                   help="Override config top_n_predictions for stage 4")
    p.add_argument("--pubmed_email", default=None,
                   help="Override the pubmed email in CONFIG (recommended)")
    return p.parse_args()


def main():
    args = parse_args()

    # Apply CLI overrides to CONFIG
    cfg = load_config()
    cfg["paths"]["data_dir"] = args.data_dir
    cfg["paths"]["results_dir"] = args.results_dir
    cfg["force_cpu"] = args.force_cpu
    if args.pubmed_email:
        cfg["contextualisation"]["literature"]["pubmed_email"] = args.pubmed_email
    if args.hub_penalty_beta is not None:
        cfg["training"]["hub_penalty_beta"] = args.hub_penalty_beta
    if args.split_mode is not None:
        cfg["split"]["split_mode"] = args.split_mode
    if args.seed is not None:
        cfg["seed"] = args.seed

    # Parse stage selection
    if args.stages == "all":
        stages = [1, 2, 3, 4, 5, 6]
    else:
        stages = [int(s.strip()) for s in args.stages.split(",")]

    # If we're auto-generating the RUN_ID, embed the beta and split_mode
    # so concurrent ablation runs land in different folders by default
    if args.run_id is None and cfg.get("run_id") is None:
        beta = cfg["training"]["hub_penalty_beta"]
        if args.ablation_beta_zero:
            beta = 0.0
        beta_tag = f"beta{beta}".replace(".", "p").replace("-", "neg")
        split_tag = cfg["split"].get("split_mode", "dg_context")
        seed_tag = f"seed{cfg['seed']}"
        # D1: include supervision regime in RUN_ID for organised sweep folders
        regime_tag = f"reg-{args.supervision_regime}"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg["run_id"] = f"{ts}_{split_tag}_{beta_tag}_{seed_tag}_{regime_tag}"

    # Set up paths and seed
    paths = setup_paths(cfg, args.run_id)
    seed_everything(cfg["seed"])

    # D1: bake supervision_regime into the training config for permanent record
    cfg["training"]["supervision_regime"] = args.supervision_regime
    
    # Snapshot the complete resolved config to the run folder (every CLI
    # override is now baked in; this is the single source of truth for what
    # this run did)
    with open(paths["results_dir"] / "run_config.json", "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    # Master logger
    master_log = setup_logger(paths["logs_dir"] / "pipeline.log",
                               name="pipeline_orchestrator")
    log_versions(master_log)
    master_log.info(f"=== PGx R-GCN Combined Pipeline ===")
    master_log.info(f"RUN_ID: {paths['run_id']}")
    master_log.info(f"DATA_DIR: {paths['data_dir']}")
    master_log.info(f"RESULTS_DIR: {paths['results_dir']}")
    master_log.info(f"Stages to run: {stages}")
    master_log.info(f"Seed: {cfg['seed']}")
    master_log.info(f"Split mode: {cfg['split'].get('split_mode', 'dg_context')}")
    master_log.info(f"Hub-penalty beta: "
                    f"{0.0 if args.ablation_beta_zero else cfg['training']['hub_penalty_beta']}")
    master_log.info(f"Config snapshot written to {paths['results_dir'] / 'run_config.json'}")

    # Pre-flight check on pubmed_email (warn if still default)
    if 5 in stages and not args.skip_literature:
        email = cfg["contextualisation"]["literature"]["pubmed_email"]
        if "your.real.email" in email or "your.email" in email:
            master_log.warning(
                "PUBMED_EMAIL still set to placeholder. Stage 5 will run but "
                "NCBI may rate-limit or reject. Use --pubmed_email or edit CONFIG."
            )

    # Detect device once (re-used by stages 2-4)
    device = select_device(cfg)
    master_log.info(f"Device: {device}")

    t0 = time.time()
    if 1 in stages:
        master_log.info("--- STAGE 1: Build graph ---")
        logger = setup_logger(paths["logs_dir"] / "01_build_graph.log",
                              name="01_build_graph")
        log_versions(logger)
        run_stage_01_build_graph(cfg, paths, logger, args)

    if 2 in stages:
        master_log.info("--- STAGE 2: Train model ---")
        logger = setup_logger(paths["logs_dir"] / "02_train_model.log",
                              name="02_train_model")
        log_versions(logger)
        run_stage_02_train(cfg, paths, logger, args, device)

    if 3 in stages:
        master_log.info("--- STAGE 3: Evaluate ---")
        logger = setup_logger(paths["logs_dir"] / "03_evaluate.log",
                              name="03_evaluate")
        run_stage_03_evaluate(cfg, paths, logger, args, device)

    if 4 in stages:
        master_log.info("--- STAGE 4: Interpret ---")
        logger = setup_logger(paths["logs_dir"] / "04_interpret.log",
                              name="04_interpret")
        run_stage_04_interpret(cfg, paths, logger, args, device)

    if 5 in stages:
        master_log.info("--- STAGE 5: Contextualise ---")
        logger = setup_logger(paths["logs_dir"] / "05_contextualise.log",
                              name="05_contextualise")
        run_stage_05_contextualise(cfg, paths, logger, args)

    if 6 in stages:
        master_log.info("--- STAGE 6: Phenotype propagation ---")
        logger = setup_logger(paths["logs_dir"] / "06_propagate.log",
                              name="06_propagate_phenotypes")
        run_stage_06_propagate(cfg, paths, logger, args)

    dt = time.time() - t0
    master_log.info(f"=== Pipeline complete in {dt/60:.1f} minutes ===")
    master_log.info(f"All artifacts in: {paths['results_dir']}")


if __name__ == "__main__":
    main()
