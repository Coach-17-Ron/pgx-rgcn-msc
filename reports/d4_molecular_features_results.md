# D4: Molecular Drug Features for Cold-Drug Prediction

## Objective

This experiment tested whether replacing learned drug-ID embeddings with molecular fingerprints improves drug–gene ranking, particularly for unseen drugs under the strict-cold protocol.

## Molecular representation

Drug structures were obtained from PharmGKB `chemicals.tsv`.

Morgan fingerprints were generated with:

- radius: 2
- size: 2,048 bits
- chirality: included
- implementation: RDKit
- storage type: `uint8`

For molecular feature mode:

- drugs with valid SMILES used a projected Morgan fingerprint plus a drug-type embedding;
- drugs without SMILES used a shared learned missing-feature embedding plus a drug-type embedding;
- drug-ID embeddings were not used;
- non-drug nodes retained learned node-ID and type embeddings.

## SMILES coverage

Across 5,210 graph drug nodes:

| Measure | Count |
|---|---:|
| Valid SMILES | 2,615 |
| Missing SMILES | 2,595 |
| Invalid SMILES | 0 |
| Unique canonical structures | 2,606 |

Across the three strict-cold test splits, 1,260 of 1,569 drug–gene test pairs involved drugs with molecular fingerprints.

After excluding test structures overlapping with training or context-only structures, 1,258 test pairs remained. The structure-clean and full SMILES-covered results were nearly identical, indicating that duplicate molecular structures did not materially influence the findings.

## Overall molecular-model performance

Mean test performance across three seeds:

| Model | Protocol | Mean MRR | SD |
|---|---|---:|---:|
| GCN | dg_context | 0.082973 | 0.011369 |
| R-GCN | dg_context | 0.068729 | 0.022078 |
| RGAT | dg_context | 0.045621 | 0.008731 |
| GCN | strict_cold | 0.074727 | 0.020408 |
| R-GCN | strict_cold | 0.064686 | 0.010962 |
| RGAT | strict_cold | 0.006468 | 0.005568 |

GCN remained the strongest molecular-feature architecture under both protocols.

## Strict-cold SMILES-covered comparison

The primary analysis compared molecular and ID-feature models on exactly the same strict-cold, SMILES-covered, structure-clean test pairs.

| Model | Mean ID MRR | Mean molecular MRR | Mean delta MRR | Positive seeds | Significant positive seeds | Significant negative seeds |
|---|---:|---:|---:|---:|---:|---:|
| GCN | 0.086588 | 0.080286 | -0.006302 | 0/3 | 0/3 | 0/3 |
| R-GCN | 0.075425 | 0.073262 | -0.002163 | 2/3 | 1/3 | 1/3 |
| RGAT | 0.045118 | 0.006953 | -0.038165 | 0/3 | 0/3 | 3/3 |

Paired bootstrap confidence intervals were calculated separately within each seed.

For GCN, all three molecular-feature estimates were lower than the matched ID baseline, but none of the seed-level confidence intervals excluded zero.

For R-GCN, the effect was heterogeneous: seed 123 improved significantly, seed 2026 declined significantly, and seed 42 was inconclusive.

For RGAT, molecular features caused a substantial and statistically clear decline for all three seeds.

## Interpretation

The experiment does not support a general claim that Morgan fingerprints improve cold-drug drug–gene ranking.

Molecular features provided deterministic representations for unseen drugs and produced better performance for SMILES-covered drugs than for drugs using the shared missing-SMILES embedding. However, replacing learned drug-ID embeddings with a simple projected Morgan fingerprint did not consistently improve performance relative to the matched ID-feature models.

The effect depended strongly on graph architecture and split:

- GCN remained comparatively stable but showed a small mean decline.
- R-GCN showed unstable, seed-dependent effects.
- RGAT deteriorated consistently and substantially.

These findings suggest that molecular structure alone is not sufficient to improve cold-drug ranking through direct feature replacement. More effective integration may require feature fusion, pretrained molecular encoders, or architecture-specific regularisation rather than substituting ID embeddings with a single linear fingerprint projection.

## Reproducibility

Feature generation:

```bash
python scripts/data/build_molecular_features.py
```

Matched molecular-versus-ID analysis:

```bash
python scripts/evaluate/compare_molecular_vs_id.py
```

All 18 molecular training runs and all 18 evaluations completed successfully. The full test suite passed with 23 tests.
