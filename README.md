# PGx R-GCN MSc Project

Audited implementation for cold-drug pharmacogenomic gene ranking using an evidence-typed relational graph convolutional network.

## Graph schema

The graph contains four node types:

- drug
- gene
- variant
- phenotype

Phenotypes broadly include diseases, adverse drug reactions, clinical outcomes and treatment-response phenotypes.

Haplotypes are excluded because they fall outside the predefined four-node schema.

## Audited graph targets

- 39,076 nodes
- 47,072 directed edges
- 56 relation types
- 14,698 haplotype-related rows excluded
- zero unmapped in-scope relationships
- zero duplicate directed edges

## Planned workflow

1. Graph construction
2. Split construction and leakage audit
3. Feature construction
4. R-GCN training
5. Ranking evaluation
6. Phenotype and pathway interpretation
