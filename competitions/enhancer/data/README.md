# Enhancer Data

Download from the Google Drive link in the competition description doc.

Place the following files in this directory (`competitions/enhancer/data/`):

```
data/
  X.csv          # Feature matrix (chr + 12 molecular features for training)
  y.csv          # Binary labels (0 or 1)
  X_full.csv     # Full feature matrix (all 30 columns, including metadata and experimental outcomes)
```

## Features in X.csv

Only the `chr` column and 12 molecular features are included in `X.csv`:

| Feature | Description |
|---|---|
| chr | Chromosome (used for per-chromosome scoring, not a training feature) |
| distanceToTSS | Distance from candidate element to gene TSS (bp) |
| numTSSEnhGene | Number of TSS of other genes between element and gene |
| numCandidateEnhGene | Number of other candidate elements between element and gene |
| normalizedDNase_enh | Chromatin accessibility signal of the element |
| normalizedDNase_prom | Chromatin accessibility signal of the gene promoter |
| numNearbyEnhancers | Number of other candidate elements within 5000 bp |
| sumNearbyEnhancers | Sum of activities of nearby candidate elements within 5000 bp |
| ubiquitousExpressedGene | Whether the gene is ubiquitously expressed across cell types |
| 3DContact | 3D interaction frequency between element and gene promoter |
| 3DContact_squared | Squared 3D interaction frequency |
| normalizedDNase_enh_squared | Squared chromatin accessibility of the element |
| ABC.Score | Activity-By-Contact score |

## X_full.csv

`X_full.csv` contains all 30 columns from the original dataset, including genomic coordinates, gene metadata, experimental outcomes (EffectSize, pValueAdjusted, Significant), and statistical power estimates. These additional columns should **not** be used as training features.
