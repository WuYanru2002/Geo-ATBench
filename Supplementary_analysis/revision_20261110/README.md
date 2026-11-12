# Geo-ATBench revision materials

Version: **1.1**  
Release date: **2026-11-10**  
Code release: https://github.com/WuYanru2002/Geo-ATBench/releases/tag/v1.1  
Dataset v1.0: https://zenodo.org/records/18980673  
Dataset v1.1: https://zenodo.org/records/24102468
Dataset concept DOI (all versions): https://doi.org/10.5281/zenodo.18980672

## Contents

- `code/`: scripts demonstrating the disjoint-split, temporal, fusion-ablation, shuffled-GSC training, and blinded audio-only annotation procedures.
- `results/`: aggregate evaluation tables. The original model and fusion identifiers, metric names, and aggregation details are retained in each CSV so that results remain interpretable.
- `splits/`: fixed Train/Validation/Test clip assignments for the three evaluation protocols.
- `MANIFEST.json`: file checksums and split-size checks.

Pretrained backbone checkpoints are not stored in this Git repository because of GitHub file-size limits. The intended archive for the PANNs, AST, and CLAP initialization files is the Zenodo v1.1 dataset record above; verify its file list before citing the checkpoints as available. These are not fine-tuned experiment checkpoints.

## Fixed split sizes

| Protocol | Train | Validation | Test |
|---|---:|---:|---:|
| original_split | 2697 | 578 | 579 |
| recording_disjoint | 2681 | 565 | 608 |
| location_disjoint | 2731 | 463 | 660 |

Each protocol covers the same 3,854 unique clip IDs, with disjoint clip IDs across
its Train/Validation/Test files. Recording and spatial independence require the
associated source/location metadata in addition to these clip-assignment checks.
Five matched-seed training runs are repeated training evaluations on fixed splits.


## Version history

- **1.1 (2026-11-10):** add this revision supplement.
- **Dataset 1.0:** original dataset record at https://zenodo.org/records/18980673.
