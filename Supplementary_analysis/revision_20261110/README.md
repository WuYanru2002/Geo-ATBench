# Geo-ATBench revision materials

Version: **1.1**  
Release date: **2026-11-10**  
Code release: https://github.com/WuYanru2002/Geo-ATBench/releases/tag/v1.1  
Dataset v1.0: https://zenodo.org/records/18980673  
Dataset concept DOI (all versions): https://doi.org/10.5281/zenodo.18980672

This supplement contains the fixed split assignments, revision utility code,
and experiment summary tables supplied for the November 2026 revision of
Geo-ATBench. It supplements the existing dataset and repository.

## Contents

- `splits/original_split/`: fixed original Train/Validation/Test clip assignments.
- `splits/recording_disjoint/`: fixed source-recording-disjoint assignments.
- `splits/location_disjoint/`: fixed continent-based assignments (Europe for
  training, North America for testing, the other listed continents for validation).
- `code/recording_disjoint.py`: grouping helper using explicit source-recording IDs.
- `code/location_disjoint.py`: continent partition helper with source/location checks.
- `code/inter_attention_scaling.py`: feature-axis attention and fusion implementation
  with `sqrt_demb` and `one` scaling options.
- `code/temporal_inference.py`: paired current/historical GSC inference helper.
- `code/train_with_shuffled_gsc.py`: partition-specific GSC derangement and training helpers.
- `results/`: supplied experiment summaries; metric names are given in each CSV header.
- `MANIFEST.json`: SHA-256 hashes of the 19 supplied source files.
- `VERSION`: material version.

The split CSVs contain `id` and `class_name` columns. Join clip IDs to the existing
Geo-ATBench dataset for audio and GSC. Source-recording/location grouping helpers
expect additional metadata as described by their function arguments; those fields
are not present in the two-column assignment CSVs.

The `__main__` blocks in the supplied Python files are small synthetic usage
examples. Full-data execution requires the dataset, backbone models/checkpoints,
and experiment configuration. The `results/` CSVs are aggregate summaries, not
per-clip prediction scores or per-seed result tables. In particular,
`inter_ablation_scaling.csv` contains micro-AP, as stated by its column headers.
The original filenames, including `disjiont`, are preserved.

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

## Dependencies and usage

The utility files use Python, NumPy, PyTorch, and scikit-learn as applicable.
Import the supplied functions/classes into the experiment pipeline. Their
synthetic examples can also be run directly, for example:

```bash
python code/recording_disjoint.py
python code/location_disjoint.py
```

No library versions have been added beyond the supplied source, because the
authors' exact training environment is not included in this material folder.
The existing repository license and original dataset/audio license terms apply.

## Version history

- **1.1 (2026-11-10):** add this revision supplement; preserve the supplied
  source files byte-for-byte and record their checksums.
- **Dataset 1.0:** original dataset record at https://zenodo.org/records/18980673.
