# Geo-ATBench revision materials

Version: **1.1**  
Release date: **2026-11-10**  
Code release: https://github.com/WuYanru2002/Geo-ATBench/releases/tag/v1.1  
Dataset v1.0: https://zenodo.org/records/18980673  
Dataset concept DOI (all versions): https://doi.org/10.5281/zenodo.18980672

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
