# Supplementary Analyses

This directory contains seven supplementary files organized into two folders:

- `Overlap_Analysis/` contains 3 files, which quantify POI-based
  overlap among cross-validation partitions.
- `Sound_Event_Geographic_Feature_Analysis/` contains 4 files,
  which document POI--sound-event relationships, geographic coverage, and the
  GSC-only baseline.

## Overlap Analysis

The following files are located in `Overlap_Analysis/`.

1. **`Geo-AT_poi_text_overlap.xlsx`**
   - POI-text overlap analysis for Geo-AT. Each audio clip is represented by
     selected category-level POI text attributes from the 11 OSM feature
     categories.
   - **`poi_text_overlap`**: reports mean pairwise POI-text overlap between 
      different partitions.
   - **`data_all_partition_text_overlap`**: reports directional sample-level
     overlap, including numerator/denominator counts and percentages for query
     clips.

2. **`Geo-AT_bert_overlap.xlsx`**
   - BERT-based POI overlap analysis for Geo-AT, using the same representative
     category-level POI attributes as the text-overlap analysis.
   - **`bert_overlap`**: reports mean POI similarity derived from BERT
     embeddings, with comparisons performed within the same OSM category.
   - **`data_all_partion_bert_overlap`**: reports directional sample-level
     BERT-overlap counts and percentages.

3. **`poi_text_overlap_AGL1k.xlsx`**
   - POI-text overlap analysis for AGL1k.
   - **`poi_text_overlap_AGL1k`**: reports the mean pairwise and directional
     sample-level POI-text overlap results used as a reference analysis for the
     independent AGL1k dataset.

## Sound-Event and Geographic Feature Analysis

The following files are located in `Sound_Event_Geographic_Feature_Analysis/`.

4. **`complete_lists_of_the_top-50_POI_values_for_each_class.csv`**
   - Contains ranked POI values for each of the 28 Geo-AT sound-event classes.
   - Columns: `event`, `excel_rank`, `poi_value`,
     `P_value_given_event_pct`, `P_value_given_non_event_pct`,
     `discrimination_gap_pp`, and `implied_balanced_accuracy_pct`.
   - POI values are ranked by the difference between their prevalence in clips
     with and without the corresponding event. The final column gives the
     balanced accuracy implied by using that POI value as a binary cue.

5. **`Top1-related_POI_value_and_implied_Balanced_Accuracy_for_each_class.csv`**
   - A compact 28-row summary containing the most-related POI value for every
     Geo-AT sound-event class.
   - Columns: `Sound event`, `Top1-related POI value`, `P(value|event)`,
     `P(value|non-event)`, and `Implied Balanced Accuracy`.

6. **`number_of_clips_and_unique_recording_locations_per_class.xlsx`**
   - **`Sheet1`**: reports the number of clips (`Nc`), unique recording
     locations (`Lc`), and the ratio `Lc/Nc` for each Geo-AT event class.
   - **`Sheet2`**: provides illustrative event--POI associations, including
     POI value, `P(value|event)`, `P(value|non-event)`, and their difference.

7. **`GSC-only_AGL1K.xlsx`**
   - **`Sheet1`**: reports results for the GSC-only baseline
     on AGL1k, including Micro F1, Macro F1, Micro mAP, and Macro mAP.
