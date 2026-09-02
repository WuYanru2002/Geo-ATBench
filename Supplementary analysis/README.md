# Supplementary Files

This directory contains seven supplementary result tables. The first three
provide POI-overlap analyses used to assess potential geographic-context
overlap across dataset partitions. The remaining files document POI--sound-event
relationships, dataset geographic coverage, and the independent AGL1K GSC-only
baseline.

## Files

1. **`Geo-AT_poi_text_overlap.xlsx`**
   - Geo-AT POI-text overlap analysis.
   - The workbook reports pairwise mean overlap and directional sample-level
     overlap between training, validation, and test partitions.

2. **`Geo-AT_bert_overlap.xlsx`**
   - Geo-AT BERT-based POI overlap analysis.
   - It reports within-category BERT similarity overlap between partitions.

3. **`poi_text_overlap_AGL1k.xlsx`**
   - POI-text overlap analysis for AGL1K, reported in the same split-comparison
     format as the Geo-AT text-overlap analysis.

4. **`complete lists of the top-50 POI values for each class.csv`**
   - A table of the ranked POI values associated with each of the 28 Geo-AT
     sound-event classes.
   - Columns: `event`, `excel_rank`, `poi_value`, `P_value_given_event_pct`,
     `P_value_given_non_event_pct`, `discrimination_gap_pp`, and
     `implied_balanced_accuracy_pct`.
   - For every event, POI values are ranked by their discriminative difference
     between clips containing that event and clips not containing it. The final
     column is the balanced accuracy implied by using that single POI value as
     a binary cue.

5. **`Top1-related POI value and implied Balanced Accuracy for each sound-event class.csv`**
   - A compact 28-row summary: one most-related POI value for each Geo-AT
     sound-event class.
   - Columns: `Sound event`, `Top1-related POI value`, `P(value|event)`,
     `P(value|non-event)`, and `Implied Balanced Accuracy`.

6. **`number of clips and unique recording locations per class.xlsx`**
   - **Sheet1** reports, for each Geo-AT event class, the number of clips
     (`Nc`), the number of unique recording locations (`Lc`), and the ratio
     `Lc/Nc`.
   - **Sheet2** lists illustrative event--POI associations, including POI
     value, `P(value|event)`, `P(value|non-event)`, and their difference.

7. **`GSC-only baseline results on the independent AGL1K dataset.xlsx`**
   - A single-sheet summary of the POI/GSC-only baseline on the independent
     AGL1K dataset.
   - It reports 1,444 available AGL1K clips and the split sizes
     (959 training, 168 validation, 283 test), together with Micro F1,
     Macro F1, Micro mAP, and Macro mAP.
