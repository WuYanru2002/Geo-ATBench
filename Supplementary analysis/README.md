# Supplementary Files

This directory contains four result tables used to document POI--sound-event
relationships, dataset geographic coverage, and the independent AGL1K GSC-only
baseline.

## Files

1. **`complete lists of the top-50 POI values for each class.csv`**
   - A table of the ranked POI values associated with each of the
     28 Geo-AT sound-event classes.
   - Columns: `event`, `excel_rank`, `poi_value`, `P_value_given_event_pct`,
     `P_value_given_non_event_pct`, `discrimination_gap_pp`, and
     `implied_balanced_accuracy_pct`.
   - For every event, POI values are ranked by their discriminative difference
     between clips containing that event and clips not containing it. The final
     column is the balanced accuracy implied by using that single POI value as
     a binary cue.

2. **`Top1-related POI value and implied Balanced Accuracy for each sound-event class.csv`**
   - A compact 28-row summary: one most-related POI value for each Geo-AT
     sound-event class.
   - Columns: `Sound event`, `Top1-related POI value`, `P(value|event)`,
     `P(value|non-event)`, and `Implied Balanced Accuracy`.

3. **`number of clips and unique recording  locations per class.xlsx`**
   - **Sheet1** reports, for each Geo-AT event class, the number of clips
     (`Nc`), the number of unique recording locations (`Lc`), and the ratio
     `Lc/Nc`. The class columns are distributed across multiple blocks in the
     sheet.
   - **Sheet2** lists illustrative event--POI associations, including POI
     value, `P(value|event)`, `P(value|non-event)`, and their difference. For
     example, it includes POI cues such as `beach` and `coastline` for Waves.

4. **`GSC-only baseline results on the independent AGL1K dataset.xlsx`**
   - A single-sheet summary of the POI/GSC-only baseline on the independent
     AGL1K dataset.
   - It reports 1,444 available AGL1K clips and the fold-1 split sizes
     (959 training, 168 validation, 283 test), together with Micro F1,
     Macro F1, Micro mAP, and Macro mAP.
