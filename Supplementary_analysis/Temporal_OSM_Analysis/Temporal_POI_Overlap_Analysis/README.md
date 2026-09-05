# Temporal Raw-POI Overlap

This folder contains the raw-POI temporal-overlap analysis for the 579 clips in the Geo-AT test set. Each current POI record is compared with the historical OSM record

`poi__temporal_overlap.csv` reports the mean clip-level overlap grouped by time. The observed overlap is 88.87% for 2012-2015, 93.40% for 2016-2020, 90.39% for 2021-2025, and 91.51% overall.

For each clip, the calculation considers the 11 OSM categories. A category is active if it has at least one POI value in either the current or historical record. An active category is counted as matching when its value sets are identical, or when either value set is a subset of the other. The clip-level score is the number of matching active categories divided by the number of active categories.

`calculate_poi_temporal_overlap.py` is a implementation of this calculation. To run the example, place two JSON lists named `current_poi.json` and `historical_poi.json` beside the script; both must use an `id` field and a `poi_texts` list.
