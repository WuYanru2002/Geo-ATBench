""" Example of Mean Pairwise Overlap for POI category-value sets."""


OSM_CATEGORIES = (
    "Aeroway", "Amenity", "Building", "Highway", "Landuse", "Leisure",
    "Natural", "Public_transport", "Railway", "Tourism", "Waterway",
)


def mean_pairwise_overlap(input1, input2):
    """Return the mean POI overlap across all cross-set sample pairs.

    Each sample maps an OSM category to a set (or list) of POI values. For a
    category to match, both sets must be non-empty and either identical or a
    subset of the other. The pair score is normalized by the categories in
    which at least one sample contains POI information.
    """
    if not input1 or not input2:
        return 0.0

    pair_scores = []
    for sample1 in input1:
        for sample2 in input2:
            active_categories = [
                category for category in OSM_CATEGORIES
                if sample1.get(category) and sample2.get(category)
                or sample1.get(category) or sample2.get(category)
            ]
            if not active_categories:
                pair_scores.append(0.0)
                continue

            matching_categories = 0
            for category in active_categories:
                values1 = set(sample1.get(category, ()))
                values2 = set(sample2.get(category, ()))
                if values1 and values2 and (values1 == values2 or values1.issubset(values2) or values2.issubset(values1)):
                    matching_categories += 1
            pair_scores.append(matching_categories / len(active_categories))

    return sum(pair_scores) / len(pair_scores)


def main():
    input1 = [
        {"Amenity": {"restaurant"}, "Highway": {"residential", "footway"}},
        {"Amenity": {"school"}, "Natural": {"park"}},
    ]
    input2 = [
        {"Amenity": {"restaurant", "cafe"}, "Highway": {"residential"}},
        {"Amenity": {"cafe"}, "Natural": {"park"}},
    ]
    print(f"Mean pairwise overlap: {mean_pairwise_overlap(input1, input2):.4%}")


if __name__ == "__main__":
    main()
