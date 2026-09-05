"""Example of directional sample-level POI overlap."""


OSM_CATEGORIES = (
    "Aeroway", "Amenity", "Building", "Highway", "Landuse", "Leisure",
    "Natural", "Public_transport", "Railway", "Tourism", "Waterway",
)


def sample_level_overlap(input1, input2):
    """Return the fraction of input1 samples with a matching POI representation in input2.

    For every query sample, a reference sample matches when all categories in
    the union of their non-empty POI categories satisfy the identical-or-
    subset rule. The result is directional because input1 is the denominator.
    """
    if not input1:
        return 0.0

    overlapping_samples = 0
    for query_sample in input1:
        found_match = False
        for reference_sample in input2:
            active_categories = [
                category for category in OSM_CATEGORIES
                if query_sample.get(category) or reference_sample.get(category)
            ]
            if not active_categories:
                continue

            matches = True
            for category in active_categories:
                query_values = set(query_sample.get(category, ()))
                reference_values = set(reference_sample.get(category, ()))
                if not (
                    query_values
                    and reference_values
                    and (
                        query_values == reference_values
                        or query_values.issubset(reference_values)
                        or reference_values.issubset(query_values)
                    )
                ):
                    matches = False
                    break
            if matches:
                found_match = True
                break
        overlapping_samples += int(found_match)

    return overlapping_samples / len(input1)


def main():
    input1 = [
        {"Amenity": {"restaurant"}, "Highway": {"residential"}},
        {"Amenity": {"school"}, "Natural": {"park"}},
    ]
    input2 = [
        {"Amenity": {"restaurant", "cafe"}, "Highway": {"residential"}},
        {"Amenity": {"cafe"}, "Natural": {"park"}},
    ]
    print(f"Sample-level overlap: {sample_level_overlap(input1, input2):.4%}")


if __name__ == "__main__":
    main()
