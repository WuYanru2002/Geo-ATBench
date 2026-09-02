"""
Each input is a list of samples.  A sample is a dictionary whose keys are
OSM categories and whose values are the selected POI attribute texts.
"""

OSM_CATEGORIES = (
    "Aeroway", "Amenity", "Building", "Highway", "Landuse", "Leisure",
    "Natural", "Public_transport", "Railway", "Tourism", "Waterway",
)


def mean_pairwise_overlap(input1, input2):
    """Return the mean fraction of matching attributes over all cross-set pairs."""
    if not input1 or not input2:
        return 0.0

    total = 0.0
    for sample1 in input1:
        for sample2 in input2:
            matching_categories = sum(
                sample1.get(category) is not None
                and sample1.get(category) == sample2.get(category)
                for category in OSM_CATEGORIES
            )
            total += matching_categories / len(OSM_CATEGORIES)
    return total / (len(input1) * len(input2))


def main():
    # Replace these two example lists with the two sample sets to compare.
    input1 = [
        {"Amenity": "restaurant", "Highway": "residential", "Natural": "tree"},
        {"Amenity": "school", "Highway": "footway", "Natural": "grassland"},
    ]
    input2 = [
        {"Amenity": "restaurant", "Highway": "primary", "Natural": "tree"},
        {"Amenity": "cafe", "Highway": "footway", "Natural": "grassland"},
    ]
    print(f"Mean pairwise overlap: {mean_pairwise_overlap(input1, input2):.4%}")


if __name__ == "__main__":
    main()
