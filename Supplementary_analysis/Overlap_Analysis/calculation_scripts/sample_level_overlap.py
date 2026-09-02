"""
Each input is a list of samples.  A sample is a dictionary whose keys are
OSM categories and whose values are the selected POI attribute texts.
"""

OSM_CATEGORIES = (
    "Aeroway", "Amenity", "Building", "Highway", "Landuse", "Leisure",
    "Natural", "Public_transport", "Railway", "Tourism", "Waterway",
)
MINIMUM_MATCHING_CATEGORIES = 11


def sample_level_overlap(input1, input2):
    """Return the fraction of input1 samples with a qualifying match in input2.

    A qualifying match has at least MINIMUM_MATCHING_CATEGORIES matching
    category-level attributes.  The result is directional: swapping input1 and
    input2 can give a different value.
    """
    if not input1:
        return 0.0

    overlapping_samples = 0
    for sample1 in input1:
        for sample2 in input2:
            matching_categories = sum(
                sample1.get(category) is not None
                and sample1.get(category) == sample2.get(category)
                for category in OSM_CATEGORIES
            )
            if matching_categories >= MINIMUM_MATCHING_CATEGORIES:
                overlapping_samples += 1
                break
    return overlapping_samples / len(input1)


def main():
    # Replace these two example lists with the two sample sets to compare.
    input1 = [
        {category: "same_attribute" for category in OSM_CATEGORIES},
        {category: "first_set_attribute" for category in OSM_CATEGORIES},
    ]
    input2 = [
        {category: "same_attribute" for category in OSM_CATEGORIES},
        {category: "second_set_attribute" for category in OSM_CATEGORIES},
    ]
    print(f"Sample-level overlap: {sample_level_overlap(input1, input2):.4%}")


if __name__ == "__main__":
    main()
