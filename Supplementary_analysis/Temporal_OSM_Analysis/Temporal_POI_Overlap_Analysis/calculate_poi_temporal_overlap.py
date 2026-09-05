"""Calculate clip-aligned raw POI overlap between current and historical OSM JSON files."""

import json
from pathlib import Path


OSM_CATEGORIES = (
    "Aeroway", "Amenity", "Building", "Highway", "Landuse", "Leisure",
    "Natural", "Public_transport", "Railway", "Tourism", "Waterway",
)
def raw_poi_overlap(current_record, historical_record):
    """Return active-category-normalized overlap using the identical-or-subset rule."""
    def values_by_category(record):
        values = {category: set() for category in OSM_CATEGORIES}
        for item in record.get("poi_texts", []):
            try:
                key, value = next(iter(json.loads("{" + item + "}").items()))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if key in values:
                values[key].add(value)
        return values

    current = values_by_category(current_record)
    historical = values_by_category(historical_record)
    active = [category for category in OSM_CATEGORIES if current[category] or historical[category]]
    if not active:
        return 0.0
    matches = sum(
        bool(current[category] and historical[category]
             and (current[category] == historical[category]
                  or current[category].issubset(historical[category])
                  or historical[category].issubset(current[category])))
        for category in active
    )
    return matches / len(active)


def main():
    current_json = Path("current_poi.json")
    historical_json = Path("historical_poi.json")
    current = {str(row["id"]): row for row in json.loads(current_json.read_text(encoding="utf-8"))}
    historical = {str(row["id"]): row for row in json.loads(historical_json.read_text(encoding="utf-8"))}
    scores = [raw_poi_overlap(current[audio_id], historical[audio_id]) for audio_id in current if audio_id in historical]
    print(f"Mean raw POI overlap: {sum(scores) / len(scores):.2%}")


if __name__ == "__main__":
    main()
