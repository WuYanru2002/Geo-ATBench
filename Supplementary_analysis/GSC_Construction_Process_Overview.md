# GSC Construction Workflow

This document describes how the Geographic Semantic Context (GSC) representation was constructed for each audio clip in Geo-ATBench.

```mermaid
flowchart LR
    A[Clip GPS coordinate] --> B[Overpass API query]
    B --> C[OSM entities and tags]
    C --> D[Keep tags from 11 predefined OSM feature categories]
    D --> E[English POI text units: Key: value]
    E --> F[BERT-base-uncased]
    F --> G[CLS vector for each POI text]
    G --> H[Mean pooling]
    H --> I[768-dimensional GSC vector]
```

## 1. OSM Feature Categories

We used OSM feature keys as geographic semantic categories. Eleven categories were retained from the OSM taxonomy because they cover environmental information potentially relevant to sound events, such as transportation infrastructure and buildings. Highly specific, rare, clearly overlapping, or weakly relevant categories, such as military facilities, were excluded.

## 2. POI Text Construction

For each clip, nearby OSM entities were retrieved from the clip GPS coordinate through the Overpass API. OSM entities provide a set of tags. Only tags belonging to the predefined 11 OSM feature categories were retained; non-geographic metadata, including names, addresses, phone numbers, e-mail addresses, and websites, was discarded.

For example, the following OSM tags:

```json
{
  "bus": "yes",
  "highway": "bus_stop",
  "name": "Tuinbouwweg",
  "public_transport": "platform",
  "ref:IFOPT": "NL:Q:63190290"
}
```

produce the POI text units:

```text
Highway: bus_stop
Public_transport: platform
```

Similarly, from a restaurant entity with address, contact, and operating-hour metadata, only the geographic semantic tag is retained:

```text
Amenity: restaurant
```

## 3. BERT Encoding and Pooling

Each POI text unit is encoded independently using `bert-base-uncased`. For a text unit, the representation is the final-layer hidden vector of the `[CLS]` token:

```python
cls_embedding = outputs.last_hidden_state[:, 0, :]
```

Each text unit therefore produces a 768-dimensional vector. If a clip has $N$ retained POI text units, their `[CLS]` vectors are mean-pooled to obtain one 768-dimensional GSC vector:

```text
GSC embedding = mean(CLS_1, CLS_2, ..., CLS_N)
```

This clip-level GSC vector is used together with the audio representation in the early-, intermediate-, and late-fusion models.

## 4. Handling Clips Without POIs

Clips without valid POI text were removed during dataset construction. Therefore, every clip included in the training, validation, and test sets has at least one valid POI text unit.
## End-to-End Example

Suppose an OSM query returns:

```json
{
  "highway": "bus_stop",
  "public_transport": "platform",
  "amenity": "restaurant",
  "waterway": "river",
  "name": "Tuinbouwweg",
  "phone": "+31 73 511 9133"
}
```

After filtering, the clip has four POI text units:

```text
Highway: bus_stop
Public_transport: platform
Amenity: restaurant
Waterway: river
```

Each unit is encoded by BERT into a 768-dimensional `[CLS]` vector. The four vectors are then averaged to form the final 768-dimensional GSC embedding for that clip.
