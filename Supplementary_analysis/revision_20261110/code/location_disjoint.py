CONTINENT_PARTITION = {
    'Europe': 'train',
    'North America': 'test',
    'Asia': 'validation',
    'Africa': 'validation',
    'Oceania': 'validation',
    'South America': 'validation',
    'Antarctica': 'validation',
}


def location_disjoint(records):
    result = {part: [] for part in ('train', 'validation', 'test')}
    owners = {'source_recording_id': {}, 'location_key': {}}
    ids = set()
    for row in records:
        clip_id = str(row['id'])
        if clip_id in ids:
            raise ValueError('Clip IDs must be unique')
        ids.add(clip_id)
        part = CONTINENT_PARTITION[row['continent']]
        for field, assigned in owners.items():
            key = str(row[field])
            if not key:
                raise ValueError(f'Missing {field}')
            if key in assigned and assigned[key] != part:
                raise ValueError(f'{field} crosses partitions: {key}')
            assigned[key] = part
        result[part].append(dict(row))
    return {part: sorted(rows, key=lambda row: str(row['id'])) for part, rows in result.items()}


def main():
    records = [{'id': str(i), 'source_recording_id': str(i), 'location_key': str(i), 'continent': continent} for i, continent in enumerate(CONTINENT_PARTITION)]
    parts = location_disjoint(records)
    print({part: {'clips': len(rows), 'continents': sorted({r['continent'] for r in rows})} for part, rows in parts.items()})


if __name__ == '__main__':
    main()
