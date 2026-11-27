import random
from collections import defaultdict


def recording_disjoint(records, source_counts, seed=1081):
    if len({str(row['id']) for row in records}) != len(records):
        raise ValueError('Clip IDs must be unique')
    groups = defaultdict(list)
    for row in records:
        clip_id = str(row['id'])
        source = clip_id if clip_id.startswith('foi') else str(row.get('source_recording_id') or row.get('original_id') or '')
        if not source:
            raise ValueError(f'Missing explicit source ID for {clip_id}')
        groups[source].append(dict(row, source_recording_id=source))
    if len(source_counts) != 3 or min(source_counts) <= 0 or sum(source_counts) != len(groups):
        raise ValueError('Three positive source counts must sum to the source total')
    sources = sorted(groups)
    random.Random(seed).shuffle(sources)
    result, start = {}, 0
    for part, count in zip(('train', 'validation', 'test'), source_counts):
        result[part] = sorted([row for source in sources[start:start + count] for row in groups[source]], key=lambda row: str(row['id']))
        start += count
    return result


def main():
    records = [{'id': f'{source}_{clip}', 'original_id': str(source)} for source in range(5) for clip in range(2)]
    records.append({'id': 'foi_example'})
    parts = recording_disjoint(records, source_counts=[4, 1, 1], seed=1081)
    print({part: {'clips': len(rows), 'sources': len({r['source_recording_id'] for r in rows})} for part, rows in parts.items()})


if __name__ == '__main__':
    main()
