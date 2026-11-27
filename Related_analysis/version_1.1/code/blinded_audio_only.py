import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score


def aggregate_annotations(annotations, clip_ids, events, uncertainty_aware):
    if len(annotations) != 3 or len(set(clip_ids)) != len(clip_ids):
        raise ValueError('Require three annotators and unique clip IDs.')
    votes = np.zeros((3, len(clip_ids), len(events)), dtype=int)
    for a, annotator in enumerate(annotations):
        if set(annotator) != set(clip_ids):
            raise ValueError('Annotator clip IDs must match the reference.')
        for i, clip_id in enumerate(clip_ids):
            entry = annotator[clip_id]
            selected = set(entry['selected'])
            if not selected <= set(events) or entry['uncertain'] not in (0, 1):
                raise ValueError('Unknown event or invalid uncertainty flag.')
            for c, event in enumerate(events):
                votes[a, i, c] = (1 if event in selected else
                                  -1 if uncertainty_aware and entry['uncertain'] else 0)
    return np.where((votes == 1).sum(axis=0) >= 2, 1,
                    np.where((votes == 0).sum(axis=0) >= 2, 0, -1))


def binary_annotation_agreement(annotations, clip_ids, events):
    raw = np.array([[[int(event in person[key]['selected']) for event in events]
                     for key in clip_ids] for person in annotations])
    positive_votes = raw.sum(axis=0).ravel()
    counts = np.column_stack((3-positive_votes, positive_votes))
    observed = float(np.mean(np.sum(counts*(counts-1), axis=1)/6))
    proportions = counts.sum(axis=0)/counts.sum()
    expected = float(np.sum(proportions**2))
    kappa = (observed-expected)/(1-expected) if expected < 1 else None
    return dict(observed_agreement=observed, Fleiss_kappa=kappa,
                uncertain_annotations=sum(person[key]['uncertain']
                    for person in annotations for key in clip_ids),
                annotator_clip_judgments=len(annotations)*len(clip_ids))


def compare_labels(original, consensus):
    valid = consensus != -1
    added = (original == 0) & (consensus == 1)
    removed = (original == 1) & (consensus == 0)
    intersection = int(((original == 1) & (consensus == 1)).sum())
    union = int((((original == 1) | (consensus == 1)) & valid).sum())
    return dict(added=int(added.sum()), removed=int(removed.sum()),
                uncertain=int((~valid).sum()),
                positive_label_Jaccard=intersection/union if union else None,
                completely_matching_clips=int((valid.all(axis=1) &
                    (original == consensus).all(axis=1)).sum()),
                per_event_added=added.sum(axis=0).tolist(),
                per_event_removed=removed.sum(axis=0).tolist())


def evaluate_unchanged_predictions(predictions, clip_ids, events, original, blind):
    if not np.isin(original, [0, 1]).all() or not np.isin(blind, [0, 1]).all():
        raise ValueError('Model evaluation requires binary reference labels.')
    result = {}
    for name, by_id in predictions.items():
        if set(by_id) != set(clip_ids):
            raise ValueError('Prediction Test IDs must match annotated clips.')
        if any(set(row) != set(events) for row in by_id.values()):
            raise ValueError('Prediction event names must match the reference.')
        scores = np.array([[by_id[key][event] for event in events] for key in clip_ids])
        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError('Prediction probabilities must be finite and in [0, 1].')
        result[name] = dict(original_micro_AP=float(average_precision_score(
            original.ravel(), scores.ravel())), blind_micro_AP=float(
            average_precision_score(blind.ravel(), scores.ravel())))
    gains = {reference+'_gain': result['fusion'][reference+'_micro_AP']-
             result['audio_only'][reference+'_micro_AP'] for reference in ('original', 'blind')}
    return dict(models=result, **gains)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(encoding='utf-8-sig') as stream:
        data = json.load(stream)
    events = data['events']
    reference = data['original_labels']
    clip_ids = list(reference)
    if not events or not clip_ids or len(set(events)) != len(events):
        raise ValueError('Require nonempty clip IDs and unique event names.')
    if any(set(row) != set(events) for row in reference.values()):
        raise ValueError('Reference event names must match the event list.')
    original = np.array([[reference[key][event] for event in events] for key in clip_ids])
    if not np.isin(original, [0, 1]).all():
        raise ValueError('Reference labels must be binary.')
    annotations = data['annotations']
    aware = aggregate_annotations(annotations, clip_ids, events, True)
    binary = aggregate_annotations(annotations, clip_ids, events, False)
    output = dict(event_order=events,
        annotation_agreement=binary_annotation_agreement(annotations, clip_ids, events),
        uncertainty_aware_comparison=compare_labels(original, aware),
        binary_majority_comparison=compare_labels(original, binary))
    if 'predictions' in data:
        output['unchanged_model_evaluation'] = evaluate_unchanged_predictions(
            data['predictions'], clip_ids, events, original, binary)
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
