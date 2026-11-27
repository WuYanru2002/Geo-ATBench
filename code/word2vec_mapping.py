import re

import numpy as np


def generate_mapping(source_labels, target_labels, word_vectors, top_k=3):
    if not source_labels or not 1 <= top_k <= len(source_labels):
        raise ValueError('Source labels and a valid top_k are required')

    def encode(label):
        phrases = (label, label.replace(' ', '_'), label.lower().replace(' ', '_'))
        keys = next(([key] for key in phrases if key in word_vectors), None)
        if keys is None:
            keys = []
            for token in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", label):
                key = next((key for key in (token, token.lower()) if key in word_vectors), None)
                if key is not None:
                    keys.append(key)
        if not keys:
            raise ValueError(f'No word vectors for {label!r}')
        vector = np.mean([word_vectors[key] for key in keys], axis=0)
        norm = np.linalg.norm(vector)
        if norm == 0 or not np.isfinite(vector).all():
            raise ValueError(f'Invalid word vector for {label!r}')
        return vector / norm

    source_vectors = np.stack([encode(label) for label in source_labels])
    mapping = {}
    for target in target_labels:
        similarity = source_vectors @ encode(target)
        indices = np.argsort(-similarity, kind='stable')[:top_k]
        mapping[target] = [source_labels[index] for index in indices]
    return mapping
