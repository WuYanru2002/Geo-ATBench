import numpy as np


def make_assignment(size, seed, partition, epoch):
    if size < 2:
        raise ValueError('Each subset must contain at least two samples.')
    rng = np.random.default_rng(np.random.SeedSequence(
        [20261028, seed, {'train': 1, 'validation': 2, 'test': 3}[partition], epoch]
    ))
    while True:
        assignment = rng.permutation(size)
        if np.all(assignment != np.arange(size)):
            return assignment


def shuffled_gsc_training(subsets, initialize, train_epoch, evaluate,
                          save_checkpoint, load_checkpoint,
                          seeds=(42, 43, 44, 45, 46), epochs=100, patience=15):
    for partition in ('train', 'validation', 'test'):
        audio, labels, gsc = subsets[partition]
        if len(audio) != len(labels) or len(audio) != len(gsc) or len(audio) < 2:
            raise ValueError(f'Invalid subset: {partition}')
    results = []
    for seed in seeds:
        state = initialize(seed)
        fixed = {}
        for partition in ('validation', 'test'):
            audio, labels, gsc = subsets[partition]
            assignment = make_assignment(len(gsc), seed, partition, 0)
            fixed[partition] = (audio, labels, np.asarray(gsc)[assignment])
        best_score = -np.inf
        best_epoch = 0
        stale_epochs = 0
        for epoch in range(1, epochs + 1):
            audio, labels, gsc = subsets['train']
            assignment = make_assignment(len(gsc), seed, 'train', epoch)
            train_epoch(state, (audio, labels, np.asarray(gsc)[assignment]), epoch)
            validation = evaluate(state, fixed['validation'], 'validation')
            score = float(validation['f1_samples'])
            if not np.isfinite(score):
                raise ValueError('Validation F1-samples must be finite.')
            if score > best_score:
                best_score = score
                best_epoch = epoch
                stale_epochs = 0
                save_checkpoint(state, seed)
            else:
                stale_epochs += 1
            if stale_epochs >= patience:
                break
        load_checkpoint(state, seed)
        test_result = evaluate(state, fixed['test'], 'test')
        results.append({'seed': seed, 'best_epoch': best_epoch,
                        'test_result': test_result})
    return results


def main():
    subsets = {}
    for partition in ('train', 'validation', 'test'):
        audio = np.array([f'{partition}_{i}' for i in range(4)])
        labels = np.eye(4, dtype=int)
        gsc = np.arange(8).reshape(4, 2)
        subsets[partition] = (audio, labels, gsc)

    def initialize(seed):
        return {'seed': seed}

    def train_epoch(state, batch, epoch):
        print('train', state['seed'], epoch, batch[0].tolist(), batch[2].tolist())

    def evaluate(state, batch, partition):
        print(partition, state['seed'], batch[0].tolist(), batch[2].tolist())
        return {'f1_samples': 0.0} if partition == 'validation' else {}

    def save_checkpoint(state, seed):
        print('save_checkpoint', seed)

    def load_checkpoint(state, seed):
        print('load_checkpoint', seed)

    shuffled_gsc_training(subsets, initialize, train_epoch, evaluate,
                          save_checkpoint, load_checkpoint, epochs=3)


if __name__ == '__main__':
    main()
