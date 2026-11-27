import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PART_CODES = {'train': 1, 'validation': 2, 'test': 3}


def derangement(size, seed, part, epoch, namespace=20261028):
    if size < 2 or part not in PART_CODES:
        raise ValueError('A recognized partition and at least two samples are required')
    rng = np.random.default_rng(np.random.SeedSequence([namespace, seed, PART_CODES[part], epoch]))
    while True:
        donors = rng.permutation(size)
        if np.all(donors != np.arange(size)):
            return donors


class ShuffledGSCDataset(Dataset):
    def __init__(self, audio, gsc, labels, part, seed):
        if len(audio) != len(gsc) or len(audio) != len(labels):
            raise ValueError('Audio, GSC and label lengths must agree')
        self.audio, self.gsc, self.labels = audio, gsc, labels
        self.part, self.seed = part, seed
        self.donors = torch.empty(len(audio), dtype=torch.long).share_memory_()
        self.set_epoch(1 if part == 'train' else 0)

    def set_epoch(self, epoch):
        if (self.part == 'train' and epoch < 1) or (self.part != 'train' and epoch != 0):
            raise ValueError('Train epochs start at 1; evaluation mapping stays at epoch 0')
        self.donors.copy_(torch.as_tensor(derangement(len(self.audio), self.seed, self.part, epoch)))

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, index):
        return self.audio[index], self.gsc[int(self.donors[index])], self.labels[index]


def train_epoch(model, loader, optimizer, epoch):
    loader.dataset.set_epoch(epoch)
    model.train()
    total = 0.0
    for audio, gsc, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(model(audio, gsc), labels)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(audio)
    return total / len(loader.dataset)


def main():
    torch.manual_seed(42)
    data = {part: ShuffledGSCDataset(torch.randn(12, 8), torch.randn(12, 6), torch.randint(0, 2, (12, 3)).float(), part, 42) for part in PART_CODES}
    loader = DataLoader(data['train'], batch_size=4, shuffle=True)
    model = nn.Bilinear(8, 6, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    for epoch in (1, 2):
        print({'epoch': epoch, 'loss': train_epoch(model, loader, optimizer, epoch)})
    model.eval()
    with torch.inference_mode():
        for part in ('validation', 'test'):
            audio, gsc, labels = next(iter(DataLoader(data[part], batch_size=12)))
            print(part, model(audio, gsc).shape)


if __name__ == '__main__':
    main()
