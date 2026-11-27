import copy
import math

import torch
from torch import nn


class FeatureAxisAttention(nn.Module):
    def __init__(self, dimension, scaling):
        super().__init__()
        if scaling not in ('one', 'sqrt_demb'):
            raise ValueError('scaling must be one or sqrt_demb')
        self.scale = 1.0 if scaling == 'one' else math.sqrt(dimension)

    def forward(self, query, key, value):
        q, k, v = query.unsqueeze(-1), key.unsqueeze(-1), value.unsqueeze(-1)
        weights = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / self.scale, dim=-1)
        return torch.bmm(weights, v).squeeze(-1)


class FeatureEnhancer(nn.Module):
    def __init__(self, dimension, scaling):
        super().__init__()
        self.cross_attn = FeatureAxisAttention(dimension, scaling)
        self.ffn = nn.Sequential(nn.Linear(dimension, 4 * dimension), nn.ReLU(), nn.Dropout(.1), nn.Linear(4 * dimension, dimension))
        self.norm1, self.norm2 = nn.LayerNorm(dimension), nn.LayerNorm(dimension)
        self.dropout1, self.dropout2 = nn.Dropout(.1), nn.Dropout(.1)

    def forward(self, query, context):
        x = self.norm1(query + self.dropout1(self.cross_attn(query, context, context)))
        return self.norm2(x + self.dropout2(self.ffn(x)))


class InterFusion(nn.Module):
    def __init__(self, dimension, gsc_dimension, classes, scaling):
        super().__init__()
        hidden = (gsc_dimension + dimension) // 2
        self.gsc_projection = nn.Identity() if gsc_dimension == dimension else nn.Sequential(nn.Linear(gsc_dimension, hidden), nn.ReLU(), nn.Linear(hidden, dimension))
        self.audio_query = FeatureEnhancer(dimension, scaling)
        self.gsc_query = FeatureEnhancer(dimension, scaling)
        self.fusion_projection = nn.Linear(2 * dimension, dimension)
        self.classifier = nn.Sequential(nn.Linear(dimension, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(.3), nn.Linear(512, classes))

    def forward(self, audio, gsc):
        gsc = self.gsc_projection(gsc)
        a = self.audio_query(audio, gsc) + gsc
        g = self.gsc_query(gsc, audio) + audio
        return self.classifier(self.fusion_projection(torch.cat((a, g), dim=-1)))


def main():
    torch.manual_seed(42)
    initial = InterFusion(16, 12, 3, 'sqrt_demb').state_dict()
    audio, gsc = torch.randn(8, 16), torch.randn(8, 12)
    labels = torch.randint(0, 2, (8, 3)).float()
    for scaling in ('sqrt_demb', 'one'):
        torch.manual_seed(42)
        model = InterFusion(16, 12, 3, scaling)
        model.load_state_dict(copy.deepcopy(initial))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
        for _ in range(2):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(audio, gsc), labels)
            loss.backward()
            optimizer.step()
        print({'scaling': scaling, 'loss': loss.item(), 'parameters': sum(p.numel() for p in model.parameters())})


if __name__ == '__main__':
    main()
