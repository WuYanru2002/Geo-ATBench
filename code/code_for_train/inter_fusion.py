import math
import torch
from torch import nn


class FeatureEnhancer(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, query, context):
        q = query.unsqueeze(-1)
        kv = context.unsqueeze(-1)
        weights = torch.softmax((q @ kv.transpose(1, 2)) / math.sqrt(self.dim), dim=-1)
        attention = (weights @ kv).squeeze(-1)
        x = self.norm1(query + self.dropout1(attention))
        return self.norm2(x + self.dropout2(self.ffn(x)))


class InterFusion(nn.Module):
    def __init__(self, backbone, dim=128, gsc_dim=768, num_classes=28):
        super().__init__()
        self.backbone = backbone
        self.gsc_projection = nn.Sequential(nn.Linear(gsc_dim, (gsc_dim + dim) // 2), nn.ReLU(), nn.Linear((gsc_dim + dim) // 2, dim))
        self.audio_enhancer = FeatureEnhancer(dim)
        self.gsc_enhancer = FeatureEnhancer(dim)
        self.fusion_projection = nn.Linear(2 * dim, dim)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, audio, gsc):
        a = self.backbone(audio)
        g = self.gsc_projection(gsc)
        a_enhanced = self.audio_enhancer(a, g)
        g_enhanced = self.gsc_enhancer(g, a)
        fused = torch.cat((a_enhanced + g, g_enhanced + a), dim=-1)
        return self.classifier(self.fusion_projection(fused))


if __name__ == "__main__":
    model = InterFusion(nn.Linear(64, 128))
    audio = torch.randn(4, 64)
    gsc = torch.randn(4, 768)
    labels = torch.randint(0, 2, (4, 28)).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss = nn.BCEWithLogitsLoss()(model(audio, gsc), labels)
    loss.backward()
    optimizer.step()
    print(loss.item())
