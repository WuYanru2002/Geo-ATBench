import torch
from torch import nn
from torch.nn import functional as F


class LateFusion(nn.Module):
    def __init__(self, backbone, dim=128, gsc_dim=768, num_classes=28):
        super().__init__()
        self.backbone = backbone
        self.audio_head = nn.Linear(dim, num_classes)
        self.gsc_head = nn.Sequential(nn.Linear(gsc_dim, 512), nn.ReLU(), nn.LayerNorm(512), nn.Linear(512, num_classes))
        self.lambda_raw = nn.Parameter(torch.zeros(num_classes))

    def forward(self, audio, gsc):
        audio_logits = self.audio_head(self.backbone(audio))
        gsc_logits = self.gsc_head(gsc)
        return audio_logits + F.softplus(self.lambda_raw) * gsc_logits


if __name__ == "__main__":
    model = LateFusion(nn.Linear(64, 128))
    audio = torch.randn(4, 64)
    gsc = torch.randn(4, 768)
    labels = torch.randint(0, 2, (4, 28)).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss = nn.BCEWithLogitsLoss()(model(audio, gsc), labels)
    loss.backward()
    optimizer.step()
    print(loss.item())
