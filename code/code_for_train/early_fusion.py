import torch
from torch import nn


class EarlyFusion(nn.Module):
    def __init__(self, backbone, gsc_dim=768, feature_dim=64, embedding_dim=128, num_classes=28):
        super().__init__()
        self.gsc_projection = nn.Linear(gsc_dim, feature_dim)
        self.backbone = backbone
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, audio_features, gsc):
        gsc_features = self.gsc_projection(gsc).unsqueeze(1).expand(-1, audio_features.size(1), -1)
        combined = torch.cat((audio_features, gsc_features), dim=-1)
        return self.classifier(self.backbone(combined).mean(dim=1))


if __name__ == "__main__":
    model = EarlyFusion(nn.Linear(128, 128))
    audio = torch.randn(4, 10, 64)
    gsc = torch.randn(4, 768)
    labels = torch.randint(0, 2, (4, 28)).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss = nn.BCEWithLogitsLoss()(model(audio, gsc), labels)
    loss.backward()
    optimizer.step()
    print(loss.item())
