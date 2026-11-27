import torch
from torch import nn


class AudioOnly(nn.Module):
    def __init__(self, backbone, dim=128, num_classes=28):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, audio):
        return self.classifier(self.backbone(audio))


if __name__ == "__main__":
    model = AudioOnly(nn.Linear(64, 128))
    audio = torch.randn(4, 64)
    labels = torch.randint(0, 2, (4, 28)).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss = nn.BCEWithLogitsLoss()(model(audio), labels)
    loss.backward()
    optimizer.step()
    print(loss.item())
