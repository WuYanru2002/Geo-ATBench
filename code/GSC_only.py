import torch
from torch import nn


class GSCClassifier(nn.Module):
    def __init__(self, input_dim=768, num_classes=28):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, gsc):
        return self.classifier(gsc)


if __name__ == "__main__":
    model = GSCClassifier()
    gsc = torch.randn(4, 768)
    labels = torch.randint(0, 2, (4, 28)).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss = nn.BCEWithLogitsLoss()(model(gsc), labels)
    loss.backward()
    optimizer.step()
    print(loss.item())
