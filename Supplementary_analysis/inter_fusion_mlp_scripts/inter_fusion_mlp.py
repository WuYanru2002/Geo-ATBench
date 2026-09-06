"""Model-independent example of concatenation-based intermediate fusion."""

import torch
from torch import nn


class IntermediateFusionMLP(nn.Module):
    """Fuse one audio vector and one GSC vector with an MLP classifier."""

    def __init__(self, audio_dimension, gsc_dimension, hidden_dimension, num_classes):
        super().__init__()
        self.gsc_projection = nn.Sequential(
            nn.Linear(gsc_dimension, audio_dimension),
            nn.LayerNorm(audio_dimension),
            nn.ReLU(),
        )
        self.audio_norm = nn.LayerNorm(audio_dimension)
        self.fusion = nn.Sequential(
            nn.Linear(audio_dimension * 2, hidden_dimension),
            nn.LayerNorm(hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, num_classes),
        )

    def forward(self, audio_features, gsc_features):
        if audio_features.ndim != 2 or gsc_features.ndim != 2:
            raise ValueError("Expected audio [B,A] and GSC [B,D] tensors.")
        if audio_features.size(0) != gsc_features.size(0):
            raise ValueError("Audio and GSC batches must have the same size.")
        audio_features = self.audio_norm(audio_features)
        projected_gsc = self.gsc_projection(gsc_features)
        fused_features = torch.cat((audio_features, projected_gsc), dim=1)
        return self.fusion(fused_features)


def main():
    model = IntermediateFusionMLP(
        audio_dimension=2048,
        gsc_dimension=768,
        hidden_dimension=512,
        num_classes=28,
    )
    audio_features = torch.randn(4, 2048)
    gsc_features = torch.randn(4, 768)
    labels = torch.randint(0, 2, (4, 28)).float()
    logits = model(audio_features, gsc_features)
    loss = nn.BCEWithLogitsLoss()(logits, labels)
    loss.backward()
    print("audio features:", tuple(audio_features.shape))
    print("GSC features:", tuple(gsc_features.shape))
    print("logits:", tuple(logits.shape))
    print("one-step multi-label loss:", f"{loss.item():.4f}")


if __name__ == "__main__":
    main()
