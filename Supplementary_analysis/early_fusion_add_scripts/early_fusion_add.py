"""Model-independent example of additive early fusion and one training step."""

import torch
from torch import nn


def early_fusion_add(audio_features, gsc_features, gsc_projection, gsc_norm=None):
    """Project GSC features, broadcast them, and add them to audio features.

    audio_features: [batch, channels, time, frequency]
    gsc_features:   [batch, gsc_dimension]
    """
    if audio_features.ndim != 4 or gsc_features.ndim != 2:
        raise ValueError("Expected audio [B,C,T,F] and GSC [B,D] tensors.")
    if audio_features.size(0) != gsc_features.size(0):
        raise ValueError("Audio and GSC batches must have the same size.")
    if gsc_norm is not None:
        gsc_features = gsc_norm(gsc_features)
    projected_gsc = gsc_projection(gsc_features)
    projected_gsc = projected_gsc[:, :, None, None]
    return audio_features + projected_gsc


class AdditiveFusionClassifier(nn.Module):
    """Apply additive fusion, pool the feature map, and produce class logits."""

    def __init__(self, audio_channels, gsc_dimension, num_classes):
        super().__init__()
        self.gsc_projection = nn.Sequential(
            nn.Linear(gsc_dimension, audio_channels),
            nn.LayerNorm(audio_channels),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(audio_channels, num_classes)

    def forward(self, audio_features, gsc_features):
        fused = early_fusion_add(
            audio_features, gsc_features, self.gsc_projection
        )
        pooled = fused.mean(dim=(2, 3))
        return self.classifier(pooled)


def main():
    batch_size = 4
    audio_channels = 64
    gsc_dimension = 768
    audio_features = torch.randn(batch_size, audio_channels, 32, 16)
    gsc_features = torch.randn(batch_size, gsc_dimension)
    model = AdditiveFusionClassifier(audio_channels, gsc_dimension, 28)
    labels = torch.randint(0, 2, (batch_size, 28)).float()
    logits = model(audio_features, gsc_features)
    loss = nn.BCEWithLogitsLoss()(logits, labels)
    loss.backward()
    print("audio features:", tuple(audio_features.shape))
    print("GSC features:", tuple(gsc_features.shape))
    print("logits:", tuple(logits.shape))
    print("one-step multi-label loss:", f"{loss.item():.4f}")


if __name__ == "__main__":
    main()
