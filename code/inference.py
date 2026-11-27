import torch
import torch.nn.functional as F


def mapped_audio_scores(audioset_scores, class_indices):
    # PANNs and AST use AudioSet outputs; their model adapters provide scores.
    return torch.stack([audioset_scores[:, indices].max(dim=1).values
                        for indices in class_indices], dim=1)


def audio_text_scores(audio_embeddings, text_embeddings):
    # CLAP compares normalized audio and text embeddings directly.
    audio = F.normalize(audio_embeddings, dim=-1)
    text = F.normalize(text_embeddings, dim=-1)
    return audio @ text.T


if __name__ == "__main__":
    audioset_scores = torch.rand(2, 527)
    mapping = [[0, 1], [2, 3], [4]]
    print(mapped_audio_scores(audioset_scores, mapping).shape)
    print(audio_text_scores(torch.randn(2, 512), torch.randn(3, 512)).shape)
