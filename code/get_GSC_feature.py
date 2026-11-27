import torch


def extract_gsc(poi_texts, tokenizer, bert):
    if not poi_texts:
        return None
    device = next(bert.parameters()).device
    inputs = tokenizer(poi_texts, padding=True, truncation=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    bert.eval()
    with torch.no_grad():
        embeddings = bert(**inputs).last_hidden_state[:, 0, :]
    return embeddings


def mean_gsc(embeddings):
    return None if embeddings is None else embeddings.mean(dim=0)
