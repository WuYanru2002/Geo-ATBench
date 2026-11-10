import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score
from torch import nn


def temporal_inference(model, ids, audio, targets, current_gsc, history_ids, history_gsc, batch_size=32):
    ids, history_ids = list(map(str, ids)), list(map(str, history_ids))
    if batch_size < 1 or len(set(ids)) != len(ids) or len(set(history_ids)) != len(history_ids) or set(ids) != set(history_ids):
        raise ValueError('Unique paired clip IDs and a positive batch size are required')
    if not (len(ids) == len(audio) == len(targets) == len(current_gsc) == len(history_gsc)):
        raise ValueError('Input lengths must agree')
    lookup = {clip_id: i for i, clip_id in enumerate(history_ids)}
    historical = history_gsc[[lookup[clip_id] for clip_id in ids]]
    if current_gsc.shape != historical.shape:
        raise ValueError('Current and historical GSC shapes must agree')
    device = next(model.parameters()).device
    model.eval()
    predictions, metrics = {}, {}
    with torch.inference_mode():
        for condition, gsc in (('current', current_gsc), ('historical', historical)):
            blocks = [model(audio[start:start + batch_size].to(device), gsc[start:start + batch_size].to(device)).cpu() for start in range(0, len(ids), batch_size)]
            logits = torch.cat(blocks)
            if logits.shape != targets.shape or not torch.isfinite(logits).all():
                raise ValueError('Model output must be finite N-by-C logits')
            p = logits.sigmoid().numpy()
            y = targets.cpu().numpy()
            predictions[condition] = {'ids': ids, 'logits': logits.numpy(), 'probabilities': p, 'binary': p > .5}
            metrics[condition] = {'micro_AP': float(average_precision_score(y, p, average='micro')), 'f1_macro': float(f1_score(y, p > .5, average='macro', zero_division=0))}
    return {'metrics': metrics, 'delta_history_minus_current': {key: metrics['historical'][key] - metrics['current'][key] for key in metrics['current']}, 'predictions': predictions}


def main():
    torch.manual_seed(42)
    model = nn.Bilinear(8, 6, 3)
    audio, current = torch.randn(12, 8), torch.randn(12, 6)
    historical = current + .05 * torch.randn_like(current)
    labels = torch.randint(0, 2, (12, 3)).float()
    ids = [str(i) for i in range(12)]
    result = temporal_inference(model, ids, audio, labels, current, ids[::-1], historical.flip(0))
    print(result['metrics'])
    print(result['delta_history_minus_current'])


if __name__ == '__main__':
    main()
