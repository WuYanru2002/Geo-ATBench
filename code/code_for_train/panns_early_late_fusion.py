import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import sys
import warnings
import pandas as pd
import json
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from tqdm import tqdm
from torch.utils.data import DataLoader

# Standardized imports for local modules
try:
    from common_loader import PANNsDataset, PANNs_POIDataset, collate_fn_embedding_poi
except ImportError as e:
    print(f"Error importing local modules: {e}")
    sys.exit(1)

# PANNs (Cnn14) setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'audioset_tagging_cnn')))
try:
    from audioset_tagging_cnn.pytorch.models import Cnn14
except ImportError as e:
    print(f"Error: Could not import PANNs models. Ensure 'audioset_tagging_cnn' is in the path.")
    sys.exit(1)

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Config:
    """Standardized configuration for PANNs-POI Fusion"""
    # Data Paths
    BASE_DATA_DIR = 'path/to/data_splits'
    AUDIO_DIR = 'path/to/raw_audio_files'
    POI_FEAT_DIR = 'path/to/poi_features'
    OUTPUT_DIR = 'path/to/output_results'
    PRETRAINED_PATH = 'path/to/Cnn14_mAP=0.431.pth'

    # Hyperparameters
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    EPOCHS = 100
    LR = 1e-5
    WEIGHT_DECAY = 1e-4
    EARLY_STOP_PATIENCE = 15

    # Audio/PANNs Specs
    SAMPLE_RATE = 32000
    WINDOW_SIZE = 1024
    HOP_SIZE = 320
    MEL_BINS = 64
    FMIN = 50
    FMAX = 14000
    PANNS_EMBED_DIM = 2048
    POI_EMBED_DIM = 768
    AUX_WEIGHT = 0.1

def get_panns_backbone(pretrained_path, modify_for_early=False):
    """Loads Cnn14 backbone with optional input channel modification"""
    model = Cnn14(
        sample_rate=Config.SAMPLE_RATE, window_size=Config.WINDOW_SIZE,
        hop_size=Config.HOP_SIZE, mel_bins=Config.MEL_BINS, fmin=Config.FMIN,
        fmax=Config.FMAX, classes_num=527
    )
    
    if os.path.exists(pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        print(f"Warning: Pretrained weights not found at {pretrained_path}")

    if modify_for_early:
        original_conv1 = model.conv_block1.conv1
        model.conv_block1.conv1 = nn.Conv2d(
            2, original_conv1.out_channels, 
            kernel_size=original_conv1.kernel_size, stride=original_conv1.stride,
            padding=original_conv1.padding, bias=(original_conv1.bias is not None)
        )
        with torch.no_grad():
            model.conv_block1.conv1.weight[:, 0:1, :, :].copy_(original_conv1.weight)
            model.conv_block1.conv1.weight[:, 1:2, :, :].zero_()
            
    return model

class AudioClassifierPANNsEarly(nn.Module):
    """PANNs-based Early Fusion model"""
    def __init__(self, num_labels, pretrained_path):
        super().__init__()
        self.backbone = get_panns_backbone(pretrained_path, modify_for_early=True)
        self.poi_processor = nn.Linear(Config.POI_EMBED_DIM, Config.MEL_BINS)
        
        # Internal extractors to handle raw waveform to log-mel
        dummy_cnn = Cnn14(sample_rate=Config.SAMPLE_RATE, window_size=Config.WINDOW_SIZE,
                          hop_size=Config.HOP_SIZE, mel_bins=Config.MEL_BINS, fmin=Config.FMIN,
                          fmax=Config.FMAX, classes_num=527)
        self.spectrogram_extractor = dummy_cnn.spectrogram_extractor
        self.logmel_extractor = dummy_cnn.logmel_extractor

        self.classifier = nn.Sequential(
            nn.Linear(Config.PANNS_EMBED_DIM, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        )

    def forward(self, waveforms, poi_embeddings):
        if waveforms.dim() == 3:
            waveforms = waveforms.squeeze(1)

        # 1. Feature extraction
        audio_map = self.logmel_extractor(self.spectrogram_extractor(waveforms))
        poi_feat = self.poi_processor(poi_embeddings).view(poi_embeddings.shape[0], 1, 1, -1)
        poi_map = poi_feat.expand(-1, -1, audio_map.shape[2], -1)

        # 2. Backbone preprocessing (Augmentation + BN0)
        x = self.backbone.spec_augmenter(audio_map)
        x = x.transpose(1, 3)
        x = self.backbone.bn0(x)
        x = x.transpose(1, 3)

        # 3. Concatenate and pass through blocks
        x = torch.cat((x, poi_map), dim=1)
        x = self.backbone.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.backbone.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.backbone.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.backbone.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = self.backbone.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = self.backbone.conv_block6(x, pool_size=(1, 1), pool_type='avg')

        x = torch.mean(x, dim=3)
        x = torch.max(x, dim=2)[0] + torch.mean(x, dim=2)
        embedding = self.backbone.fc1(x)

        return self.classifier(embedding)

class AudioClassifierPANNsLate(nn.Module):
    """PANNs-based Late Fusion model with learnable lambda weight"""
    def __init__(self, num_labels, pretrained_path):
        super().__init__()
        self.backbone = get_panns_backbone(pretrained_path)
        self.audio_head = nn.Linear(Config.PANNS_EMBED_DIM, num_labels)
        self.poi_head = nn.Sequential(
            nn.Linear(Config.POI_EMBED_DIM, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, num_labels)
        )
        self._lambda_raw = nn.Parameter(torch.zeros(num_labels))

    def get_lambda(self):
        return torch.nn.functional.softplus(self._lambda_raw)

    def forward(self, waveforms, poi_embeddings, return_aux=False):
        if waveforms.dim() == 3:
            waveforms = waveforms.squeeze(1)

        audio_embed = self.backbone(waveforms, None)['embedding']
        audio_logits = self.audio_head(audio_embed)
        poi_logits = self.poi_head(poi_embeddings)
        
        fused_logits = audio_logits + self.get_lambda() * poi_logits

        return (audio_logits, poi_logits, fused_logits) if return_aux else fused_logits

def train_one_epoch(model, loader, optimizer, criterion, is_late_fusion=False, aux_weight=0.0):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for waveforms, targets, poi_embeds in pbar:
        waveforms, targets, poi_embeds = waveforms.to(device), targets.to(device), poi_embeds.to(device)
        optimizer.zero_grad()

        if is_late_fusion and aux_weight > 0:
            a_logits, p_logits, f_logits = model(waveforms, poi_embeds, return_aux=True)
            loss_main = criterion(f_logits, targets)
            loss_audio = criterion(a_logits, targets)
            loss_reg = (torch.tanh(p_logits).pow(2)).mean()
            loss = loss_main + aux_weight * (loss_audio + 0.1 * loss_reg)
        else:
            outputs = model(waveforms, poi_embeds)
            loss = criterion(outputs, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        
        if is_late_fusion:
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'λ_avg': f'{model.get_lambda().mean():.4f}'})
        else:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, all_preds, all_targets = 0, [], []
    with torch.no_grad():
        for waveforms, targets, poi_embeds in tqdm(loader, desc="Evaluating", leave=False):
            waveforms, targets, poi_embeds = waveforms.to(device), targets.to(device), poi_embeds.to(device)
            outputs = model(waveforms, poi_embeds)
            total_loss += criterion(outputs, targets).item()
            all_preds.append(torch.sigmoid(outputs).cpu())
            all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    return {
        'loss': total_loss / len(loader),
        'f1': f1_score(all_targets, (all_preds > 0.5).float(), average='samples', zero_division=0),
        'map': average_precision_score(all_targets, all_preds, average='micro')
    }

def main():
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    experiments = {
        "early_fusion": {"class": AudioClassifierPANNsEarly, "is_late": False},
        "late_fusion": {"class": AudioClassifierPANNsLate, "is_late": True}
    }

    results_registry = []

    for run in range(1, 6):
        print(f"\n--- Starting Split {run} ---")
        split_path = os.path.join(config.BASE_DATA_DIR, f'split_{run}')
        with open(os.path.join(split_path, 'class_labels.json'), 'r') as f:
            labels = json.load(f)
        
        train_df = pd.read_csv(os.path.join(split_path, 'train.csv'))
        val_df = pd.read_csv(os.path.join(split_path, 'validation.csv'))
        test_df = pd.read_csv(os.path.join(split_path, 'test.csv'))

        for name, exp_cfg in experiments.items():
            print(f"Experiment: PANNs_{name}")
            
            train_loader = DataLoader(PANNs_POIDataset(train_df, config, labels, data_dir_override=split_path), 
                                      batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn_embedding_poi)
            val_loader = DataLoader(PANNs_POIDataset(val_df, config, labels, data_dir_override=split_path), 
                                    batch_size=config.BATCH_SIZE, collate_fn=collate_fn_embedding_poi)
            test_loader = DataLoader(PANNs_POIDataset(test_df, config, labels, data_dir_override=split_path), 
                                     batch_size=config.BATCH_SIZE, collate_fn=collate_fn_embedding_poi)

            model = exp_cfg["class"](num_labels=len(labels), pretrained_path=config.PRETRAINED_PATH).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
            criterion = nn.BCEWithLogitsLoss()
            scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

            best_f1, counter = 0.0, 0
            model_path = os.path.join(config.OUTPUT_DIR, f"panns_{name}_run_{run}.pth")

            for epoch in range(config.EPOCHS):
                train_loss = train_one_epoch(model, train_loader, optimizer, criterion, exp_cfg["is_late"], config.AUX_WEIGHT)
                val_metrics = evaluate(model, val_loader, criterion)
                scheduler.step()

                print(f"Epoch {epoch+1} | Val F1: {val_metrics['f1']:.4f}")

                if val_metrics['f1'] > best_f1:
                    best_f1 = val_metrics['f1']
                    torch.save(model.state_dict(), model_path)
                    counter = 0
                else:
                    counter += 1
                    if counter >= config.EARLY_STOP_PATIENCE: break

            # Test evaluation
            model.load_state_dict(torch.load(model_path))
            test_metrics = evaluate(model, test_loader, criterion)
            results_registry.append({'run': run, 'exp': name, **test_metrics})

    pd.DataFrame(results_registry).to_csv(os.path.join(config.OUTPUT_DIR, 'results_summary.csv'), index=False)

if __name__ == '__main__':
    main()