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
import torchaudio.transforms as T

# Standardized imports for local modules
try:
    from common_loader import CLAPDataset, CLAP_POIDataset, collate_fn_embedding_poi
except ImportError as e:
    print(f"Error importing local modules: {e}")
    sys.exit(1)

# CLAP Wrapper setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'CLAP')))
try:
    from CLAPWrapper import CLAPWrapper
except ImportError as e:
    print(f"Error importing CLAPWrapper: {e}")
    sys.exit(1)

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Config:
    """Standardized configuration for CLAP-POI Fusion"""
    # Data Paths
    BASE_DATA_DIR = 'path/to/data_splits'
    AUDIO_DIR = 'path/to/raw_audio_files'
    POI_FEAT_DIR = 'path/to/poi_features'
    OUTPUT_DIR = 'path/to/output_results'
    CLAP_WEIGHTS = 'path/to/CLAP_weights_2022.pth'

    # Hyperparameters
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    EPOCHS = 100
    LR = 1e-5
    WEIGHT_DECAY = 1e-4
    EARLY_STOP_PATIENCE = 15

    # Audio/Model Specs
    SAMPLE_RATE = 48000
    DURATION = 10
    CLAP_EMBED_DIM = 1024
    POI_EMBED_DIM = 768
    MEL_BINS = 64
    AUX_WEIGHT = 0.1

class AudioClassifierCLAPEarly(nn.Module):
    """CLAP-based Early Fusion model"""
    def __init__(self, num_labels, clap_weights_path):
        super().__init__()
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE, n_fft=1024, win_length=1024,
            hop_length=320, n_mels=Config.MEL_BINS, f_min=50, f_max=14000
        )

        clap_wrapper = CLAPWrapper(clap_weights_path, device)
        self.audio_encoder = clap_wrapper.clap.audio_encoder

        # Modify input channels to 2 (Audio + POI map)
        self.audio_encoder.base.bn0 = nn.BatchNorm2d(2)
        original_conv1 = self.audio_encoder.base.conv_block1.conv1
        self.audio_encoder.base.conv_block1.conv1 = nn.Conv2d(
            in_channels=2,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=(original_conv1.bias is not None)
        )
        
        with torch.no_grad():
            self.audio_encoder.base.conv_block1.conv1.weight[:, 0:1, :, :].copy_(original_conv1.weight)
            self.audio_encoder.base.conv_block1.conv1.weight[:, 1:2, :, :].zero_()

        self.poi_processor = nn.Linear(Config.POI_EMBED_DIM, Config.MEL_BINS)
        self.classifier = nn.Sequential(
            nn.Linear(Config.CLAP_EMBED_DIM, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        )

    def forward(self, waveforms, poi_embeddings):
        mel_spec = self.mel_spectrogram(waveforms)
        audio_map = (mel_spec + 1e-6).log()

        # Process POI and expand to match audio time dimension
        poi_feat = self.poi_processor(poi_embeddings)
        poi_map = poi_feat.unsqueeze(2).expand(-1, -1, audio_map.shape[2])
        combined_map = torch.cat((audio_map.unsqueeze(1), poi_map.unsqueeze(1)), dim=1)

        # Backbone pass
        cnn14 = self.audio_encoder.base
        x = cnn14.bn0(combined_map)
        x = cnn14.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = cnn14.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = cnn14.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = cnn14.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = cnn14.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = cnn14.conv_block6(x, pool_size=(2, 2), pool_type='avg')

        x = torch.mean(x, dim=3)
        x = torch.max(x, dim=2)[0] + torch.mean(x, dim=2)
        audio_embed = self.audio_encoder.projection(cnn14.fc1(x))

        return self.classifier(audio_embed)

class AudioClassifierCLAPLate(nn.Module):
    """CLAP-based Late Fusion model with learnable lambda weight"""
    def __init__(self, num_labels, clap_weights_path):
        super().__init__()
        clap_wrapper = CLAPWrapper(clap_weights_path, device)
        self.backbone = clap_wrapper.clap.audio_encoder
        
        self.audio_head = nn.Linear(Config.CLAP_EMBED_DIM, num_labels)
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
        audio_embed, _ = self.backbone(waveforms)
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
        "early_fusion": {"class": AudioClassifierCLAPEarly, "is_late": False},
        "late_fusion": {"class": AudioClassifierCLAPLate, "is_late": True}
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
            print(f"Experiment: CLAP_{name}")
            
            train_loader = DataLoader(CLAP_POIDataset(train_df, config, labels, data_dir_override=split_path), 
                                      batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn_embedding_poi)
            val_loader = DataLoader(CLAP_POIDataset(val_df, config, labels, data_dir_override=split_path), 
                                    batch_size=config.BATCH_SIZE, collate_fn=collate_fn_embedding_poi)
            test_loader = DataLoader(CLAP_POIDataset(test_df, config, labels, data_dir_override=split_path), 
                                     batch_size=config.BATCH_SIZE, collate_fn=collate_fn_embedding_poi)

            model = exp_cfg["class"](num_labels=len(labels), clap_weights_path=config.CLAP_WEIGHTS).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
            criterion = nn.BCEWithLogitsLoss()
            scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

            best_f1, counter = 0.0, 0
            model_path = os.path.join(config.OUTPUT_DIR, f"clap_{name}_run_{run}.pth")

            for epoch in range(config.EPOCHS):
                train_loss = train_one_epoch(model, train_loader, optimizer, criterion, exp_cfg["is_late"], config.AUX_WEIGHT)
                val_metrics = evaluate(model, val_loader, criterion)
                scheduler.step()

                print(f"Epoch {epoch+1} | Val F1: {val_metrics['f1']:.4f} | Val mAP: {val_metrics['map']:.4f}")

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