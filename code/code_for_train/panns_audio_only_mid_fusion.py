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
import math

# Local module imports
try:
    from common_loader import PANNsDataset, PANNs_POIDataset, collate_fn_embedding_poi
except ImportError as e:
    print(f"Error importing local modules: {e}")
    sys.exit(1)

# PANNs (Cnn14) setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'audioset_tagging_cnn')))
try:
    from audioset_tagging_cnn.pytorch.models import Cnn14
except ImportError:
    print("Error: Could not import PANNs backbone. Check 'audioset_tagging_cnn' path.")
    sys.exit(1)

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_fn_audio_only(batch):
    """Collate function for (waveform, target) pairs"""
    waveforms = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    return waveforms, targets

class Config:
    """Standardized configuration for PANNs Mid-fusion"""
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

# ===================================================================
# 2. Mid-fusion Modules (Attention & Enhancers)
# ===================================================================

class POIProcessor(nn.Module):
    """Projects POI embeddings to match PANNs embedding dimension"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, (input_dim + output_dim) // 2), nn.ReLU(),
            nn.Linear((input_dim + output_dim) // 2, output_dim)
        )

    def forward(self, poi_embedding):
        return self.projection(poi_embedding)

class CustomAttention(nn.Module):
    """Basic Scaled Dot-Product Attention"""
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.softmax = nn.Softmax(dim=-1)
        self.scale = math.sqrt(embed_dim)

    def forward(self, query, key, value):
        query_r, key_r, value_r = query.unsqueeze(-1), key.unsqueeze(-1), value.unsqueeze(-1)
        attn_scores = torch.bmm(query_r, key_r.transpose(1, 2)) / self.scale
        attn_weights = self.softmax(attn_scores)
        output = torch.bmm(attn_weights, value_r)
        return output.squeeze(-1)

class FeatureEnhancer(nn.Module):
    """Transformer-style Cross-Attention block for feature enhancement"""
    def __init__(self, d_model, dropout_rate=0.1):
        super().__init__()
        self.cross_attn = CustomAttention(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.ReLU(),
            nn.Dropout(dropout_rate), nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, query, context):
        attn_output = self.cross_attn(query=query, key=context, value=context)
        x = self.norm1(query + self.dropout1(attn_output))
        ffn_output = self.ffn(x)
        return self.norm2(x + self.dropout2(ffn_output))

# ===================================================================
# 3. Model Definitions
# ===================================================================

def get_panns_backbone(pretrained_path):
    backbone = Cnn14(
        sample_rate=Config.SAMPLE_RATE, window_size=Config.WINDOW_SIZE,
        hop_size=Config.HOP_SIZE, mel_bins=Config.MEL_BINS, fmin=Config.FMIN,
        fmax=Config.FMAX, classes_num=527
    )
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    backbone.load_state_dict(checkpoint['model'], strict=False)
    return backbone

class AudioClassifierPANNsAudioOnly(nn.Module):
    """Baseline: Audio-only PANNs model"""
    def __init__(self, num_labels, pretrained_path):
        super().__init__()
        self.backbone = get_panns_backbone(pretrained_path)
        self.classifier = nn.Sequential(
            nn.Linear(Config.PANNS_EMBED_DIM, 512), 
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        )

    def forward(self, waveforms, poi_embeddings=None):
        if waveforms.dim() == 3:
            waveforms = waveforms.squeeze(1)
        audio_embed = self.backbone(waveforms, None)['embedding']
        return self.classifier(audio_embed)

class AudioClassifierPANNsMidFusion(nn.Module):
    """Experiment: Mid-fusion via Concat and Residual attention-enhanced streams"""
    def __init__(self, num_labels, pretrained_path):
        super().__init__()
        self.backbone = get_panns_backbone(pretrained_path)
        self.poi_processor = POIProcessor(Config.POI_EMBED_DIM, Config.PANNS_EMBED_DIM)
        self.enhancer_audio = FeatureEnhancer(d_model=Config.PANNS_EMBED_DIM)
        self.enhancer_poi = FeatureEnhancer(d_model=Config.PANNS_EMBED_DIM)
        
        self.fusion_projection = nn.Linear(Config.PANNS_EMBED_DIM * 2, Config.PANNS_EMBED_DIM)
        self.classifier = nn.Sequential(
            nn.Linear(Config.PANNS_EMBED_DIM, 512), 
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        )

    def forward(self, waveforms, poi_embeddings):
        if waveforms.dim() == 3:
            waveforms = waveforms.squeeze(1)
        
        audio_embed = self.backbone(waveforms, None)['embedding']
        poi_embed_projected = self.poi_processor(poi_embeddings)
        
        enhanced_audio = self.enhancer_audio(query=audio_embed, context=poi_embed_projected)
        enhanced_poi = self.enhancer_poi(query=poi_embed_projected, context=audio_embed)
        
        # Dual-stream additive residual fusion
        fused_stream_1 = enhanced_audio + poi_embed_projected
        fused_stream_2 = enhanced_poi + audio_embed
        
        concatenated_feat = torch.cat((fused_stream_1, fused_stream_2), dim=-1)
        projected_feat = self.fusion_projection(concatenated_feat)
        return self.classifier(projected_feat)

# ===================================================================
# 4. Training and Evaluation Functions
# ===================================================================

def train_one_epoch(model, loader, optimizer, criterion, use_poi):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="Training", leave=False)
    for data in pbar:
        optimizer.zero_grad()
        if use_poi:
            waveforms, targets, poi_embeds = data
            poi_embeds = poi_embeds.to(device)
        else:
            waveforms, targets = data
            poi_embeds = None
        
        waveforms, targets = waveforms.to(device), targets.to(device)
        outputs = model(waveforms, poi_embeds)
        loss = criterion(outputs, targets)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    return total_loss / len(loader)

def evaluate(model, loader, criterion, use_poi):
    model.eval()
    total_loss, all_preds, all_targets = 0.0, [], []
    with torch.no_grad():
        for data in tqdm(loader, desc="Evaluating", leave=False):
            if use_poi:
                waveforms, targets, poi_embeds = data
                poi_embeds = poi_embeds.to(device)
            else:
                waveforms, targets = data
                poi_embeds = None
            
            waveforms, targets = waveforms.to(device), targets.to(device)
            outputs = model(waveforms, poi_embeds)
            
            total_loss += criterion(outputs, targets).item()
            all_preds.append(torch.sigmoid(outputs).cpu())
            all_targets.append(targets.cpu())

    all_preds, all_targets = torch.cat(all_preds), torch.cat(all_targets)
    return {
        'loss': total_loss / len(loader),
        'f1': f1_score(all_targets, (all_preds > 0.5).float(), average='samples', zero_division=0),
        'map': average_precision_score(all_targets, all_preds, average='micro')
    }

def main():
    config = Config()
    
    # Standardized Debug Check
    print("\n" + "=" * 50)
    print("Verifying data loading environment...")
    split_1_path = os.path.join(config.BASE_DATA_DIR, 'split_1')
    if not os.path.exists(split_1_path):
        print(f"Error: Split 1 path not found at {split_1_path}")
        sys.exit(1)
    print("Environment check passed.")
    print("=" * 50 + "\n")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    experiments = {
        "audio_only": {"class": AudioClassifierPANNsAudioOnly, "poi": False},
        "mid_fusion": {"class": AudioClassifierPANNsMidFusion, "poi": True}
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
            DatasetClass = PANNs_POIDataset if exp_cfg['poi'] else PANNsDataset
            collate_fn = collate_fn_embedding_poi if exp_cfg['poi'] else collate_fn_audio_only

            train_loader = DataLoader(DatasetClass(train_df, config, labels), batch_size=config.BATCH_SIZE, 
                                      shuffle=True, collate_fn=collate_fn, drop_last=True)
            val_loader = DataLoader(DatasetClass(val_df, config, labels), batch_size=config.BATCH_SIZE, 
                                    collate_fn=collate_fn)
            test_loader = DataLoader(DatasetClass(test_df, config, labels), batch_size=config.BATCH_SIZE, 
                                     collate_fn=collate_fn)

            model = exp_cfg["class"](num_labels=len(labels), pretrained_path=config.PRETRAINED_PATH).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
            criterion = nn.BCEWithLogitsLoss()
            scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

            best_f1, counter = -1.0, 0
            model_path = os.path.join(config.OUTPUT_DIR, f"panns_{name}_run_{run}_best.pth")

            for epoch in range(config.EPOCHS):
                train_one_epoch(model, train_loader, optimizer, criterion, exp_cfg['poi'])
                metrics = evaluate(model, val_loader, criterion, exp_cfg['poi'])
                scheduler.step()

                print(f"Epoch {epoch+1:02d} | Val F1: {metrics['f1']:.4f}")

                if metrics['f1'] > best_f1:
                    best_f1 = metrics['f1']
                    torch.save(model.state_dict(), model_path)
                    counter = 0
                else:
                    counter += 1
                    if counter >= config.EARLY_STOP_PATIENCE: break

            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=device))
                test_results = evaluate(model, test_loader, criterion, exp_cfg['poi'])
                results_registry.append({'run': run, 'exp': name, **test_results})

    pd.DataFrame(results_registry).to_csv(os.path.join(config.OUTPUT_DIR, 'all_results.csv'), index=False)

if __name__ == '__main__':
    main()