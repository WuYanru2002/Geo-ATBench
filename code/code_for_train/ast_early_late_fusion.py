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
import timm
import numpy as np

# Local module imports
try:
    from models.ast_models import ASTModel
    from common_loader import ASTDataset, AST_POIDataset, collate_fn_embedding_poi
except ImportError as e:
    print(f"Error importing local modules: {e}")
    sys.exit(1)

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Config:
    """Standardized configuration for Audio-POI Fusion"""
    # Data Paths
    BASE_DATA_DIR = 'path/to/data_splits'
    AUDIO_DIR = 'path/to/raw_audio_files'
    POI_FEAT_DIR = 'path/to/poi_features'
    OUTPUT_DIR = 'path/to/output_results'

    # Hyperparameters
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    EPOCHS = 100
    LR = 1e-5
    WEIGHT_DECAY = 1e-4
    EARLY_STOP_PATIENCE = 15

    # Audio/Model Specs
    SAMPLE_RATE = 16000
    DURATION = 10
    N_MELS = 128
    INPUT_TDIM = 1024
    AST_EMBED_DIM = 768
    POI_EMBED_DIM = 768
    AUX_WEIGHT = 0.1

def verify_audio_files(df, audio_dir):
    """Checks if audio files exist at the specified path"""
    print(f"Verifying audio files in: {audio_dir}")
    missing_count = 0
    for idx, row in df.head(10).iterrows():
        filename = f"{row['id']}.wav" if not str(row['id']).endswith('.wav') else str(row['id'])
        if not os.path.exists(os.path.join(audio_dir, filename)):
            missing_count += 1
    if missing_count > 0:
        print(f"Warning: Found {missing_count} missing files in first 10 rows.")

class AudioClassifierASTEarly(nn.Module):
    """AST-based Early Fusion model"""
    def __init__(self, num_labels, finetune_ast=True):
        super().__init__()
        self.ast_model = ASTModel(label_dim=527, audioset_pretrain=True, model_size='base384')
        
        for param in self.ast_model.parameters():
            param.requires_grad = finetune_ast

        num_patches = 1212
        embed_dim = Config.AST_EMBED_DIM
        
        # Position embedding for [CLS], [POI], and patches
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 1 + num_patches, embed_dim))
        
        with torch.no_grad():
            original_pos = self.ast_model.v.pos_embed
            self.pos_embed[:, 0:1, :].copy_(original_pos[:, 0:1, :]) # CLS
            # Index 1 is reserved for POI token (initialized to zero)
            self.pos_embed[:, 2:, :].copy_(original_pos[:, 1:, :]) # Patches

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        )

    def forward(self, mel_db, poi_embeddings):
        B = mel_db.shape[0]
        if mel_db.dim() == 3:
            mel_db = mel_db.unsqueeze(1)

        x = self.ast_model.v.patch_embed(mel_db)
        cls_token = self.ast_model.v.cls_token.expand(B, -1, -1)
        poi_token = poi_embeddings.unsqueeze(1)

        x = torch.cat((cls_token, poi_token, x), dim=1)
        x = x + self.pos_embed
        x = self.ast_model.v.pos_drop(x)

        for blk in self.ast_model.v.blocks:
            x = blk(x)
        x = self.ast_model.v.norm(x)

        return self.classifier(x[:, 0, :])

class AudioClassifierASTLate(nn.Module):
    """AST-based Late Fusion model with learnable lambda weight"""
    def __init__(self, num_labels, finetune_ast=True):
        super().__init__()
        self.ast_model = ASTModel(label_dim=527, audioset_pretrain=True, model_size='base384')
        for param in self.ast_model.parameters():
            param.requires_grad = finetune_ast
            
        self.audio_head = nn.Linear(Config.AST_EMBED_DIM, num_labels)
        self.poi_head = nn.Sequential(
            nn.Linear(Config.POI_EMBED_DIM, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, num_labels)
        )
        self._lambda_raw = nn.Parameter(torch.zeros(num_labels))

    def get_lambda(self):
        return torch.nn.functional.softplus(self._lambda_raw)

    def forward(self, mel_db, poi_embeddings, return_aux=False):
        if mel_db.dim() == 3:
            mel_db = mel_db.unsqueeze(1)
        
        B = mel_db.shape[0]
        x = self.ast_model.v.patch_embed(mel_db)
        cls_tokens = self.ast_model.v.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = x + self.ast_model.v.pos_embed[:, :x.shape[1], :]
        x = self.ast_model.v.pos_drop(x)

        for blk in self.ast_model.v.blocks:
            x = blk(x)
        x = self.ast_model.v.norm(x)

        audio_logits = self.audio_head(x[:, 0, :])
        poi_logits = self.poi_head(poi_embeddings)
        
        fused_logits = audio_logits + self.get_lambda() * poi_logits

        return (audio_logits, poi_logits, fused_logits) if return_aux else fused_logits

def train_one_epoch(model, loader, optimizer, criterion, is_late_fusion=False, aux_weight=0.0):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for mel_db, targets, poi_embeds in pbar:
        mel_db, targets, poi_embeds = mel_db.to(device), targets.to(device), poi_embeds.to(device)
        optimizer.zero_grad()

        if is_late_fusion and aux_weight > 0:
            a_logits, p_logits, f_logits = model(mel_db, poi_embeds, return_aux=True)
            loss_main = criterion(f_logits, targets)
            loss_audio = criterion(a_logits, targets)
            loss_reg = (torch.tanh(p_logits).pow(2)).mean()
            loss = loss_main + aux_weight * (loss_audio + 0.1 * loss_reg)
        else:
            outputs = model(mel_db, poi_embeds)
            loss = criterion(outputs, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, all_preds, all_targets = 0, [], []
    
    with torch.no_grad():
        for mel_db, targets, poi_embeds in tqdm(loader, desc="Evaluating", leave=False):
            mel_db, targets, poi_embeds = mel_db.to(device), targets.to(device), poi_embeds.to(device)
            outputs = model(mel_db, poi_embeds)
            
            total_loss += criterion(outputs, targets).item()
            all_preds.append(torch.sigmoid(outputs).cpu())
            all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    metrics = {
        'loss': total_loss / len(loader),
        'f1': f1_score(all_targets, (all_preds > 0.5).float(), average='samples', zero_division=0),
        'map': average_precision_score(all_targets, all_preds, average='micro'),
        'acc': accuracy_score(all_targets, (all_preds > 0.5).float())
    }
    return metrics

def main():
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    experiments = {
        "early_fusion": {"class": AudioClassifierASTEarly, "is_late": False},
        "late_fusion": {"class": AudioClassifierASTLate, "is_late": True}
    }

    results_registry = []

    for run in range(1, 6):
        print(f"\n--- Starting Split {run} ---")
        split_path = os.path.join(config.BASE_DATA_DIR, f'split_{run}')
        
        with open(os.path.join(split_path, 'class_labels.json'), 'r') as f:
            labels = json.load(f)
        
        # Load datasets
        train_df = pd.read_csv(os.path.join(split_path, 'train.csv'))
        val_df = pd.read_csv(os.path.join(split_path, 'validation.csv'))
        test_df = pd.read_csv(os.path.join(split_path, 'test.csv'))

        if run == 1: verify_audio_files(train_df, config.AUDIO_DIR)

        for name, exp_cfg in experiments.items():
            print(f"Experiment: {name}")
            
            train_loader = DataLoader(AST_POIDataset(train_df, config, labels, data_dir_override=split_path), 
                                      batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn_embedding_poi)
            val_loader = DataLoader(AST_POIDataset(val_df, config, labels, data_dir_override=split_path), 
                                    batch_size=config.BATCH_SIZE, collate_fn=collate_fn_embedding_poi)
            test_loader = DataLoader(AST_POIDataset(test_df, config, labels, data_dir_override=split_path), 
                                     batch_size=config.BATCH_SIZE, collate_fn=collate_fn_embedding_poi)

            model = exp_cfg["class"](num_labels=len(labels)).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
            criterion = nn.BCEWithLogitsLoss()
            scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

            best_f1, counter = 0.0, 0
            model_path = os.path.join(config.OUTPUT_DIR, f"model_{name}_run_{run}.pth")

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

            # Final Test
            model.load_state_dict(torch.load(model_path))
            test_metrics = evaluate(model, test_loader, criterion)
            results_registry.append({'run': run, 'exp': name, **test_metrics})
            print(f"Test F1: {test_metrics['f1']:.4f}")

    # Export results
    pd.DataFrame(results_registry).to_csv(os.path.join(config.OUTPUT_DIR, 'final_metrics.csv'), index=False)

if __name__ == '__main__':
    main()