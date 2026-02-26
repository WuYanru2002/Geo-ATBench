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
from torch.utils.data import DataLoader, Dataset
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class Config:
    RADII_TO_TEST = ['5m', '10m', '25m', '50m', '75m', '100m', '200m', '500m', '1000m']

    BASE_DATA_DIR = 'data_1'
    NUM_SPLITS = 5
    BASE_POI_FEAT_DIR = 'outputs/poi_features_by_radius'
    OUTPUT_DIR = 'poi_only_ablation'

    BATCH_SIZE = 64
    NUM_WORKERS = 0
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 15
    POI_EMBED_DIM = 768

class POIDataset(Dataset):
    def __init__(self, dataframe, config, class_labels, current_radius):
        self.df = dataframe
        self.class_labels = class_labels
        self.poi_feat_dir = os.path.join(config.BASE_POI_FEAT_DIR, f"poi_features_{current_radius}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_id = row['id']
        class_name = row['class_name']
        poi_feat_path = os.path.join(self.poi_feat_dir, f"{file_id}_poi_cls_embeddings.npy")
        poi_embedding = torch.from_numpy(np.load(poi_feat_path)).float().squeeze(0)
        target = torch.zeros(len(self.class_labels), dtype=torch.float)
        label_indices = [self.class_labels.index(label) for label in class_name.split(',') if
                         label in self.class_labels]
        if label_indices:
            target[label_indices] = 1.0
        return poi_embedding, target


def collate_fn_poi(batch):
    embeddings = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    embeddings_tensor = torch.stack(embeddings)
    targets_tensor = torch.stack(targets)
    return embeddings_tensor, targets_tensor



class POIClassifier(nn.Module):
    def __init__(self, num_labels, input_dim=Config.POI_EMBED_DIM):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_labels)
        )

    def forward(self, poi_embeddings):
        return self.classifier(poi_embeddings)


def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for poi_embeddings, targets in pbar:
        poi_embeddings, targets = poi_embeddings.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(poi_embeddings)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    return total_loss / len(dataloader)


def evaluate_epoch(model, dataloader, criterion, desc="Evaluating"):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=desc, leave=False)
        for poi_embeddings, targets in pbar:
            poi_embeddings, targets = poi_embeddings.to(device), targets.to(device)
            outputs = model(poi_embeddings)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
    avg_loss = total_loss / len(dataloader)
    all_preds, all_targets = torch.cat(all_preds), torch.cat(all_targets)
    binary_preds = (all_preds > 0.5).float()
    f1 = f1_score(all_targets, binary_preds, average='samples', zero_division=0)
    map_score = average_precision_score(all_targets, all_preds, average='micro')
    accuracy = accuracy_score(all_targets, binary_preds)
    return avg_loss, accuracy, f1, map_score



def main():
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    all_run_results = []

    for split_num in range(1, config.NUM_SPLITS + 1):
        print(
            f"\n{'=' * 70}\n{'=' * 20}  STARTING FOLD {split_num}/{config.NUM_SPLITS}  {'=' * 20}\n{'=' * 70}")

        current_split_dir = os.path.join(config.BASE_DATA_DIR, f'split_{split_num}')
        if not os.path.exists(current_split_dir):
            print(f"'{current_split_dir}' Not exists")
            continue

        print(f"Loading data from: {current_split_dir}")
        with open(os.path.join(current_split_dir, 'class_labels.json'), 'r', encoding='utf-8') as f:
            class_labels = json.load(f)
        num_classes = len(class_labels)

        train_df = pd.read_csv(os.path.join(current_split_dir, 'train.csv'))
        val_df = pd.read_csv(os.path.join(current_split_dir, 'validation.csv'))
        test_df = pd.read_csv(os.path.join(current_split_dir, 'test.csv'))

        for radius in config.RADII_TO_TEST:
            print(f"\n{'#' * 60}\n#  [Fold {split_num}] Processing POI RADIUS: {radius}  #\n{'#' * 60}")

            poi_feat_dir = os.path.join(config.BASE_POI_FEAT_DIR, f"poi_features_{radius}")
            if not os.path.exists(poi_feat_dir):
                print(f"{radius} : {poi_feat_dir}")
                continue

            train_dataset = POIDataset(train_df, config, class_labels, radius)
            val_dataset = POIDataset(val_df, config, class_labels, radius)
            test_dataset = POIDataset(test_df, config, class_labels, radius)

            train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                                      num_workers=config.NUM_WORKERS, collate_fn=collate_fn_poi)
            val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                                    num_workers=config.NUM_WORKERS, collate_fn=collate_fn_poi)
            test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                                     num_workers=config.NUM_WORKERS, collate_fn=collate_fn_poi)

            model = POIClassifier(num_labels=num_classes).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
            criterion = nn.BCEWithLogitsLoss()
            scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-7)

            best_val_f1 = -1.0
            early_stopping_counter = 0

            for epoch in range(config.EPOCHS):
                train_loss = train_epoch(model, train_loader, optimizer, criterion)
                val_loss, val_acc, val_f1, val_map = evaluate_epoch(model, val_loader, criterion,
                                                                    f"Validating [F{split_num}, R{radius}]")
                print(
                    f"Epoch {epoch + 1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val mAP: {val_map:.4f}")
                scheduler.step()

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    model_save_path = os.path.join(config.OUTPUT_DIR, f"poi_only_split{split_num}_{radius}_best.pth")
                    torch.save(model.state_dict(), model_save_path)
                    print(f"  -> Val F1 improved to {val_f1:.4f}. Model saved.")
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= config.EARLY_STOPPING_PATIENCE:
                        print(f"Early stopping triggered for [F{split_num}, R{radius}].")
                        break

            print(f"\n--- Evaluating [F{split_num}, R{radius}] on Test Set... ---")
            best_model_path = os.path.join(config.OUTPUT_DIR, f"poi_only_split{split_num}_{radius}_best.pth")
            if os.path.exists(best_model_path):
                model.load_state_dict(torch.load(best_model_path, map_location=device))
                test_loss, test_acc, test_f1, test_map = evaluate_epoch(model, test_loader, criterion,
                                                                        f"Testing [F{split_num}, R{radius}]")
                print(f"--- Test Results for [F{split_num}, R{radius}] ---\n"
                      f"Test F1: {test_f1:.4f}, Test mAP: {test_map:.4f}\n")

                result = {
                    'split': split_num, 'radius': radius, 'test_f1': test_f1, 'test_map': test_map,
                    'test_loss': test_loss, 'test_accuracy': test_acc, 'best_val_f1': best_val_f1
                }
                all_run_results.append(result)

    if not all_run_results:
        return

    detailed_results_df = pd.DataFrame(all_run_results)
    detailed_results_path = os.path.join(config.OUTPUT_DIR, 'all_splits_detailed_results.csv')
    detailed_results_df.to_csv(detailed_results_path, index=False)
    print(f"\nDetailed results for all runs saved to: {detailed_results_path}")

    agg_functions = {
        'test_f1': ['mean', 'std'],
        'test_map': ['mean', 'std'],
        'test_accuracy': ['mean', 'std']
    }
    aggregated_results = detailed_results_df.groupby('radius').agg(agg_functions).reset_index()

    aggregated_results.columns = ['_'.join(col).strip() for col in aggregated_results.columns.values]
    aggregated_results.rename(columns={'radius_': 'radius'}, inplace=True)

    aggregated_results['radius'] = pd.Categorical(aggregated_results['radius'], categories=config.RADII_TO_TEST,
                                                  ordered=True)
    aggregated_results = aggregated_results.sort_values('radius')

    final_results_path = os.path.join(config.OUTPUT_DIR, 'final_aggregated_results_mean_std.csv')
    aggregated_results.to_csv(final_results_path, index=False)

    print(f"\n{'=' * 70}\nFINAL AGGREGATED RESULTS\n{'=' * 70}")
    print(aggregated_results.to_string(index=False))
    print(f"\nAggregated results (mean/std) saved to: {final_results_path}")


if __name__ == '__main__':
    main()


