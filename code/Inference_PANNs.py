import torch
import torch.nn as nn
import os
import sys
import warnings
import pandas as pd
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.font_manager as fm

# Import your custom loader
from common_loader import PANNsDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'audioset_tagging_cnn')))
try:
    from audioset_tagging_cnn.pytorch.models import Cnn14
except ImportError:
    print("Cannot import PANNs")
    sys.exit(1)

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set font for plots
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


class Config:
    BASE_DATA_DIR = 'data'
    AUDIO_DIR = 'audio_files'
    OUTPUT_DIR = 'panns_zero_shot'
    PRETRAINED_PTH_PATH = 'Cnn14_mAP=0.431.pth'
    GEO_MAPPING_PATH = 'AudioSet-to-Geo-ATBench_mapping_table.json'
    AUDIOSET_LABELS_PATH = 'audioset_labels.txt'
    BATCH_SIZE = 16
    NUM_WORKERS = 0
    SAMPLE_RATE = 32000
    WINDOW_SIZE = 1024
    HOP_SIZE = 320
    MEL_BINS = 64
    FMIN = 50
    FMAX = 14000
    PANNS_EMBED_DIM = 2048


# Add missing collate function
def collate_fn_embedding(batch):
    """
    Audio-only dataset collate function, handling (waveform, target) pairs
    """
    waveforms, targets = zip(*batch)
    waveforms_stacked = torch.stack(waveforms, 0)
    targets_stacked = torch.stack(targets, 0)
    return waveforms_stacked, targets_stacked


def load_audioset_labels(labels_file):
    """Load AudioSet label list"""
    audioset_labels = {}
    with open(labels_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            label = line.strip()
            audioset_labels[idx] = label
    return audioset_labels


def create_label_mapping(geo_mapping_path, audioset_labels):
    """Create mapping from geo labels to AudioSet indices"""
    with open(geo_mapping_path, 'r', encoding='utf-8') as f:
        geo_mapping = json.load(f)

    # Create reverse mapping from AudioSet labels to indices
    audioset_label_to_idx = {label: idx for idx, label in audioset_labels.items()}

    # Create mapping from geo labels to AudioSet indices
    geo_to_audioset_idx = {}

    for geo_label, audioset_matches in geo_mapping.items():
        indices = []

        if isinstance(audioset_matches, str):
            # Single match
            if audioset_matches in audioset_label_to_idx:
                indices.append(audioset_label_to_idx[audioset_matches])
        elif isinstance(audioset_matches, list):
            # Multiple matches
            for match in audioset_matches:
                if match in audioset_label_to_idx:
                    indices.append(audioset_label_to_idx[match])

        geo_to_audioset_idx[geo_label] = indices

        # Print mapping info for debugging
        if indices:
            print(f"Mapping '{geo_label}' -> {indices} ({[audioset_labels[i] for i in indices]})")
        else:
            print(f"Warning: No AudioSet labels found for '{geo_label}'")

    return geo_to_audioset_idx


def get_panns_model(pretrained_path):
    """Get pre-trained PANNs model"""
    model = Cnn14(
        sample_rate=Config.SAMPLE_RATE,
        window_size=Config.WINDOW_SIZE,
        hop_size=Config.HOP_SIZE,
        mel_bins=Config.MEL_BINS,
        fmin=Config.FMIN,
        fmax=Config.FMAX,
        classes_num=527
    )

    checkpoint = torch.load(pretrained_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model


class PANNsZeroShot(nn.Module):
    """PANNs Zero-Shot classifier"""

    def __init__(self, pretrained_path, target_classes, geo_mapping_path, audioset_labels_path):
        super().__init__()
        self.panns_model = get_panns_model(pretrained_path)
        self.target_classes = target_classes

        # Load AudioSet labels
        self.audioset_labels = load_audioset_labels(audioset_labels_path)
        print(f"Loaded {len(self.audioset_labels)} AudioSet labels")

        # Create label mapping
        self.geo_to_audioset_mapping = create_label_mapping(geo_mapping_path, self.audioset_labels)

        # Create mapping from target classes to AudioSet indices
        self.class_to_indices = {}
        for class_name in target_classes:
            if class_name in self.geo_to_audioset_mapping:
                self.class_to_indices[class_name] = self.geo_to_audioset_mapping[class_name]
            else:
                print(f"Warning: Target class '{class_name}' not found in mapping file")
                self.class_to_indices[class_name] = []

    def forward(self, waveforms):
        if waveforms.dim() == 3:
            waveforms = waveforms.squeeze(1)

        with torch.no_grad():
            panns_output = self.panns_model(waveforms, None)
            clipwise_output = panns_output['clipwise_output']  # [B, 527]
            embedding = panns_output['embedding']  # [B, 2048]

            # Apply sigmoid activation
            audioset_probs = torch.sigmoid(clipwise_output)

            # Map to target classes
            target_predictions = self.map_to_target_classes(audioset_probs)

        return target_predictions, embedding, audioset_probs

    def map_to_target_classes(self, audioset_predictions):
        """Map AudioSet predictions to target classes"""
        batch_size = audioset_predictions.size(0)
        num_target_classes = len(self.target_classes)
        target_preds = torch.zeros(batch_size, num_target_classes, device=audioset_predictions.device)

        class_to_idx = {name: idx for idx, name in enumerate(self.target_classes)}

        for class_name, audioset_indices in self.class_to_indices.items():
            if audioset_indices and class_name in class_to_idx:
                class_idx = class_to_idx[class_name]
                # Take max of corresponding AudioSet class predictions
                target_preds[:, class_idx] = audioset_predictions[:, audioset_indices].max(dim=1)[0]

        return target_preds


def extract_embeddings_and_predictions(model, dataloader, class_labels, target_classes):
    """Extract embeddings, predictions, and true labels, return mapped class names"""
    model.eval()
    embeddings_list = []
    predictions_list = []
    audioset_preds_list = []
    targets_list = []
    class_names_list = []

    # Handle whether class_labels is list or dict
    if isinstance(class_labels, dict):
        idx_to_class = {idx: name for name, idx in class_labels.items()}
    else:
        # If list, create index to class name mapping
        idx_to_class = {idx: name for idx, name in enumerate(class_labels)}

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Zero-Shot Inference")
        for waveforms, targets in pbar:
            waveforms = waveforms.to(device)
            targets = targets.to(device)

            predictions, embeddings, audioset_preds = model(waveforms)

            embeddings_list.append(embeddings.cpu().numpy())
            predictions_list.append(predictions.cpu().numpy())
            audioset_preds_list.append(audioset_preds.cpu().numpy())
            targets_list.append(targets.cpu().numpy())

            # Record class names - now use prediction results to determine dominant class
            predictions_np = predictions.cpu().numpy()
            targets_np = targets.cpu().numpy()

            for i, (pred_vector, target_vector) in enumerate(zip(predictions_np, targets_np)):
                # Prioritize using true labels
                active_classes = np.where(target_vector == 1)[0]
                if len(active_classes) > 0:
                    # If multiple labels, choose the one with highest prediction probability
                    if len(active_classes) > 1:
                        best_class_idx = active_classes[np.argmax(pred_vector[active_classes])]
                    else:
                        best_class_idx = active_classes[0]
                    class_names_list.append(target_classes[best_class_idx])
                else:
                    # No true labels, use highest predicted class
                    best_pred_idx = np.argmax(pred_vector)
                    class_names_list.append(target_classes[best_pred_idx])

    embeddings = np.vstack(embeddings_list)
    predictions = np.vstack(predictions_list)
    audioset_preds = np.vstack(audioset_preds_list)
    targets = np.vstack(targets_list)

    return embeddings, predictions, audioset_preds, targets, class_names_list


def analyze_top_audioset_predictions(audioset_preds, audioset_labels, top_k=10):
    """Analyze top AudioSet predictions"""
    # Calculate average prediction score for each class
    mean_scores = np.mean(audioset_preds, axis=0)

    # Find top-k classes
    top_indices = np.argsort(mean_scores)[-top_k:][::-1]

    print(f"\nTop {top_k} AudioSet class predictions:")
    for i, idx in enumerate(top_indices):
        print(f"{i + 1}. {audioset_labels[idx]}: {mean_scores[idx]:.4f}")


def plot_tsne_visualization(embeddings, class_names, target_classes, save_path,
                            title="PANNs Zero-Shot Audio Embeddings t-SNE"):
    """Plot t-SNE visualization, using target class labels"""
    print("Starting t-SNE computation...")

    if len(embeddings) > 2000:
        indices = np.random.choice(len(embeddings), 2000, replace=False)
        embeddings_sample = embeddings[indices]
        class_names_sample = [class_names[i] for i in indices]
    else:
        embeddings_sample = embeddings
        class_names_sample = class_names

    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_sample) - 1), n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings_sample)

    # Count classes
    from collections import Counter
    class_counter = Counter(class_names_sample)

    # Use all target classes that have samples
    available_classes = [cls for cls in target_classes if cls in class_counter and class_counter[cls] > 0]
    available_classes = sorted(available_classes, key=lambda x: class_counter[x], reverse=True)

    # Plot
    plt.figure(figsize=(18, 14))

    # Generate enough distinct colors for all classes
    if len(available_classes) <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
    else:
        # Use multiple color maps for more classes
        colors1 = plt.cm.tab20(np.linspace(0, 1, 20))
        colors2 = plt.cm.Set3(np.linspace(0, 1, 12))
        colors3 = plt.cm.Paired(np.linspace(0, 1, 12))
        colors = np.concatenate([colors1, colors2, colors3])

    for i, cls in enumerate(available_classes):
        mask = [cn == cls for cn in class_names_sample]
        if any(mask):
            indices = np.where(mask)[0]
            color_idx = i % len(colors)
            plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1],
                        c=[colors[color_idx]], label=f'{cls}',  # REMOVED THE COUNT HERE
                        alpha=0.7, s=50, edgecolors='black', linewidth=0.5)

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)

    # Single column legend with larger font
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14, ncol=1)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"t-SNE plot saved to: {save_path}")
    print(f"Class statistics: {dict(class_counter.most_common())}")


def evaluate_zero_shot(predictions, targets, class_names):
    """Evaluate Zero-Shot performance"""
    binary_preds = (predictions > 0.5).astype(float)

    # Overall performance
    f1_samples = f1_score(targets, binary_preds, average='samples', zero_division=0)
    f1_macro = f1_score(targets, binary_preds, average='macro', zero_division=0)
    f1_micro = f1_score(targets, binary_preds, average='micro', zero_division=0)
    map_score = average_precision_score(targets, predictions, average='micro')
    accuracy = accuracy_score(targets, binary_preds)

    # Per-class performance
    class_f1_scores = f1_score(targets, binary_preds, average=None, zero_division=0)

    print("\nPer-class F1 scores:")
    for i, (class_name, f1) in enumerate(zip(class_names, class_f1_scores)):
        print(f"{class_name}: {f1:.4f}")

    return accuracy, f1_samples, f1_macro, f1_micro, map_score, class_f1_scores


def main():
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Check if necessary files exist
    if not os.path.exists(config.GEO_MAPPING_PATH):
        print(f"Error: Mapping file {config.GEO_MAPPING_PATH} not found")
        return
    if not os.path.exists(config.AUDIOSET_LABELS_PATH):
        print(f"Error: AudioSet labels file {config.AUDIOSET_LABELS_PATH} not found")
        return

    all_results = []

    # Zero-Shot evaluation - modified to run on all data in split_1
    for run_num in range(1, 2):  # Only run on split_1
        print(f"\n{'=' * 50}\nPANNs Zero-Shot Evaluation on Split {run_num} (ALL DATA)\n{'=' * 50}")

        current_data_dir = os.path.join(config.BASE_DATA_DIR, f'split_{run_num}')

        # Load data
        class_labels_path = os.path.join(current_data_dir, 'class_labels.json')
        with open(class_labels_path, 'r', encoding='utf-8') as f:
            class_labels = json.load(f)

        # Handle whether class_labels is list or dict
        if isinstance(class_labels, dict):
            target_classes = list(class_labels.keys())
        else:
            target_classes = class_labels
            # If list, convert to dict format for Dataset
            class_labels = {name: idx for idx, name in enumerate(class_labels)}

        print(f"Target classes: {target_classes}")

        # Load ALL data from split_1: train + val + test
        all_dataframes = []

        # Load train set
        train_path = os.path.join(current_data_dir, 'train.csv')
        if os.path.exists(train_path):
            train_df = pd.read_csv(train_path)
            all_dataframes.append(train_df)
            print(f"Loaded train set: {len(train_df)} samples")

        # Load validation set
        val_path = os.path.join(current_data_dir, 'validation.csv')
        if os.path.exists(val_path):
            val_df = pd.read_csv(val_path)
            all_dataframes.append(val_df)
            print(f"Loaded validation set: {len(val_df)} samples")

        # Load test set
        test_path = os.path.join(current_data_dir, 'test.csv')
        if os.path.exists(test_path):
            test_df = pd.read_csv(test_path)
            all_dataframes.append(test_df)
            print(f"Loaded test set: {len(test_df)} samples")

        # Combine all dataframes
        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            # Remove potential duplicates based on filename/path
            if 'filename' in combined_df.columns:
                combined_df = combined_df.drop_duplicates(subset=['filename'], keep='first')
            elif 'path' in combined_df.columns:
                combined_df = combined_df.drop_duplicates(subset=['path'], keep='first')
            print(f"Combined dataset: {len(combined_df)} unique samples")
        else:
            print("No data files found!")
            continue

        # Create dataset from ALL data
        all_dataset = PANNsDataset(combined_df, config, class_labels)
        all_loader = DataLoader(all_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                                num_workers=config.NUM_WORKERS, collate_fn=collate_fn_embedding)

        # Create Zero-Shot model
        model = PANNsZeroShot(
            config.PRETRAINED_PTH_PATH,
            target_classes,
            config.GEO_MAPPING_PATH,
            config.AUDIOSET_LABELS_PATH
        ).to(device)

        # Perform Zero-Shot inference on ALL data
        print(f"\nPerforming Zero-Shot inference on ALL {len(combined_df)} samples...")
        embeddings, predictions, audioset_preds, targets, class_names = extract_embeddings_and_predictions(
            model, all_loader, class_labels, target_classes)

        # Analyze AudioSet predictions
        analyze_top_audioset_predictions(audioset_preds, model.audioset_labels)

        # Evaluate performance on ALL data
        accuracy, f1_samples, f1_macro, f1_micro, map_score, class_f1_scores = evaluate_zero_shot(
            predictions, targets, target_classes)

        print(f"\n--- Zero-Shot Results for Split {run_num} (ALL DATA) ---")
        print(f"Total samples: {len(combined_df)}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 (samples): {f1_samples:.4f}")
        print(f"F1 (macro): {f1_macro:.4f}")
        print(f"F1 (micro): {f1_micro:.4f}")
        print(f"mAP: {map_score:.4f}")

        # Generate t-SNE visualization for ALL data
        print("\nGenerating t-SNE visualization for ALL data...")
        tsne_save_path = os.path.join(config.OUTPUT_DIR, f"panns_zero_shot_tsne_split_{run_num}_all_data.png")
        plot_tsne_visualization(embeddings, class_names, target_classes, tsne_save_path,
                                f"PANNs Zero-Shot Audio Embeddings t-SNE - Split {run_num} (ALL DATA)")

        # Record results
        result = {
            'run': run_num,
            'data_type': 'all_data',
            'total_samples': len(combined_df),
            'accuracy': accuracy,
            'f1_samples': f1_samples,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'map_score': map_score
        }

        # Add per-class F1 scores
        for i, class_name in enumerate(target_classes):
            result[f'f1_{class_name}'] = class_f1_scores[i]

        all_results.append(result)

    # Save results
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(config.OUTPUT_DIR, 'zero_shot_results_all_data.csv')
    results_df.to_csv(results_path, index=False)

    print(f"\n{'=' * 60}\nPANNs ZERO-SHOT RESULTS (ALL DATA)\n{'=' * 60}")
    print(f"Total samples processed: {results_df['total_samples'].iloc[0]}")
    print(f"Accuracy: {results_df['accuracy'].iloc[0]:.4f}")
    print(f"F1 (samples): {results_df['f1_samples'].iloc[0]:.4f}")
    print(f"F1 (macro): {results_df['f1_macro'].iloc[0]:.4f}")
    print(f"F1 (micro): {results_df['f1_micro'].iloc[0]:.4f}")
    print(f"mAP: {results_df['map_score'].iloc[0]:.4f}")

    print(f"\nDetailed results saved to: {results_path}")


if __name__ == '__main__':
    main()
