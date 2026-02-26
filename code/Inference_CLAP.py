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
import torch.nn.functional as F

from common_loader import CLAPDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'CLAP')))
try:
    from CLAPWrapper import CLAPWrapper
except ImportError:
    print("Error: Cannot import CLAPWrapper from 'CLAP' directory")
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
    OUTPUT_DIR = 'clap_zero_shot'
    CLAP_WEIGHTS = 'CLAP_weights_2022.pth'
    GEO_MAPPING_PATH = 'AudioSet-to-Geo-ATBench_mapping_table.json'
    AUDIOSET_LABELS_PATH = 'audioset_labels.txt'
    BATCH_SIZE = 16
    NUM_WORKERS = 0
    SAMPLE_RATE = 48000
    DURATION = 10
    CLAP_EMBED_DIM = 1024


# Collate function for CLAP embeddings
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


def get_clap_model(weights_path):
    """Get pre-trained CLAP model"""
    clap_wrapper = CLAPWrapper(weights_path, device)
    model = clap_wrapper.clap
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


class CLAPZeroShot(nn.Module):
    """CLAP Zero-Shot classifier - FINAL SIMPLIFIED IMPLEMENTATION"""

    def __init__(self, target_classes, clap_weights_path):
        super().__init__()

        # 1. 正确初始化 CLAPWrapper，指明我们需要处理文本和音频
        #    这是解决所有问题的关键！
        self.clap_wrapper = CLAPWrapper(clap_weights_path, device, audio_only=False)
        self.target_classes = target_classes

        # 2. 使用 Wrapper 提供的方法，直接获取文本特征
        print("Pre-computing text embeddings for target classes...")
        with torch.no_grad():
            text_prompts = [f"This is a sound of {c}" for c in self.target_classes]
            # Wrapper 内部会处理分词、编码和归一化，非常方便
            self.text_embeddings = self.clap_wrapper.get_text_embeddings(text_prompts)

        print("Text embeddings computed successfully.")

    def forward(self, waveforms):
        with torch.no_grad():
            waveforms = waveforms.to(device)

            # 1. 使用 Wrapper 提供的方法，直接获取音频特征
            #    注意：CLAPWrapper 的 get_audio_embeddings 需要一个文件路径列表，
            #    而不是张量。但我们的 DataLoader 已经加载了张量。
            #    因此，我们直接调用底层的 audio_encoder，这仍然是最高效的方式。
            audio_embeddings, _ = self.clap_wrapper.clap.audio_encoder(waveforms)

            # 手动归一化音频特征
            audio_embeddings_norm = F.normalize(audio_embeddings, p=2, dim=-1)

            # 2. 计算相似度 (矩阵乘法)
            similarity = torch.matmul(audio_embeddings_norm, self.text_embeddings.T)

            # 3. 使用 logit_scale 缩放
            scaled_logits = similarity * self.clap_wrapper.clap.logit_scale.exp()

            # 4. 转换为概率
            predictions = torch.sigmoid(scaled_logits)

            # 保留原始音频嵌入
            embedding = audio_embeddings

            # 创建虚拟张量以保持兼容性
            batch_size = waveforms.size(0)
            audioset_preds = torch.zeros(batch_size, 527, device=waveforms.device)

        return predictions, embedding, audioset_preds


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
        pbar = tqdm(dataloader, desc="CLAP Zero-Shot Inference")
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
                            title="CLAP Zero-Shot Audio Embeddings t-SNE"):
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
                        c=[colors[color_idx]], label=f'{cls}',
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
    if not os.path.exists(config.CLAP_WEIGHTS):
        print(f"Error: CLAP weights file {config.CLAP_WEIGHTS} not found")
        return

    audioset_labels = load_audioset_labels(config.AUDIOSET_LABELS_PATH)
    all_results = []

    # Zero-Shot evaluation - run on all data in split_1
    for run_num in range(1, 2):  # Only run on split_1
        print(f"\n{'=' * 50}\nCLAP Zero-Shot Evaluation on Split {run_num} (ALL DATA)\n{'=' * 50}")

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
            elif 'id' in combined_df.columns:
                combined_df = combined_df.drop_duplicates(subset=['id'], keep='first')
            print(f"Combined dataset: {len(combined_df)} unique samples")
        else:
            print("No data files found!")
            continue

        # Create dataset from ALL data
        all_dataset = CLAPDataset(combined_df, config, class_labels)
        all_loader = DataLoader(all_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                                num_workers=config.NUM_WORKERS, collate_fn=collate_fn_embedding)

        # Create Zero-Shot model
        model = CLAPZeroShot(
            target_classes,
            config.CLAP_WEIGHTS
        ).to(device)

        # Perform Zero-Shot inference on ALL data
        print(f"\nPerforming CLAP Zero-Shot inference on ALL {len(combined_df)} samples...")
        embeddings, predictions, audioset_preds, targets, class_names = extract_embeddings_and_predictions(
            model, all_loader, class_labels, target_classes)

        # Analyze AudioSet predictions
        analyze_top_audioset_predictions(audioset_preds, audioset_labels)

        # Evaluate performance on ALL data
        accuracy, f1_samples, f1_macro, f1_micro, map_score, class_f1_scores = evaluate_zero_shot(
            predictions, targets, target_classes)

        print(f"\n--- CLAP Zero-Shot Results for Split {run_num} (ALL DATA) ---")
        print(f"Total samples: {len(combined_df)}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 (samples): {f1_samples:.4f}")
        print(f"F1 (macro): {f1_macro:.4f}")
        print(f"F1 (micro): {f1_micro:.4f}")
        print(f"mAP: {map_score:.4f}")

        # Generate t-SNE visualization for ALL data
        print("\nGenerating CLAP t-SNE visualization for ALL data...")
        tsne_save_path = os.path.join(config.OUTPUT_DIR, f"clap_zero_shot_tsne_split_{run_num}_all_data.png")
        plot_tsne_visualization(embeddings, class_names, target_classes, tsne_save_path,
                                f"CLAP Zero-Shot Audio Embeddings t-SNE - Split {run_num} (ALL DATA)")

        # Record results
        result = {
            'run': run_num,
            'model': 'CLAP',
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
    results_path = os.path.join(config.OUTPUT_DIR, 'clap_zero_shot_results_all_data.csv')
    results_df.to_csv(results_path, index=False)

    print(f"\n{'=' * 60}\nCLAP ZERO-SHOT RESULTS (ALL DATA)\n{'=' * 60}")
    print(f"Total samples processed: {results_df['total_samples'].iloc[0]}")
    print(f"Accuracy: {results_df['accuracy'].iloc[0]:.4f}")
    print(f"F1 (samples): {results_df['f1_samples'].iloc[0]:.4f}")
    print(f"F1 (macro): {results_df['f1_macro'].iloc[0]:.4f}")
    print(f"F1 (micro): {results_df['f1_micro'].iloc[0]:.4f}")
    print(f"mAP: {results_df['map_score'].iloc[0]:.4f}")

    print(f"\nDetailed results saved to: {results_path}")


if __name__ == '__main__':
    main()
