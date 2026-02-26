import os
import json
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset


# ===================================================================
# Part 1: Audio-Only


class PANNsDataset(Dataset):
    """Loads raw waveform for PANNs models."""

    def __init__(self, df, config, class_labels, data_dir_override=None):
        self.df = df
        self.config = config
        self.audio_dir = config.AUDIO_DIR
        self.sample_rate = config.SAMPLE_RATE
        self.duration = 10
        self.target_len = self.sample_rate * self.duration

        self.all_labels = class_labels
        self.class_to_idx = {label: i for i, label in enumerate(self.all_labels)}
        self.num_classes = len(self.all_labels)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = os.path.join(self.audio_dir, f"{row['id']}.wav")
        try:
            wav, sr = torchaudio.load(audio_path)
            if wav.ndim > 1: wav = wav.mean(dim=0)
            if sr != self.sample_rate:
                wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.sample_rate)
            if wav.shape[0] < self.target_len:
                wav = torch.nn.functional.pad(wav, (0, self.target_len - wav.shape[0]))
            elif wav.shape[0] > self.target_len:
                wav = wav[:self.target_len]
        except Exception:
            wav = torch.zeros(self.target_len)

        waveform = wav.unsqueeze(0).float()

        labels = row['class_name'].split(',')
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        for label in labels:
            if label in self.class_to_idx:
                target[self.class_to_idx[label]] = 1.0
        return waveform, target


class ASTDataset(Dataset):
    """Loads Mel-spectrogram for AST models."""

    def __init__(self, df, config, class_labels, data_dir_override=None):
        self.df = df
        self.config = config
        self.audio_dir = config.AUDIO_DIR
        self.label_list = class_labels
        self.class_to_idx = {label: i for i, label in enumerate(self.label_list)}
        self.num_classes = len(self.label_list)
        self.max_length = config.SAMPLE_RATE * config.DURATION

        self.mel_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE, n_fft=1024, hop_length=160,
            n_mels=config.N_MELS, f_min=config.FMIN, f_max=config.FMAX
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = os.path.join(self.config.AUDIO_DIR, f"{row['id']}.wav")
        try:
            wav, sr = torchaudio.load(audio_path)
            if wav.ndim > 1: wav = wav.mean(dim=0)
            if sr != self.config.SAMPLE_RATE:
                wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.config.SAMPLE_RATE)
            if wav.shape[0] < self.max_length:
                wav = torch.nn.functional.pad(wav, (0, self.max_length - wav.shape[0]))
            elif wav.shape[0] > self.max_length:
                wav = wav[:self.max_length]
        except Exception:
            wav = torch.zeros(self.max_length)

        mel = self.mel_extractor(wav)
        mel_db = self.db_transform(mel)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

        if mel_db.shape[1] < self.config.INPUT_TDIM:
            mel_db = torch.nn.functional.pad(mel_db, (0, self.config.INPUT_TDIM - mel_db.shape[1]))
        elif mel_db.shape[1] > self.config.INPUT_TDIM:
            mel_db = mel_db[:, :self.config.INPUT_TDIM]
        mel_db = mel_db.unsqueeze(0)

        labels = row['class_name'].split(',')
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        for label in labels:
            if label in self.class_to_idx:
                target[self.class_to_idx[label]] = 1.0
        return mel_db, target


class CLAPDataset(Dataset):
    """Loads raw waveform for CLAP models."""

    def __init__(self, df, config, class_labels, data_dir_override=None):
        self.df = df
        self.config = config
        self.audio_dir = config.AUDIO_DIR
        self.label_list = class_labels
        self.class_to_idx = {label: i for i, label in enumerate(self.label_list)}
        self.num_classes = len(self.label_list)
        self.max_length = config.SAMPLE_RATE * config.DURATION

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = os.path.join(self.audio_dir, f"{row['id']}.wav")
        try:
            wav, sr = torchaudio.load(audio_path)
            if wav.ndim > 1: wav = wav.mean(dim=0)
            if sr != self.config.SAMPLE_RATE:
                wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.config.SAMPLE_RATE)
            if wav.shape[0] < self.max_length:
                wav = torch.nn.functional.pad(wav, (0, self.max_length - wav.shape[0]))
            elif wav.shape[0] > self.max_length:
                wav = wav[:self.max_length]
        except Exception:
            wav = torch.zeros(self.max_length)

        waveform = wav.float()
        labels = row['class_name'].split(',')
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        for label in labels:
            if label in self.class_to_idx:
                target[self.class_to_idx[label]] = 1.0
        return waveform, target


# ===================================================================
# art 2: Multimodal (Audio + POI) Datasets

def _load_poi_embedding(poi_feat_dir, audio_id, poi_embed_dim):
    poi_embedding_path = os.path.join(poi_feat_dir, f"{audio_id}_poi_cls_embeddings.npy")
    #print(f"POI Path: {poi_embedding_path}") 
    if os.path.exists(poi_embedding_path):
        try:
            poi_embedding = np.load(poi_embedding_path)
            #print(f"POI Loaded: Original Shape {poi_embedding.shape}, Non-zero ratio: {(poi_embedding != 0).mean():.2f}")

            if poi_embedding.ndim == 2:
                if poi_embedding.shape[1] != poi_embed_dim:
                    #print(f"POI Embed Dim Mismatch: Expected {poi_embed_dim} in dim=1, Got {poi_embedding.shape[1]}. Returning zeros.")
                    return np.zeros(poi_embed_dim, dtype=np.float32)
                if poi_embedding.shape[0] == 0:
                    #print("POI Empty Matrix. Returning zeros.")
                    return np.zeros(poi_embed_dim, dtype=np.float32)
                # 平均聚合
                poi_embedding = np.mean(poi_embedding, axis=0)
                #print(f"POI Aggregated (mean): New Shape {poi_embedding.shape}")
            elif poi_embedding.ndim == 1:
                if poi_embedding.shape[0] != poi_embed_dim:
                    #print(f"POI Dimension Mismatch: Expected {poi_embed_dim}, Got {poi_embedding.shape[0]}. Returning zeros.")
                    return np.zeros(poi_embed_dim, dtype=np.float32)
            else:

                return np.zeros(poi_embed_dim, dtype=np.float32)

            return poi_embedding
        except Exception as e:

            return np.zeros(poi_embed_dim, dtype=np.float32)
    else:
        return np.zeros(poi_embed_dim, dtype=np.float32)



class PANNs_POIDataset(PANNsDataset):

    def __init__(self, df, config, class_labels, data_dir_override=None):
        super().__init__(df, config, class_labels, data_dir_override=data_dir_override)
        self.poi_feat_dir = config.POI_FEAT_DIR
        self.poi_embed_dim = config.POI_EMBED_DIM

    def __getitem__(self, idx):
        # Get waveform and target from the parent class
        waveform, target = super().__getitem__(idx)

        # Load POI embedding
        row = self.df.iloc[idx]
        audio_id = str(row['id']) 
        poi_embedding_np = _load_poi_embedding(self.poi_feat_dir, audio_id, self.poi_embed_dim)
        poi_embedding = torch.tensor(poi_embedding_np, dtype=torch.float32)

        if idx < 5:
            audio_path = os.path.join(self.audio_dir, f"{audio_id}.wav")
            print(f"Sample {idx}: Audio Path: {audio_path}, Exists: {os.path.exists(audio_path)}")
            print(f"       Waveform Shape: {waveform.shape}, Non-zero ratio: {(waveform != 0).float().mean():.2f}")
            print(f"       POI Shape: {poi_embedding.shape}, Non-zero ratio: {(poi_embedding != 0).float().mean():.2f}")
            print(f"       Target Sum: {target.sum().item()} (non-zero labels)")

        return waveform, target, poi_embedding


class AST_POIDataset(ASTDataset):

    def __init__(self, df, config, class_labels, data_dir_override=None):
        super().__init__(df, config, class_labels, data_dir_override=data_dir_override)
        self.poi_feat_dir = config.POI_FEAT_DIR
        self.poi_embed_dim = config.POI_EMBED_DIM

    def __getitem__(self, idx):
        # Get mel spectrogram and target from the parent class
        mel_db, target = super().__getitem__(idx)

        # Load POI embedding
        row = self.df.iloc[idx]
        audio_id = str(row['id']) 
        poi_embedding_np = _load_poi_embedding(self.poi_feat_dir, audio_id, self.poi_embed_dim)
        poi_embedding = torch.tensor(poi_embedding_np, dtype=torch.float32)

        if idx < 5:
            audio_path = os.path.join(self.audio_dir, f"{audio_id}.wav")
            print(f"Sample {idx}: Audio Path: {audio_path}, Exists: {os.path.exists(audio_path)}")
            print(f"       Mel DB Shape: {mel_db.shape}, Non-zero ratio: {(mel_db != 0).float().mean():.2f}")
            print(f"       POI Shape: {poi_embedding.shape}, Non-zero ratio: {(poi_embedding != 0).float().mean():.2f}")
            print(f"       Target Sum: {target.sum().item()} (non-zero labels)")

        return mel_db, target, poi_embedding


class CLAP_POIDataset(CLAPDataset):

    def __init__(self, df, config, class_labels, data_dir_override=None):
        super().__init__(df, config, class_labels, data_dir_override=data_dir_override)
        self.poi_feat_dir = config.POI_FEAT_DIR
        self.poi_embed_dim = config.POI_EMBED_DIM

    def __getitem__(self, idx):
        # Get waveform and target from the parent class
        waveform, target = super().__getitem__(idx)

        # Load POI embedding
        row = self.df.iloc[idx]
        audio_id = str(row['id'])  
        poi_embedding_np = _load_poi_embedding(self.poi_feat_dir, audio_id, self.poi_embed_dim)
        poi_embedding = torch.tensor(poi_embedding_np, dtype=torch.float32)

        if idx < 5:
            audio_path = os.path.join(self.audio_dir, f"{audio_id}.wav")
            print(f"Sample {idx}: Audio Path: {audio_path}, Exists: {os.path.exists(audio_path)}")
            print(f"       Waveform Shape: {waveform.shape}, Non-zero ratio: {(waveform != 0).float().mean():.2f}")
            print(f"       POI Shape: {poi_embedding.shape}, Non-zero ratio: {(poi_embedding != 0).float().mean():.2f}")
            print(f"       Target Sum: {target.sum().item()} (non-zero labels)")

        return waveform, target, poi_embedding


# ===================================================================
# Part 3: Collate Function for Multimodal 

def collate_fn_embedding_poi(batch):
    audio_features, targets, poi_embeddings = zip(*batch)

    audio_features_stacked = torch.stack(audio_features, 0)
    targets_stacked = torch.stack(targets, 0)
    poi_embeddings_stacked = torch.stack(poi_embeddings, 0)

    return audio_features_stacked, targets_stacked, poi_embeddings_stacked
