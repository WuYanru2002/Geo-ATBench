import os
import json
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

def create_splits(args):
    """
    Creates 5 independent stratified splits for cross-validation runs.
    Ensures singleton classes are merged into training sets to avoid splitting errors.
    """
    print(f"Loading raw data from: {args.json_path}")
    with open(args.json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. Parse JSON and extract labels
    rows = []
    all_labels_set = set()
    for item in data:
        # Extract unique labels per file and sort them
        labels = sorted(list(set([seg['label'] for seg in item.get('segments', [])])))
        if labels:
            # Standardized column names: 'id' and 'class_name'
            rows.append({'id': item['id'], 'class_name': ','.join(labels)})
            all_labels_set.update(labels)

    df = pd.DataFrame(rows)
    all_labels = sorted(list(all_labels_set))
    num_classes = len(all_labels)
    print(f"Detected {num_classes} unique classes across {len(df)} samples.")

    # 2. Multi-label Binarization for stratification analysis
    mlb = MultiLabelBinarizer(classes=all_labels)
    y = mlb.fit_transform([s.split(',') for s in df['class_name']])

    # 3. Handle Singleton Classes (classes with only 1 sample)
    # Singletons must go to the training set as they cannot be split
    class_counts = pd.Series(np.sum(y, axis=0), index=all_labels)
    singleton_classes = class_counts[class_counts == 1].index.tolist()

    if singleton_classes:
        print(f"Warning: {len(singleton_classes)} singleton classes detected: {singleton_classes}")
        
        singleton_indices = np.any(mlb.transform([s.split(',') for s in df['class_name']])[:, 
                                  [all_labels.index(c) for c in singleton_classes]], axis=1)
        
        singleton_df = df[singleton_indices]
        multi_sample_df = df[~singleton_indices]
        print(f"Singleton samples: {len(singleton_df)} | Multi-sample entries: {len(multi_sample_df)}")
    else:
        singleton_df = pd.DataFrame()
        multi_sample_df = df

    # 4. Generate 5 Independent Splits
    for run_num in range(1, 6):
        print(f"\n--- Generating Split {run_num}/5 ---")
        
        # Define standardized output path
        split_dir = os.path.join(args.output_dir, f'split_{run_num}')
        os.makedirs(split_dir, exist_ok=True)

        # Unique seed for each run to ensure different distributions
        current_seed = 42 + (run_num - 1) * 1000

        # Perform stratification on multi-sample data
        y_multi = mlb.fit_transform([s.split(',') for s in multi_sample_df['class_name']])
        # Simplistic stratification based on the first label (common practice for multi-label split)
        stratify_labels = np.argmax(y_multi, axis=1)

        # Split Test Set
        remaining_df, test_df = train_test_split(
            multi_sample_df, 
            test_size=args.test_size, 
            stratify=stratify_labels, 
            random_state=current_seed
        )

        # Recalculate stratification for the remaining Train/Val split
        y_rem = mlb.fit_transform([s.split(',') for s in remaining_df['class_name']])
        stratify_rem = np.argmax(y_rem, axis=1)

        # Split Train and Validation
        val_ratio_adj = args.val_size / (1 - args.test_size)
        train_df, val_df = train_test_split(
            remaining_df, 
            test_size=val_ratio_adj, 
            stratify=stratify_rem, 
            random_state=current_seed
        )

        # Merge singletons back into the training set
        if not singleton_df.empty:
            train_df = pd.concat([train_df, singleton_df], ignore_index=True)
            # Shuffle training set to mix singletons
            train_df = train_df.sample(frac=1, random_state=current_seed).reset_index(drop=True)

        # 5. Export Standardized Files
        train_df.to_csv(os.path.join(split_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(split_dir, 'validation.csv'), index=False)
        test_df.to_csv(os.path.join(split_dir, 'test.csv'), index=False)

        # Export class labels for consistency across the project
        with open(os.path.join(split_dir, 'class_labels.json'), 'w', encoding='utf-8') as f:
            json.dump(all_labels, f, indent=4)

        print(f"Run {run_num} saved to {split_dir}")
        print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    print("\n[Status] All splits generated successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-label Dataset Stratification for 5-Run Experiments")
    parser.add_argument('--json_path', type=str, default='path/to/source.json', help='Path to source JSON')
    parser.add_argument('--output_dir', type=str, default='data_splits', help='Output base directory')
    parser.add_argument('--test_size', type=float, default=0.15, help='Test set ratio')
    parser.add_argument('--val_size', type=float, default=0.15, help='Validation set ratio')
    
    args = parser.parse_args()

    if args.test_size + args.val_size >= 1.0:
        raise ValueError("Combined test and validation size must be less than 1.0")

    create_splits(args)