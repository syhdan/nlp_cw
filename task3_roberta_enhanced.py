"""
Enhanced RoBERTa PCL Classifier
================================

Combines notebook-style data loading with advanced training features:
- Uses DontPatronizeMe class for data loading
- Random downsampling (configurable ratio)
- Backtranslation augmentation option (multi-language)
- Class weighting: 1/sqrt(rp) and 1/sqrt(rn)
- Cosine schedule with warmup
- Gradient clipping
- Checkpointing and metrics tracking

Usage:
    # With downsampling (2:1 ratio like notebook)
    python task3_roberta_enhanced.py --downsample --downsample_ratio 2.0
    
    # With perfect balance
    python task3_roberta_enhanced.py --downsample --downsample_ratio 1.0
    
    # With augmentation
    python task3_roberta_enhanced.py --augment --num_augments 2
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
    MarianMTModel,
    MarianTokenizer
)
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)
from collections import Counter
import json
import argparse
import ast
import re
from tqdm import tqdm
import random
from urllib import request

# Set random seeds
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================================
# DOWNLOAD DONT PATRONIZE ME MODULE
# ============================================================================

def download_dpm_module():
    """Download the DontPatronizeMe module if not present"""
    if not os.path.exists('dont_patronize_me.py'):
        print("Downloading DontPatronizeMe module...")
        module_url = "https://raw.githubusercontent.com/Perez-AlmendrosC/dontpatronizeme/master/semeval-2022/dont_patronize_me.py"
        with request.urlopen(module_url) as f, open('dont_patronize_me.py', 'w') as outf:
            a = f.read()
            outf.write(a.decode('utf-8'))
        print("✓ Module downloaded")

# ============================================================================
# TEXT PREPROCESSING
# ============================================================================

def clean_text(text):
    """Clean and preprocess text"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def is_text_valid(text, min_tokens=5):
    """Check if text is valid (not too short)"""
    if pd.isna(text) or not isinstance(text, str):
        return False
    
    if not text or len(text.strip()) == 0:
        return False
    
    tokens = text.split()
    return len(tokens) >= min_tokens

# ============================================================================
# BACKTRANSLATION AUGMENTATION
# ============================================================================

class BackTranslationAugmenter:
    """Backtranslation augmentation using MarianMT with multiple language pairs"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.language_pairs = []
        
        # Define multiple language pairs for diversity
        pairs_to_load = [
            ('French', 'Helsinki-NLP/opus-mt-en-fr', 'Helsinki-NLP/opus-mt-fr-en'),
            ('German', 'Helsinki-NLP/opus-mt-en-de', 'Helsinki-NLP/opus-mt-de-en'),
        ]
        
        print("Loading backtranslation models for multiple languages...")
        
        for lang_name, en_to_lang, lang_to_en in pairs_to_load:
            try:
                print(f"  Loading EN->{lang_name}->EN...")
                
                tokenizer_out = MarianTokenizer.from_pretrained(en_to_lang)
                model_out = MarianMTModel.from_pretrained(en_to_lang).to(device)
                model_out.eval()
                
                tokenizer_back = MarianTokenizer.from_pretrained(lang_to_en)
                model_back = MarianMTModel.from_pretrained(lang_to_en).to(device)
                model_back.eval()
                
                self.language_pairs.append({
                    'name': lang_name,
                    'out_tokenizer': tokenizer_out,
                    'out_model': model_out,
                    'back_tokenizer': tokenizer_back,
                    'back_model': model_back
                })
                
                print(f"    ✓ {lang_name} models loaded")
            except Exception as e:
                print(f"    ⚠ Error loading {lang_name} models: {e}")
        
        if len(self.language_pairs) == 0:
            print("⚠ No translation models loaded successfully")
        else:
            print(f"\n✓ {len(self.language_pairs)} language pairs ready: {', '.join([p['name'] for p in self.language_pairs])}")
    
    def translate(self, text, src_tokenizer, src_model, max_length=512):
        """Translate text using a model"""
        if not text or len(text.strip()) == 0:
            return text
        
        try:
            inputs = src_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)
            
            with torch.no_grad():
                translated = src_model.generate(**inputs, max_length=max_length)
            
            translated_text = src_tokenizer.decode(translated[0], skip_special_tokens=True)
            return translated_text
        except Exception as e:
            print(f"Translation error: {e}")
            return text
    
    def backtranslate(self, text, lang_pair_idx=0):
        """Backtranslate: EN -> LANG -> EN using specified language pair"""
        if len(self.language_pairs) == 0:
            return text
        
        pair = self.language_pairs[lang_pair_idx % len(self.language_pairs)]
        
        # EN -> Target Language
        translated = self.translate(text, pair['out_tokenizer'], pair['out_model'])
        
        # Target Language -> EN
        back_text = self.translate(translated, pair['back_tokenizer'], pair['back_model'])
        
        return back_text
    
    def augment_batch(self, texts, lang_pair_idx=0, batch_size=8):
        """Augment a batch of texts using specified language pair"""
        if len(self.language_pairs) == 0:
            return texts
        
        pair_name = self.language_pairs[lang_pair_idx % len(self.language_pairs)]['name']
        augmented = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            for text in tqdm(batch, desc=f"{pair_name} - Batch {i//batch_size + 1}", leave=False):
                aug_text = self.backtranslate(text, lang_pair_idx)
                augmented.append(aug_text)
        
        return augmented

def augment_minority_class(df, augmenter, num_augments_per_sample=2, cache_file='augmented_data_cache.csv'):
    """Augment minority class (PCL) by creating N augmented samples per original
    
    Args:
        df: Original dataframe
        augmenter: BackTranslationAugmenter instance
        num_augments_per_sample: Number of augmented samples per PCL example
        cache_file: Path to cache file for augmented data
    
    Returns:
        DataFrame with augmented data
    """
    
    # Check if cached augmented data exists
    if os.path.exists(cache_file):
        print(f"\n{'='*80}")
        print(f"FOUND CACHED AUGMENTED DATA: {cache_file}")
        print(f"{'='*80}")
        
        try:
            cached_df = pd.read_csv(cache_file)
            
            # Verify cache has same configuration
            cache_info_file = cache_file.replace('.csv', '_info.json')
            if os.path.exists(cache_info_file):
                with open(cache_info_file, 'r') as f:
                    cache_info = json.load(f)
                
                if (cache_info.get('num_augments') == num_augments_per_sample and
                    cache_info.get('num_original_samples') == len(df)):
                    
                    print(f"✓ Cache is valid!")
                    print(f"  Cached samples: {len(cached_df)}")
                    print(f"  Non-PCL: {(cached_df['label']==0).sum()}")
                    print(f"  PCL:     {(cached_df['label']==1).sum()}")
                    print(f"\nUsing cached data (skipping augmentation)...")
                    return cached_df
                else:
                    print(f"⚠ Cache configuration mismatch - regenerating...")
            else:
                print(f"✓ Using cached augmented data")
                print(f"  Cached samples: {len(cached_df)}")
                return cached_df
                
        except Exception as e:
            print(f"⚠ Error loading cache: {e}")
            print(f"  Regenerating augmented data...")
    
    # Separate classes
    pcl_df = df[df['label'] == 1].copy()
    non_pcl_df = df[df['label'] == 0].copy()
    
    print(f"\nOriginal class distribution:")
    print(f"  Non-PCL: {len(non_pcl_df)}")
    print(f"  PCL:     {len(pcl_df)}")
    
    print(f"\nCreating {num_augments_per_sample} augmented samples per PCL example...")
    print(f"Total augmented samples to create: {len(pcl_df) * num_augments_per_sample}")
    
    # Create augmented data
    augmented_data = []
    
    # For each PCL sample, create num_augments_per_sample backtranslated versions
    # using DIFFERENT language pairs for diversity
    for aug_num in range(num_augments_per_sample):
        lang_pair_idx = aug_num
        pair_name = augmenter.language_pairs[lang_pair_idx % len(augmenter.language_pairs)]['name'] if augmenter.language_pairs else 'Unknown'
        
        print(f"\nAugmentation set {aug_num + 1}/{num_augments_per_sample} - Using {pair_name}...")
        augmented_texts = augmenter.augment_batch(pcl_df['text'].tolist(), lang_pair_idx=lang_pair_idx)
        
        for i, (idx, row) in enumerate(pcl_df.iterrows()):
            augmented_data.append({
                'par_id': f"aug_{aug_num}_{row['par_id']}",
                'text': augmented_texts[i],
                'label': 1,  # PCL
                'community': row.get('community', 'unknown')
            })
    
    # Create augmented dataframe
    aug_df = pd.DataFrame(augmented_data)
    
    # Combine with original
    combined_df = pd.concat([df, aug_df], ignore_index=True)
    
    print(f"\nAfter augmentation:")
    print(f"  Total samples: {len(combined_df)}")
    print(f"  Non-PCL: {(combined_df['label']==0).sum()}")
    print(f"  PCL:     {(combined_df['label']==1).sum()}")
    
    # Save to cache
    print(f"\nSaving augmented data to cache: {cache_file}")
    combined_df.to_csv(cache_file, index=False)
    
    # Save cache info
    cache_info = {
        'num_augments': num_augments_per_sample,
        'num_original_samples': len(df),
        'num_pcl_samples': len(pcl_df),
        'num_augmented_samples': len(aug_df),
        'total_samples': len(combined_df),
        'timestamp': pd.Timestamp.now().isoformat()
    }
    cache_info_file = cache_file.replace('.csv', '_info.json')
    with open(cache_info_file, 'w') as f:
        json.dump(cache_info, f, indent=2)
    
    print(f"✓ Cache saved successfully!")
    
    return combined_df

def downsample_majority_class(df, target_ratio=2.0):
    """Downsample majority class (Non-PCL) to balance dataset
    
    Args:
        df: DataFrame with 'label' column
        target_ratio: Desired ratio of Non-PCL to PCL (default 2.0 for 2:1 like notebook)
    
    Returns:
        DataFrame with downsampled majority class
    """
    # Separate classes
    pcl_df = df[df['label'] == 1].copy()
    non_pcl_df = df[df['label'] == 0].copy()
    
    print(f"\nOriginal class distribution:")
    print(f"  Non-PCL: {len(non_pcl_df)}")
    print(f"  PCL:     {len(pcl_df)}")
    
    # Calculate target non-PCL count
    target_non_pcl_count = int(len(pcl_df) * target_ratio)
    
    if target_non_pcl_count >= len(non_pcl_df):
        print(f"\nNo downsampling needed (target={target_non_pcl_count} >= current={len(non_pcl_df)})")
        return df
    
    print(f"\nDownsampling Non-PCL from {len(non_pcl_df)} to {target_non_pcl_count}...")
    print(f"  Strategy: Random sampling (not sequential like notebook)")
    
    # Randomly sample from majority class (better than notebook's sequential approach)
    non_pcl_downsampled = non_pcl_df.sample(n=target_non_pcl_count, random_state=SEED)
    
    # Combine with minority class
    combined_df = pd.concat([non_pcl_downsampled, pcl_df], ignore_index=True)
    
    # Shuffle the combined dataset
    combined_df = combined_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    print(f"\nAfter downsampling:")
    print(f"  Total samples: {len(combined_df)}")
    print(f"  Non-PCL: {(combined_df['label']==0).sum()}")
    print(f"  PCL:     {(combined_df['label']==1).sum()}")
    print(f"  Ratio (Non-PCL:PCL): {(combined_df['label']==0).sum()/(combined_df['label']==1).sum():.2f}:1")
    
    return combined_df

# ============================================================================
# DATASET
# ============================================================================

class PCLDataset(Dataset):
    """PCL Dataset"""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ============================================================================
# DATA LOADING (Notebook-style with DontPatronizeMe)
# ============================================================================

def load_pcl_data_from_dpm(augment=False, num_augments_per_sample=2, downsample=False, downsample_ratio=2.0, device='cpu', force_regenerate=False):
    """Load PCL dataset using DontPatronizeMe module (notebook-style)"""
    print("\n" + "="*80)
    print("LOADING PCL DATASET (Using DontPatronizeMe Module)")
    print("="*80)
    
    # Download module if needed
    download_dpm_module()
    
    # Import after download
    from dont_patronize_me import DontPatronizeMe
    
    # Initialize DPM
    dpm = DontPatronizeMe('.', '.')
    dpm.load_task1()
    
    # Load paragraph IDs
    train_ids = pd.read_csv('data/train_semeval_parids-labels.csv')
    dev_ids = pd.read_csv('data/dev_semeval_parids-labels.csv')
    
    train_ids.par_id = train_ids.par_id.astype(str)
    dev_ids.par_id = dev_ids.par_id.astype(str)
    
    data = dpm.train_task1_df
    
    print(f"Total paragraphs: {len(data)}")
    print(f"Train IDs: {len(train_ids)}")
    print(f"Dev IDs: {len(dev_ids)}")
    
    # Rebuild training set
    print("\nRebuilding training set...")
    train_rows = []
    for idx in range(len(train_ids)):
        parid = train_ids.par_id[idx]
        keyword = data.loc[data.par_id == parid].keyword.values[0]
        text = data.loc[data.par_id == parid].text.values[0]
        label = data.loc[data.par_id == parid].label.values[0]
        
        train_rows.append({
            'par_id': parid,
            'community': keyword,
            'text': text,
            'label': label
        })
    
    train_df = pd.DataFrame(train_rows)
    
    # Rebuild dev set
    print("Rebuilding dev set...")
    dev_rows = []
    for idx in range(len(dev_ids)):
        parid = dev_ids.par_id[idx]
        keyword = data.loc[data.par_id == parid].keyword.values[0]
        text = data.loc[data.par_id == parid].text.values[0]
        label = data.loc[data.par_id == parid].label.values[0]
        
        dev_rows.append({
            'par_id': parid,
            'community': keyword,
            'text': text,
            'label': label
        })
    
    dev_df = pd.DataFrame(dev_rows)
    
    # Clean text
    print("\nCleaning text...")
    train_df['text'] = train_df['text'].apply(clean_text)
    dev_df['text'] = dev_df['text'].apply(clean_text)
    
    # Filter short texts
    print("Filtering short texts (< 5 tokens)...")
    train_df = train_df[train_df['text'].apply(is_text_valid)].reset_index(drop=True)
    dev_df = dev_df[dev_df['text'].apply(is_text_valid)].reset_index(drop=True)
    
    print(f"\n✓ Train set: {len(train_df)} samples")
    print(f"✓ Dev set: {len(dev_df)} samples")
    
    print(f"\nClass distribution (train):") 
    print(f"  Non-PCL: {(train_df['label']==0).sum()} ({(train_df['label']==0).sum()/len(train_df)*100:.1f}%)")
    print(f"  PCL:     {(train_df['label']==1).sum()} ({(train_df['label']==1).sum()/len(train_df)*100:.1f}%)")
    
    # Data balancing strategies (can do both!)
    if augment:
        print("\n" + "="*80)
        print("DATA AUGMENTATION (BACKTRANSLATION)")
        print("="*80)
        
        # Generate cache filename based on num_augments
        cache_file = f'augmented_data_cache_x{num_augments_per_sample}.csv'
        
        # Delete cache if force regeneration
        if force_regenerate and os.path.exists(cache_file):
            print(f"\n⚠ Force regeneration enabled - deleting cache: {cache_file}")
            os.remove(cache_file)
            cache_info_file = cache_file.replace('.csv', '_info.json')
            if os.path.exists(cache_info_file):
                os.remove(cache_info_file)
        
        augmenter = BackTranslationAugmenter(device=device)
        train_df = augment_minority_class(train_df, augmenter, num_augments_per_sample=num_augments_per_sample, cache_file=cache_file)
    
    if downsample:
        print("\n" + "="*80)
        print("DOWNSAMPLING MAJORITY CLASS (Enhanced vs Notebook)")
        print("="*80)
        print("Notebook approach: Sequential sampling (first N samples)")
        print("This approach:     Random sampling (better generalization)")
        train_df = downsample_majority_class(train_df, target_ratio=downsample_ratio)
    
    return train_df, dev_df

# ============================================================================
# TRAINING
# ============================================================================

def train_model(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs,
    learning_rate,
    warmup_ratio,
    class_weights,
    accumulation_steps=1,
    use_rdrop=False,
    rdrop_alpha=1.0,
    weight_decay=0.001
):
    """Training loop with weighted loss, gradient accumulation, and optional R-Drop"""
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Scheduler
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Loss function with class weights
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    best_f1 = 0
    history = {
        'train_loss': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': [],
        'learning_rate': []
    }
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        train_pbar = tqdm(train_loader, desc='Training', leave=False)
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            if use_rdrop:
                # R-Drop: Two forward passes with different dropout
                outputs1 = model(input_ids=input_ids, attention_mask=attention_mask)
                outputs2 = model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Classification loss (average of both)
                ce_loss = (criterion(outputs1.logits, labels) + criterion(outputs2.logits, labels)) / 2
                
                # R-Drop KL divergence loss
                kl_loss = compute_kl_loss(outputs1.logits, outputs2.logits)
                
                # Combined loss
                loss = ce_loss + rdrop_alpha * kl_loss
            else:
                # Standard training
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
            
            # Scale loss by accumulation steps
            loss = loss / accumulation_steps
            loss.backward()
            
            # Only update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Clear cache to prevent memory buildup
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            total_loss += loss.item() * accumulation_steps  # Unscale for logging
            
            current_lr = scheduler.get_last_lr()[0]
            train_pbar.set_postfix({
                'loss': f'{loss.item() * accumulation_steps:.4f}',
                'lr': f'{current_lr:.2e}'
            })
        
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc='Validating', leave=False)
            for batch in val_pbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        f1 = f1_score(all_labels, all_preds, average='binary')
        precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
        current_lr = scheduler.get_last_lr()[0]
        
        history['train_loss'].append(avg_loss)
        history['val_f1'].append(f1)
        history['val_precision'].append(precision)
        history['val_recall'].append(recall)
        history['learning_rate'].append(current_lr)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Val F1:     {f1:.4f}")
        print(f"  Precision:  {precision:.4f}")
        print(f"  Recall:     {recall:.4f}")
        print(f"  Learn Rate: {current_lr:.2e}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_dir = 'checkpoints_enhanced'
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(
                model.state_dict(),
                f'{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pt'
            )
            print(f"  ✓ Checkpoint saved: epoch_{epoch+1}")
        
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), 'best_model_enhanced.pt')
            print(f"  ✓ New best F1: {best_f1:.4f}")
    
    return history, best_f1

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, test_loader, device):
    """Evaluate model"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        eval_pbar = tqdm(test_loader, desc='Evaluating')
        for batch in eval_pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print("\nClassification Report:")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=['Non-PCL', 'PCL'],
        digits=4
    ))
    
    f1 = f1_score(all_labels, all_preds, average='binary')
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    accuracy = accuracy_score(all_labels, all_preds)
    
    print(f"\nKey Metrics:")
    print(f"  F1:        {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print("="*80)
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy
    }

# ============================================================================
# R-DROP REGULARIZATION
# ============================================================================

def compute_kl_loss(p_logits, q_logits):
    """Compute KL divergence loss between two distributions
    
    Args:
        p_logits: First set of logits
        q_logits: Second set of logits
    
    Returns:
        Symmetric KL divergence loss
    """
    import torch.nn.functional as F
    
    p_loss = F.kl_div(
        F.log_softmax(p_logits, dim=-1),
        F.softmax(q_logits, dim=-1),
        reduction='batchmean'
    )
    q_loss = F.kl_div(
        F.log_softmax(q_logits, dim=-1),
        F.softmax(p_logits, dim=-1),
        reduction='batchmean'
    )
    
    # Symmetric KL divergence
    return (p_loss + q_loss) / 2

# ============================================================================
# LAYER FREEZING
# ============================================================================

def freeze_model_layers(model, freeze_embeddings=False, freeze_layers=0):
    """Freeze layers in RoBERTa model
    
    Args:
        model: RoBERTa model
        freeze_embeddings: Whether to freeze embedding layer
        freeze_layers: Number of bottom transformer layers to freeze (0-24)
    
    Returns:
        Number of frozen parameters
    """
    frozen_params = 0
    total_params = sum(p.numel() for p in model.parameters())
    
    print("\n" + "="*80)
    print("LAYER FREEZING")
    print("="*80)
    
    # Freeze embeddings
    if freeze_embeddings:
        print("\n❄️  Freezing embeddings layer...")
        for param in model.roberta.embeddings.parameters():
            param.requires_grad = False
            frozen_params += param.numel()
        print(f"   ✓ Embeddings frozen")
    
    # Freeze transformer layers
    if freeze_layers > 0:
        print(f"\n❄️  Freezing bottom {freeze_layers} transformer layers...")
        
        # RoBERTa-base has 24 layers (0-23)
        total_layers = len(model.roberta.encoder.layer)
        freeze_layers = min(freeze_layers, total_layers)
        
        for i in range(freeze_layers):
            for param in model.roberta.encoder.layer[i].parameters():
                param.requires_grad = False
                frozen_params += param.numel()
        
        print(f"   ✓ Layers 0-{freeze_layers-1} frozen (out of {total_layers} total)")
        print(f"   🔥 Layers {freeze_layers}-{total_layers-1} + classification head remain trainable")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nParameter summary:")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Frozen parameters:    {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print("="*80)
    
    return frozen_params

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (per GPU step)')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Gradient accumulation steps (1 = no accumulation)')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--warmup', type=float, default=0.2, help='Warmup ratio')
    parser.add_argument('--max_length', type=int, default=256, help='Max sequence length')
    parser.add_argument('--augment', action='store_true', help='Use backtranslation augmentation')
    parser.add_argument('--num_augments', type=int, default=1, help='Number of augmented samples per PCL example')
    parser.add_argument('--downsample', action='store_true', help='Downsample majority class')
    parser.add_argument('--downsample_ratio', type=float, default=2.0, help='Target ratio of Non-PCL:PCL (2.0 = notebook default)')
    parser.add_argument('--force_regen', action='store_true', help='Force regeneration of augmented data (ignore cache)')
    parser.add_argument('--freeze_embeddings', action='store_true', help='Freeze embedding layer')
    parser.add_argument('--freeze_layers', type=int, default=0, help='Number of bottom transformer layers to freeze (0-24)')
    parser.add_argument('--use_rdrop', action='store_true', help='Use R-Drop regularization')
    parser.add_argument('--rdrop_alpha', type=float, default=1.0, help='R-Drop coefficient (default 1.0)')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay for optimizer')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("ENHANCED ROBERTA PCL CLASSIFIER")
    print("="*80)
    print("\nCombines:")
    print("  ✓ Notebook-style data loading (DontPatronizeMe)")
    print("  ✓ Random downsampling (vs notebook's sequential)")
    print("  ✓ Class weighting: 1/sqrt(ratio)")
    print("  ✓ Cosine schedule with warmup")
    print("  ✓ Gradient clipping")
    print("  ✓ Backtranslation augmentation option")
    print("  ✓ Layer freezing option")
    print("  ✓ R-Drop regularization option")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Clear any existing cached memory
        torch.cuda.empty_cache()
        print("✓ Cleared CUDA cache")
    
    # Load data using DPM module
    train_df, dev_df = load_pcl_data_from_dpm(
        augment=args.augment,
        num_augments_per_sample=args.num_augments,
        downsample=args.downsample,
        downsample_ratio=args.downsample_ratio,
        device=device,
        force_regenerate=args.force_regen
    )
    
    # Calculate class weights: 1/sqrt(rp) and 1/sqrt(rn)
    class_counts = Counter(train_df['label'])
    total = sum(class_counts.values())
    
    r_neg = class_counts[0] / total
    r_pos = class_counts[1] / total
    
    weight_neg = 1.0 / np.sqrt(r_neg)
    weight_pos = 1.0 / np.sqrt(r_pos)
    
    class_weights = torch.tensor([weight_neg, weight_pos], dtype=torch.float32)
    
    print(f"\nClass statistics:")
    print(f"  Non-PCL count: {class_counts[0]} (ratio: {r_neg:.4f})")
    print(f"  PCL count:     {class_counts[1]} (ratio: {r_pos:.4f})")
    print(f"\nClass weights (1/sqrt(ratio)):") 
    print(f"  Non-PCL weight: {weight_neg:.4f}")
    print(f"  PCL weight:     {weight_pos:.4f}")
    
    # Load tokenizer
    print("\nLoading RoBERTa-base tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    
    # Create datasets
    train_dataset = PCLDataset(
        train_df['text'].values,
        train_df['label'].values,
        tokenizer,
        max_length=args.max_length
    )
    
    dev_dataset = PCLDataset(
        dev_df['text'].values,
        dev_df['label'].values,
        tokenizer,
        max_length=args.max_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Dev batches: {len(dev_loader)}")
    
    # Load model
    print("\nLoading RoBERTa-base model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        'roberta-base',
        num_labels=2
    )
    model.to(device)
    
    # Apply layer freezing
    freeze_model_layers(
        model,
        freeze_embeddings=args.freeze_embeddings,
        freeze_layers=args.freeze_layers
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nFinal parameter counts:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Configuration
    strategies = []
    if args.augment:
        strategies.append('backtranslation')
    if args.downsample:
        strategies.append('downsample')
    if not strategies:
        strategies.append('none')
    
    config = {
        'model_name': 'roberta-base',
        'n_labels': 2,
        'batch_size': args.batch_size,
        'accumulation_steps': args.accumulation_steps,
        'effective_batch_size': args.batch_size * args.accumulation_steps,
        'lr': args.lr,
        'warmup': args.warmup,
        'train_size': len(train_loader),
        'weight_decay': args.weight_decay,
        'n_epochs': args.epochs,
        'balancing_strategy': '+'.join(strategies),
        'downsample': args.downsample,
        'downsample_ratio': args.downsample_ratio if args.downsample else None,
        'augment': args.augment,
        'num_augments_per_sample': args.num_augments if args.augment else 0,
        'freeze_embeddings': args.freeze_embeddings,
        'freeze_layers': args.freeze_layers,
        'use_rdrop': args.use_rdrop,
        'rdrop_alpha': args.rdrop_alpha if args.use_rdrop else 0.0,
        'class_weights': class_weights.tolist()
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Train
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    history, best_f1 = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=dev_loader,
        device=device,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup,
        class_weights=class_weights,
        accumulation_steps=args.accumulation_steps,
        use_rdrop=args.use_rdrop,
        rdrop_alpha=args.rdrop_alpha,
        weight_decay=args.weight_decay
    )
    
    # Save history
    with open('history_enhanced.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Load best model and evaluate
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    
    model.load_state_dict(torch.load('best_model_enhanced.pt'))
    final_metrics = evaluate_model(model, dev_loader, device)
    
    # Save results
    results = {
        'best_f1': best_f1,
        'final_metrics': final_metrics,
        'config': config
    }
    
    with open('results_enhanced.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ Model saved: best_model_enhanced.pt")
    print("✓ History saved: history_enhanced.json")
    print("✓ Results saved: results_enhanced.json")

if __name__ == '__main__':
    main()
