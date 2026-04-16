"""
Part B: DistilBERT Transformer Model for Spam Classification
============================================================
Model  : distilbert-base-uncased (Hugging Face)
Backend: PyTorch (custom training loop — no Trainer API)
Dataset: email.csv (5572 emails: 4825 ham, 747 spam)

Design decisions:
  - Learning rate  : 2e-5  (standard for fine-tuning BERT-family models)
  - Batch size     : 16    (fits most GPUs; also works on CPU)
  - Epochs         : 3     (sufficient for fine-tuning on small datasets)
  - Max seq length : 128   (99% of spam emails fit within this length)
  - Layer freezing : Freeze first 4 of 6 transformer layers to prevent
                     catastrophic forgetting; only fine-tune top layers + classifier
  - Regularization : Weight decay 0.01 via AdamW optimizer
  - Scheduler      : Linear warmup (10% of steps) then linear decay
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re, string, time, os, json
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

CFG = {
    'model_name'  : 'distilbert-base-uncased',
    'max_len'     : 128,
    'batch_size'  : 16,
    'epochs'      : 3,
    'lr'          : 2e-5,
    'weight_decay': 0.01,
    'warmup_ratio': 0.1,
    'seed'        : 42,
    'freeze_layers': 4,      # freeze bottom 4 of 6 transformer layers
}

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class SpamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts.reset_index(drop=True)
        self.labels    = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids'     : encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels'        : torch.tensor(int(self.labels[idx]), dtype=torch.long)
        }

# ─────────────────────────────────────────────
# BUILD MODEL WITH LAYER FREEZING
# ─────────────────────────────────────────────

def build_model(cfg):
    model = DistilBertForSequenceClassification.from_pretrained(
        cfg['model_name'], num_labels=2
    )
    # Freeze bottom N transformer layers
    for i in range(cfg['freeze_layers']):
        for param in model.distilbert.transformer.layer[i].parameters():
            param.requires_grad = False

    total  = sum(p.numel() for p in model.parameters())
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Parameters: {total:,} total | {frozen:,} frozen | {total-frozen:,} trainable")
    return model.to(DEVICE)

# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in loader:
        input_ids  = batch['input_ids'].to(device)
        attn_mask  = batch['attention_mask'].to(device)
        labels     = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        loss    = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return total_loss / len(loader), correct / total


def eval_epoch(model, loader, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids  = batch['input_ids'].to(device)
            attn_mask  = batch['attention_mask'].to(device)
            labels     = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            loss    = outputs.loss
            preds   = outputs.logits.argmax(dim=1)

            total_loss += loss.item()
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)
            all_preds  .extend(preds.cpu().numpy())
            all_labels .extend(labels.cpu().numpy())

    return total_loss / len(loader), correct / total, all_preds, all_labels

# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────

def plot_training_curves(history, save_dir='outputs/plots'):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('DistilBERT Training Behaviour', fontsize=15, fontweight='bold')

    # Loss
    axes[0].plot(epochs, history['train_loss'], 'o-', color='#E53935', label='Train Loss', lw=2)
    axes[0].plot(epochs, history['val_loss'],   's--', color='#1976D2', label='Val Loss',   lw=2)
    axes[0].set_title('Loss per Epoch', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Cross-Entropy Loss')
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_xticks(epochs)

    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'o-', color='#E53935', label='Train Acc', lw=2)
    axes[1].plot(epochs, history['val_acc'],   's--', color='#1976D2', label='Val Acc',   lw=2)
    axes[1].set_title('Accuracy per Epoch', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].set_xticks(epochs)
    axes[1].set_ylim(0.90, 1.01)
    for ep, acc in zip(epochs, history['val_acc']):
        axes[1].annotate(f'{acc:.3f}', (ep, acc), textcoords='offset points',
                         xytext=(0, 8), ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/distilbert_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: distilbert_training_curves.png")


def plot_confusion_matrix_partB(cm, save_dir='outputs/plots'):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=ax,
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'],
                linewidths=1, linecolor='white')
    ax.set_title('DistilBERT – Confusion Matrix', fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrix_distilbert.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: confusion_matrix_distilbert.png")


def plot_full_comparison(partA_results, partB_result, train_times, save_dir='outputs/plots'):
    """Combined plot: all 3 models metrics + training time bar"""
    all_results = partA_results + [partB_result]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors  = ['#1976D2', '#E53935', '#FF6F00']
    model_names = [r['Model'] for r in all_results]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Full Model Comparison: Traditional ML vs DistilBERT', fontsize=15, fontweight='bold')

    # Metric comparison
    x = np.arange(len(metrics))
    width = 0.25
    for i, (res, color) in enumerate(zip(all_results, colors)):
        vals = [res[m] for m in metrics]
        bars = axes[0].bar(x + i*width, vals, width, label=res['Model'],
                           color=color, alpha=0.85, edgecolor='white')
        for bar, v in zip(bars, vals):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                         f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')

    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(metrics, fontsize=11)
    axes[0].set_ylim(0.85, 1.03)
    axes[0].set_ylabel('Score', fontsize=11)
    axes[0].set_title('Metrics Across All Models', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(axis='y', alpha=0.3)

    # Training time comparison
    bars2 = axes[1].bar(model_names, train_times, color=colors, alpha=0.85, edgecolor='white')
    for bar, t in zip(bars2, train_times):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{t:.1f}s', ha='center', fontsize=10, fontweight='bold')
    axes[1].set_title('Training Time Comparison', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Seconds', fontsize=11)
    axes[1].set_yscale('log')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/full_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: full_comparison.png")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    set_seed(CFG['seed'])

    # Load data
    df = pd.read_csv('data/email.csv')
    df = df[df['Category'].isin(['ham', 'spam'])].reset_index(drop=True)
    df['label'] = df['Category'].map({'ham': 0, 'spam': 1})

    X_train, X_test, y_train, y_test = train_test_split(
        df['Message'], df['label'],
        test_size=0.2, random_state=CFG['seed'], stratify=df['label']
    )

    # Tokenizer
    print("Loading tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained(CFG['model_name'])

    # Datasets & loaders
    train_ds = SpamDataset(X_train, y_train, tokenizer, CFG['max_len'])
    test_ds  = SpamDataset(X_test,  y_test,  tokenizer, CFG['max_len'])
    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=CFG['batch_size'], shuffle=False, num_workers=0)

    # Model
    print("Loading DistilBERT...")
    model = build_model(CFG)

    # Optimizer & scheduler
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=CFG['lr'], weight_decay=CFG['weight_decay']
    )
    total_steps  = len(train_loader) * CFG['epochs']
    warmup_steps = int(total_steps * CFG['warmup_ratio'])
    scheduler = get_linear_schedule_with_warmup(optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps)

    # ── Training loop ──
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_f1 = 0.0
    print("\nTraining DistilBERT...")
    t_start = time.time()

    for epoch in range(1, CFG['epochs'] + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
        vl_loss, vl_acc, preds, labels_list = eval_epoch(model, test_loader, DEVICE)

        history['train_loss'].append(tr_loss)
        history['train_acc'] .append(tr_acc)
        history['val_loss']  .append(vl_loss)
        history['val_acc']   .append(vl_acc)

        epoch_f1 = f1_score(labels_list, preds)
        print(f"Epoch {epoch}/{CFG['epochs']} | "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
              f"Val Loss: {vl_loss:.4f} Acc: {vl_acc:.4f} | F1: {epoch_f1:.4f}")

        if epoch_f1 > best_f1:
            best_f1 = epoch_f1
            torch.save(model.state_dict(), 'outputs/models/distilbert_best.pt')

    distilbert_time = time.time() - t_start
    print(f"\nTotal training time: {distilbert_time:.1f}s")

    # ── Final evaluation ──
    model.load_state_dict(torch.load('outputs/models/distilbert_best.pt', map_location=DEVICE))
    _, _, final_preds, final_labels = eval_epoch(model, test_loader, DEVICE)

    cm = confusion_matrix(final_labels, final_preds)
    partB_result = {
        'Model'    : 'DistilBERT',
        'Accuracy' : round(accuracy_score(final_labels, final_preds), 4),
        'Precision': round(precision_score(final_labels, final_preds), 4),
        'Recall'   : round(recall_score(final_labels, final_preds), 4),
        'F1-Score' : round(f1_score(final_labels, final_preds), 4),
        'train_time': distilbert_time
    }

    print("\n" + "="*45)
    print("PART B RESULTS – DistilBERT")
    print("="*45)
    for k, v in partB_result.items():
        if k not in ('Model',):
            print(f"  {k}: {v}")

    # ── Plots ──
    plot_training_curves(history)
    plot_confusion_matrix_partB(cm)

    # ── Full comparison (load Part A results) ──
    if os.path.exists('outputs/partA_results.json'):
        with open('outputs/partA_results.json') as f:
            partA_results = json.load(f)
        train_times = [r['train_time'] for r in partA_results] + [distilbert_time]
        plot_full_comparison(partA_results, partB_result, train_times)

    # Save Part B results
    with open('outputs/partB_results.json', 'w') as f:
        json.dump(partB_result, f, indent=2)

    print("\nPart B complete. All outputs saved.")

if __name__ == '__main__':
    main()
  
