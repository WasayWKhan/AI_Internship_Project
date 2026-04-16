"""
Part A: Traditional Machine Learning Pipeline for Spam Classification
=====================================================================
Pipeline: Text Preprocessing → TF-IDF Vectorization → Logistic Regression + Naive Bayes
Dataset: email.csv (5572 emails: 4825 ham, 747 spam)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import time
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

# ─────────────────────────────────────────────
# 1. LOAD & CLEAN DATASET
# ─────────────────────────────────────────────

def load_data(path='data/email.csv'):
    df = pd.read_csv(path)
    # Drop malformed row (JSON artifact found at row 5572)
    df = df[df['Category'].isin(['ham', 'spam'])].reset_index(drop=True)
    df['label'] = df['Category'].map({'ham': 0, 'spam': 1})
    print(f"Dataset loaded: {len(df)} rows")
    print(f"Ham: {(df['label']==0).sum()} | Spam: {(df['label']==1).sum()}")
    return df

# ─────────────────────────────────────────────
# 2. TEXT PREPROCESSING
# ─────────────────────────────────────────────

def preprocess_text(text):
    """
    Steps:
    - Lowercase
    - Remove URLs
    - Remove numbers
    - Remove punctuation
    - Strip extra whitespace
    Rationale: Reduces vocabulary size and noise; numbers/URLs
    carry little semantic meaning for spam detection.
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)       # remove URLs
    text = re.sub(r'\d+', '', text)                   # remove digits
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()          # collapse whitespace
    return text

# ─────────────────────────────────────────────
# 3. EDA VISUALIZATIONS
# ─────────────────────────────────────────────

def plot_eda(df, save_dir='outputs/plots'):
    os.makedirs(save_dir, exist_ok=True)
    df['msg_len'] = df['Message'].str.len()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Exploratory Data Analysis', fontsize=16, fontweight='bold')

    # Class distribution
    counts = df['Category'].value_counts()
    colors = ['#2196F3', '#F44336']
    bars = axes[0].bar(counts.index, counts.values, color=colors, edgecolor='white', linewidth=1.5)
    axes[0].set_title('Class Distribution', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Category', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                     f'{val}\n({val/len(df)*100:.1f}%)', ha='center', fontsize=10, fontweight='bold')
    axes[0].set_ylim(0, 5400)
    axes[0].grid(axis='y', alpha=0.3)

    # Message length distribution
    for cat, color in zip(['ham', 'spam'], colors):
        subset = df[df['Category'] == cat]['msg_len']
        axes[1].hist(subset, bins=40, alpha=0.6, label=f'{cat} (μ={subset.mean():.0f})',
                     color=color, edgecolor='white')
    axes[1].set_title('Message Length Distribution', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Character Count', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/eda_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: eda_overview.png")

# ─────────────────────────────────────────────
# 4. TRAIN & EVALUATE
# ─────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    return {
        'Model': model_name,
        'Accuracy':  round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall':    round(recall_score(y_test, y_pred), 4),
        'F1-Score':  round(f1_score(y_test, y_pred), 4),
        'CM':        confusion_matrix(y_test, y_pred),
        'y_pred':    y_pred
    }

def plot_confusion_matrices(results, save_dir='outputs/plots'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Confusion Matrices – Traditional ML Models', fontsize=15, fontweight='bold')

    for ax, res in zip(axes, results):
        sns.heatmap(res['CM'], annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'],
                    linewidths=1, linecolor='white', cbar_kws={'shrink': 0.8})
        ax.set_title(res['Model'], fontsize=13, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('Actual', fontsize=11)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrices_partA.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: confusion_matrices_partA.png")

def plot_metric_comparison_partA(results, save_dir='outputs/plots'):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    models  = [r['Model'] for r in results]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#1976D2', '#E53935']
    for i, (res, color) in enumerate(zip(results, colors)):
        vals = [res[m] for m in metrics]
        bars = ax.bar(x + i*width, vals, width, label=res['Model'],
                      color=color, alpha=0.85, edgecolor='white')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x + width/2)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0.85, 1.02)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Part A – Model Metric Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/metric_comparison_partA.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: metric_comparison_partA.png")

# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────

def main():
    df = load_data('data/email.csv')

    # EDA plots
    plot_eda(df)

    # Preprocessing
    df['clean_text'] = df['Message'].apply(preprocess_text)

    # Train/test split (80/20, stratified to preserve class ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['label'],
        test_size=0.2, random_state=42, stratify=df['label']
    )
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

    # TF-IDF Vectorization
    # max_features=10000: keeps top 10k tokens by frequency
    # ngram_range=(1,2): unigrams + bigrams to capture phrases like "free entry"
    # sublinear_tf=True: log-normalizes term frequency to reduce impact of very common words
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2),
                            sublinear_tf=True, min_df=2)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf  = tfidf.transform(X_test)
    print(f"TF-IDF matrix shape: {X_train_tfidf.shape}")

    # ── Logistic Regression ──
    t0 = time.time()
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(X_train_tfidf, y_train)
    lr_time = time.time() - t0

    # ── Naive Bayes ──
    t0 = time.time()
    nb = MultinomialNB(alpha=0.1)
    nb.fit(X_train_tfidf, y_train)
    nb_time = time.time() - t0

    # ── Evaluate ──
    results = [
        evaluate_model(lr, X_test_tfidf, y_test, 'Logistic Regression'),
        evaluate_model(nb, X_test_tfidf, y_test, 'Naive Bayes')
    ]
    results[0]['train_time'] = lr_time
    results[1]['train_time'] = nb_time

    print("\n" + "="*55)
    print("PART A RESULTS")
    print("="*55)
    for r in results:
        print(f"\n{r['Model']}")
        print(f"  Accuracy:  {r['Accuracy']:.4f}")
        print(f"  Precision: {r['Precision']:.4f}")
        print(f"  Recall:    {r['Recall']:.4f}")
        print(f"  F1-Score:  {r['F1-Score']:.4f}")
        print(f"  Train Time:{r['train_time']:.4f}s")

    plot_confusion_matrices(results)
    plot_metric_comparison_partA(results)

    # Save results for combined plot in Part B
    import json
    summary = [{k: v for k, v in r.items() if k not in ('CM', 'y_pred')} for r in results]
    with open('outputs/partA_results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\nPart A complete. Results saved to outputs/")

if __name__ == '__main__':
    main()
      
