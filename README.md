# AI_Internship_Project
Traditional Machine learning and Transformer models for Spam classification 
# 📧 Spam Classification: Traditional ML vs DistilBERT

A comparative study of classical machine learning and transformer-based deep learning approaches for email spam classification. Built as an internship mini-project.

---

## 🎯 Objective

Design and evaluate two spam classification systems:

| Approach | Methods |
|---|---|
| **Part A – Traditional ML** | TF-IDF + Logistic Regression + Naive Bayes |
| **Part B – Transformer** | DistilBERT (fine-tuned via raw PyTorch loop) |

---

## 📁 Project Structure

```
spam_classifier/
│
├── data/
│   └── email.csv                   # Dataset (5572 emails)
│
├── src/
│   ├── part_a_traditional_ml.py    # Part A: TF-IDF + LR + NB
│   └── part_b_distilbert.py        # Part B: DistilBERT fine-tuning
│
├── outputs/
│   ├── plots/
│   │   ├── eda_overview.png
│   │   ├── confusion_matrices_partA.png
│   │   ├── metric_comparison_partA.png
│   │   ├── distilbert_training_curves.png
│   │   ├── confusion_matrix_distilbert.png
│   │   └── full_comparison.png
│   ├── models/
│   │   └── distilbert_best.pt      # Best checkpoint (after training)
│   ├── partA_results.json
│   └── partB_results.json
│
├── report/
│   └── report.md                   # 4-page technical report
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

- **Source**: [Kaggle – Spam Email Classification](https://www.kaggle.com/datasets/ashfakyeafi/spam-email-classification)
- **File**: `email.csv`
- **Size**: 5,572 emails (after cleaning)
- **Columns**: `Category` (ham/spam), `Message`

| Class | Count | Percentage |
|-------|-------|------------|
| Ham   | 4,825 | 86.6%      |
| Spam  | 747   | 13.4%      |

> ⚠️ **Note**: Row 5572 in the raw CSV is a malformed JSON artifact and is dropped during loading.

---

## 🚀 Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/spam-classifier.git
cd spam-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the dataset
Place `email.csv` inside the `data/` folder.

### 4. Run Part A (Traditional ML)
```bash
python src/part_a_traditional_ml.py
```

### 5. Run Part B (DistilBERT)
```bash
python src/part_b_distilbert.py
```

> **GPU recommended** for Part B. Part A runs in under 1 second on CPU.

---

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score | Train Time |
|-------|----------|-----------|--------|----------|------------|
| Logistic Regression | 0.9677 | 1.0000 | 0.7584 | 0.8626 | 0.02s |
| Naive Bayes | 0.9812 | 0.9923 | 0.8658 | 0.9247 | 0.002s |
| **DistilBERT** | **0.9866** | **0.9811** | **0.9396** | **0.9599** | ~312s |

### Key Takeaway
- DistilBERT achieves the highest F1-Score (0.9599) and best Recall (0.9396)
- Logistic Regression has perfect Precision (1.0) but misses many spam emails (low Recall)
- Naive Bayes offers the best trade-off in the traditional ML category
- DistilBERT training takes ~300× longer than traditional methods

---

## 🖼️ Sample Outputs

### EDA – Class Distribution & Message Length
![EDA](outputs/plots/eda_overview.png)

### Confusion Matrices – Part A
![CM Part A](outputs/plots/confusion_matrices_partA.png)

### DistilBERT Training Curves
![Training](outputs/plots/distilbert_training_curves.png)

### Full Model Comparison
![Comparison](outputs/plots/full_comparison.png)

---

## ⚙️ Model Configuration (Part B)

| Hyperparameter | Value | Justification |
|---|---|---|
| Learning rate | 2e-5 | Standard for BERT fine-tuning |
| Batch size | 16 | Memory-efficient; stable gradients |
| Epochs | 3 | Sufficient for small dataset fine-tuning |
| Max seq length | 128 | Covers 99% of messages |
| Frozen layers | 4 of 6 | Prevents catastrophic forgetting |
| Optimizer | AdamW | Weight decay regularization |
| LR Schedule | Linear warmup + decay | Stable training |

---

## 📝 Report

See [`report/report.md`](report/report.md) for the full 4-page technical report.

---

## ⚠️ Limitations

- Class imbalance (87% ham) may bias models toward majority class
- Traditional models cannot capture word context or order
- DistilBERT requires significant compute for training
- Dataset is SMS-style text; may not generalize to formal email spam

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-green)
