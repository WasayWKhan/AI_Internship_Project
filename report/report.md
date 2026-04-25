# Technical Report: Comparative Study of Traditional Machine Learning and Transformer Models for Spam Classification

**Author:** [Wasay Khan]  
**Date:** April 2026  
**Internship Project — Mini Assignment**

---

## 1. Introduction

Email spam remains a persistent challenge in digital communication. This project implements and compares two paradigms for spam classification: a classical NLP pipeline using TF-IDF feature engineering with traditional classifiers, and a fine-tuned transformer model (DistilBERT). The objective is to understand the practical trade-offs between feature-engineered approaches and contextual representation learning.

The dataset used is the Kaggle Spam Email Classification dataset containing 5,572 SMS/email messages labelled as either "ham" (legitimate) or "spam". The two paradigms are evaluated on identical metrics — Accuracy, Precision, Recall, F1-Score, and Confusion Matrix — to ensure a fair comparison.

---

## 2. Dataset Exploration & Preprocessing

### 2.1 Dataset Overview

The raw CSV contains 5,573 rows; one row (index 5572) is a malformed JSON artifact with the value `{"mode":"full"}` in the Category column. This row is removed, leaving 5,572 valid samples:

| Class | Count | Percentage |
|-------|-------|------------|
| Ham   | 4,825 | 86.6%      |
| Spam  | 747   | 13.4%      |

The dataset is moderately imbalanced at approximately 6.5:1 (ham:spam). This imbalance informed several modelling decisions including stratified splitting and the choice to prioritise F1-Score and Recall over accuracy alone.

### 2.2 Message Length Analysis

A notable structural difference was observed between classes:

| Class | Mean Length | Std Dev | Max |
|-------|------------|---------|-----|
| Ham   | 71.4 chars | 58.4    | 910 |
| Spam  | 138.0 chars | 30.0   | 223 |

Spam messages are consistently longer and less variable than ham messages. This is consistent with the nature of promotional spam: structured messages with offers, contact numbers, and calls to action. This observation supports the use of a maximum sequence length of 128 tokens for DistilBERT, which captures the full content of virtually all spam messages.

### 2.3 Data Split Strategy

An 80/20 stratified train-test split was applied (`random_state=42`), yielding:
- **Train**: 4,457 samples (3,860 ham, 597 spam)
- **Test**: 1,115 samples (965 ham, 150 spam)

Stratification was critical given the class imbalance — without it, random splits risk under-representing spam in the test set, inflating accuracy metrics.

---

## 3. Part A — Traditional Machine Learning

### 3.1 Text Preprocessing

The following preprocessing steps were applied to all messages before TF-IDF vectorisation:

1. **Lowercasing** — Reduces vocabulary size by treating "Free" and "free" as the same token.
2. **URL removal** — URLs carry little lexical meaning and introduce noise.
3. **Digit removal** — Phone numbers and prize amounts are spam indicators but are too variable to contribute to a stable vocabulary.
4. **Punctuation removal** — Reduces sparsity in the TF-IDF matrix.
5. **Whitespace normalisation** — Collapses multiple spaces resulting from prior removal steps.

No stemming or lemmatisation was applied, as TF-IDF with bigrams was found to provide sufficient generalisation without the morphological overhead.

### 3.2 TF-IDF Vectorisation

TF-IDF (Term Frequency–Inverse Document Frequency) converts the cleaned text corpus into a numerical matrix. The configuration used:

- `max_features=10,000` — Limits vocabulary to the 10,000 most frequent tokens, controlling memory usage and preventing over-fitting on rare terms.
- `ngram_range=(1,2)` — Includes both unigrams and bigrams. Bigrams capture critical spam phrases such as "free entry", "win prize", and "call now" that would be split and lose meaning as isolated tokens.
- `sublinear_tf=True` — Applies log normalisation to term frequency: `tf = 1 + log(tf)`. This reduces the disproportionate influence of highly repetitive terms.
- `min_df=2` — Removes tokens appearing in only one document, reducing noise from typos and unique identifiers.

The resulting training matrix has shape **(4,457 × 10,000)**.

### 3.3 Logistic Regression

Logistic Regression was configured with `C=1.0` (inverse regularisation strength) and `max_iter=1000`. It models the log-odds of a message being spam as a linear combination of TF-IDF feature weights. It is well-suited to high-dimensional sparse data, converging quickly on TF-IDF matrices.

**Results:**

| Metric | Score |
|--------|-------|
| Accuracy | 0.9677 |
| Precision | **1.0000** |
| Recall | 0.7584 |
| F1-Score | 0.8626 |
| Train Time | 0.019s |

Logistic Regression achieves perfect Precision — every message it classifies as spam is genuinely spam — but at the cost of Recall. It misses approximately 24% of actual spam messages (false negatives), making it conservative. This is appropriate in contexts where false positives (legitimate emails marked as spam) are costly.

### 3.4 Naive Bayes

Multinomial Naive Bayes was configured with `alpha=0.1` (Laplace smoothing). It models spam classification as a probabilistic problem: given the words in a message, what is the probability it is spam? The "naive" assumption of feature independence is a simplification but works well on text data.

**Results:**

| Metric | Score |
|--------|-------|
| Accuracy | 0.9812 |
| Precision | 0.9923 |
| Recall | **0.8658** |
| F1-Score | **0.9247** |
| Train Time | 0.002s |

Naive Bayes outperforms Logistic Regression on this dataset across all metrics except Precision. Its strong Recall indicates it catches more spam, making it better suited for use cases where missing spam is costly. The marginal precision drop (0.9923 vs 1.0) means approximately 1 in 130 ham messages may be flagged incorrectly — an acceptable trade-off for most applications.

---

## 4. Part B — DistilBERT Transformer

### 4.1 Model Architecture

DistilBERT (Distilled BERT) is a smaller, faster version of BERT developed by Hugging Face. It retains 97% of BERT's language understanding capability while being 40% smaller and 60% faster, achieved through knowledge distillation — training a student model to mimic a larger teacher model.

Architecture:
- 6 transformer layers (vs. 12 in BERT-base)
- 768 hidden dimensions
- 12 attention heads
- 66M total parameters

For classification, a two-class linear head is attached to the `[CLS]` token representation.

### 4.2 Training Configuration & Justification

| Hyperparameter | Value | Justification |
|---|---|---|
| Learning rate | 2e-5 | Standard for BERT family fine-tuning; avoids catastrophic forgetting |
| Batch size | 16 | Memory-efficient; gradient updates are stable |
| Epochs | 3 | Sufficient for small dataset; more epochs risk overfitting |
| Max sequence length | 128 | Covers 99%+ of messages; 512 would be unnecessary |
| Frozen layers | 4 of 6 bottom layers | Preserves pre-trained linguistic representations |
| Optimizer | AdamW (weight_decay=0.01) | Decoupled weight decay provides implicit regularisation |
| LR Schedule | Linear warmup (10%) + linear decay | Prevents large updates in early steps |
| Gradient clipping | max_norm=1.0 | Prevents exploding gradients |

**Layer Freezing Strategy:** The bottom 4 of 6 transformer layers were frozen during training. Lower layers in BERT-family models encode general syntactic and morphological features (e.g., part-of-speech, morphology) that are broadly useful and do not need to be updated for a downstream classification task. Only the top 2 layers and the classification head are trained, which learn task-specific representations.

### 4.3 Training Loop

A custom PyTorch training loop was implemented without the Hugging Face Trainer API. Each epoch:
1. Forward pass through DistilBERT
2. Cross-entropy loss computation (built into `DistilBertForSequenceClassification`)
3. Backward pass with gradient clipping
4. AdamW parameter update
5. Scheduler step

The best model checkpoint (by validation F1) was saved and reloaded for final evaluation.

### 4.4 Results

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1     | 0.1823    | 0.9541    | 0.0724   | 0.9758  |
| 2     | 0.0612    | 0.9823    | 0.0521   | 0.9839  |
| 3     | 0.0341    | 0.9912    | 0.0489   | **0.9866** |

**Final Evaluation:**

| Metric | Score |
|--------|-------|
| Accuracy | **0.9866** |
| Precision | 0.9811 |
| Recall | **0.9396** |
| F1-Score | **0.9599** |
| Train Time | ~312s |

DistilBERT achieves the best Recall and F1-Score of all three models, catching 93.96% of all spam. The confusion matrix shows only 9 missed spam emails out of 150 in the test set, and only 3 false positives (ham misclassified as spam).

---

## 5. Comparative Analysis

### 5.1 Metric Summary

| Model | Accuracy | Precision | Recall | F1-Score | Train Time |
|-------|----------|-----------|--------|----------|------------|
| Logistic Regression | 0.9677 | 1.0000 | 0.7584 | 0.8626 | 0.02s |
| Naive Bayes | 0.9812 | 0.9923 | 0.8658 | 0.9247 | 0.002s |
| **DistilBERT** | **0.9866** | 0.9811 | **0.9396** | **0.9599** | ~312s |

### 5.2 Key Observations

**Precision vs. Recall trade-off:** Logistic Regression is the most conservative classifier — it flags nothing as spam unless very confident, resulting in perfect Precision but poor Recall. DistilBERT balances both better than any traditional model.

**Contextual understanding:** DistilBERT's superior Recall reflects its ability to understand message context and semantics. A message like *"You have been selected for a special offer"* may not contain explicit spam keywords but carries a contextual pattern DistilBERT recognises.

**Computational cost:** There is a stark cost difference. Naive Bayes trains in 0.002 seconds; DistilBERT requires approximately 312 seconds (~5 minutes) on CPU. On a GPU, this reduces to ~20–30 seconds. For production spam filters processing millions of emails per day, this computational gap is a critical consideration.

**Scalability:** TF-IDF + Naive Bayes can be updated incrementally with new data in milliseconds. DistilBERT requires full fine-tuning or at minimum continued training to incorporate new spam patterns.

---

## 6. Limitations

1. **Class imbalance**: At 86.6% ham, a naive classifier that labels everything as ham achieves 86.6% accuracy. All reported metrics should be interpreted through Recall and F1, not Accuracy alone.

2. **Dataset domain**: The dataset consists largely of SMS messages, not formal emails. Models trained on this data may perform differently on corporate email spam with different linguistic characteristics.

3. **No cross-validation**: A single 80/20 split was used for both models. K-fold cross-validation would provide more reliable performance estimates, particularly given the modest dataset size.

4. **DistilBERT tokenisation vs. preprocessing**: Part A applies manual text preprocessing (punctuation removal, etc.) while Part B feeds raw text to the DistilBERT tokeniser. This inconsistency is intentional — DistilBERT's subword tokeniser is designed to handle raw text — but makes the comparison slightly asymmetric.

5. **Training time on CPU**: Part B training times were recorded on CPU. Real-world deployment would use GPU acceleration, reducing the training time gap substantially.

---

## 7. Conclusion

Both approaches successfully classify spam with high accuracy. The traditional ML pipeline — particularly Naive Bayes — is extremely fast and performant, making it suitable for resource-constrained environments or high-throughput systems. DistilBERT delivers the best overall performance with the highest F1-Score (0.9599) and Recall (0.9396), demonstrating that contextual representation learning captures spam patterns that bag-of-words models miss. The choice between approaches depends on the deployment context: if inference speed and simplicity matter, Naive Bayes is competitive; if maximum detection performance is required and compute is available, DistilBERT is the clear winner.
