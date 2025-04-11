# Hate Speech Classification using BiLSTM + Attention

This repository presents a deep learning-based hate speech classification system built using a Bidirectional LSTM (BiLSTM) network with an Attention mechanism. The goal is to detect and classify hate speech from online text data, leveraging various feature extraction techniques including **TF-IDF**, **GloVe**, and **Word2Vec**.

---

## 🧠 Model Architecture

The core model is a **BiLSTM with Attention**:
- **BiLSTM** captures both past and future context.
- **Attention Layer** focuses on the most informative parts of the input sequence.

---

## 📂 Approaches

### 1. **TF-IDF + BiLSTM + Attention**
- **Type**: Binary Classification (Hate vs. Non-Hate)
- **Tokenizer**: TF-IDF Vectorizer
- **Input Shape**: Sparse Matrix

### 2. **GloVe Embeddings + BiLSTM + Attention**
- **Type**: Binary Classification
- **Embedding Dimension**: 100D / 300D (based on variant used)
- **Tokenizer**: Pre-tokenized with GloVe vocabulary

### 3. **Word2Vec Embeddings + BiLSTM + Attention**
- **Type**: Multi-Class Classification (e.g., Hate, Offensive, Neutral)
- **Model**: Pretrained Word2Vec (or custom-trained on corpus)
- **Embedding Dimension**: 100D / 300D

---

## 📊 Evaluation Metrics

- **Accuracy**
- **Precision, Recall, F1-score**
- **Confusion Matrix**

---

## 📌 Highlights

- 🧠 BiLSTM architecture with a custom attention layer
- 🧪 Comparative study using different text vectorization methods
- 📈 Binary and multi-class classification support

---

## 📚 Dataset

Used the **Jigsaw Toxic Comment Classification dataset**:
- Available on [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

---

## ✨ Future Work

- Incorporate contextual embeddings (e.g., BERT, RoBERTa)
- Expand to multilingual hate speech detection
- Integrate adversarial training for robustness

---

## 🤝 Acknowledgments

- GloVe: [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)
- Word2Vec: [https://code.google.com/archive/p/word2vec/](https://code.google.com/archive/p/word2vec/)
- Jigsaw/Kaggle Dataset
