# 🎓 ASAG — Automated Short Answer Grading System

A two-stage hybrid pipeline that combines **XGBoost classification** with a **Llama 3.2 LLM judge** to grade short student answers across four semantic categories: `correct`, `partially_correct_incomplete`, `contradictory`, and `irrelevant`.

Built to run entirely **locally on Apple Silicon (M2, 8GB RAM)** — no API calls, no cloud dependencies.

---

## 🏗️ Architecture

```
Raw Data
   │
   ▼ Preprocessing
Preprocessed CSV
   ├──► Sentence Embeddings (all-MiniLM-L6-v2, 384d)
   ├──► Cosine Similarity
   └──► PCA Compression (384d → 168d, 95% variance retained)
              │
              ▼
        Feature Matrix X
              │
              ▼
   XGBoost Classifier (85% weighted accuracy, 4 classes)
              │
              ▼
      Confidence Router
         │         │
      ≥ 0.70    < 0.70
         │         │
         │         ▼
         │   RAG Prompt Assembly
         │   (FAISS IndexFlatIP, top-5 few-shot examples)
         │         │
         └────────►▼
              Llama 3.2 Judge
              (temp=0.0, JSON response)
                   │
                   ▼
             Final Output
```

---

## ✨ Features

- **Two-stage routing** — Fast XGBoost handles high-confidence cases; ambiguous cases escalate to an LLM for nuanced reasoning
- **RAG-augmented few-shot prompting** — FAISS vector index over 13,000+ training examples retrieves semantically similar graded answers at inference time
- **Per-subject threshold calibration** — Isotonic regression fits per-subject confidence thresholds stored as JSON in ChromaDB alongside reference answers
- **Negation-aware gating** — `needs_llm_review()` explicitly detects negation mismatches, ultra-short answers, and high-confidence/low-similarity conflicts before routing
- **M2-optimized training** — `float32` feature casting, `tree_method='hist'`, `n_jobs=1`, and decoupled OpenMP environments prevent kernel crashes and ARM64 segfaults
- **Streamlit frontend** — Dark-themed UI with color-coded grade badges, confidence bars, route pills (⚡ Direct / 🤖 LLM Judge / 🔄 LLM Override), and a persistent `grading_log.csv` audit trail

---

## 📊 Results

| Class | Precision | Notes |
|-------|-----------|-------|
| correct | High | 2,458 / 2,584 correctly classified |
| partially_correct_incomplete | Moderate | 136 / 196 correctly classified |
| irrelevant | Low | **Major leak** — 299/609 misclassified as `correct` |
| contradictory | Low | **Worst class** — 14/29 misclassified as `partially_correct_incomplete` |

**Overall weighted accuracy: 85%**

---

## ⚠️ Known Limitations

### 1. Contradictory Class Blindspot
The `contradictory` class has the worst recall. Sentence embeddings encode semantic *similarity*, so "Frequency increases pitch" and "Frequency does NOT increase pitch" produce nearly identical vectors. The current feature set has no explicit negation signal.

**Planned fix:** Inject boolean negation-mismatch features via regex/dependency parsing — checking whether negation words (`not`, `no`, `never`) appear in the student answer but not in the reference, or vice versa.

### 2. Irrelevant → Correct Bleed
Nearly half of all `irrelevant` answers (299/609) are predicted as `correct`. Students who write long, rambling responses that reuse vocabulary from the prompt inflate cosine similarity and Jaccard overlap without containing real information.

**Planned fix:** Add `absolute_length_difference` and TF-IDF keyphrase matching — computing cosine similarity strictly on high-IDF reference terms to penalize fluffy keyword repetition.

### 3. Partially Correct vs. Correct Overlap
57 `partially_correct_incomplete` answers bled into `correct`. Standard symmetric cosine similarity cannot detect *which* reference concepts are simply missing from a student's response.

**Planned fix:** Replace symmetric cosine similarity with a directional recall-focused overlap metric — measuring what percentage of the reference answer's core semantic chunks are absent from the student response.

---

## 🧰 Tech Stack

| Component | Tool |
|-----------|------|
| Sentence embeddings | `sentence-transformers` / `all-MiniLM-L6-v2` |
| Dimensionality reduction | `scikit-learn` PCA (384d → 168d) |
| Classifier | `XGBoost` (`multi:softprob`, `tree_method='hist'`) |
| Threshold calibration | `scikit-learn` IsotonicRegression |
| Vector index | `FAISS` (`IndexFlatIP`) |
| LLM judge | `Llama 3.2 3B` via `Ollama` |
| Reference store | `ChromaDB` |
| Dataset | ASAG2024 (HuggingFace) |
| Frontend | `Streamlit` |

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Pull the LLM (requires Ollama)
ollama pull llama3.2:3b

# 3. Preprocess and build features
python preprocess.py

# 4. Train XGBoost classifier
python train.py

# 5. Build FAISS index
python build_index.py

# 6. Launch Streamlit app
streamlit run app.py
```

---

## ⚙️ M2 Apple Silicon Notes

If you're running on ARM64 macOS, set these **before any imports** to avoid OpenMP conflicts between PyTorch (SentenceTransformer) and XGBoost:

```python
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```

When moving a model trained on Colab (x86) to M2, always save/load via XGBoost's native JSON format:

```python
model.save_model("xgb_model.json")   # on Colab
model.load_model("xgb_model.json")   # on M2
```

---

## 📁 Project Structure

```
Student Response/
├── preprocess.py          # Feature extraction pipeline
├── train.py               # XGBoost training + threshold calibration
├── build_index.py         # FAISS index construction
├── grade.py               # Inference: router + LLM judge
├── app.py                 # Streamlit frontend
├── grading_log.csv        # Auto-generated audit trail
├── xgb_model.json         # Trained classifier
├── pca_model.pkl          # Fitted PCA transformer
├── faiss_index.bin        # FAISS vector index
└── thresholds.json        # Per-subject calibrated thresholds
```

---

## 📄 License

MIT
