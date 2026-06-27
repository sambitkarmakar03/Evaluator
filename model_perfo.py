import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# ── Load artifacts
BASE = "/Users/sambit_03/Desktop/Student Response/"

model   = joblib.load(BASE + "xgb_model.pkl")
le      = joblib.load(BASE + "label_encoder.pkl")
pca     = joblib.load(BASE + "pca.pkl")
scaler  = joblib.load(BASE + "scaler.pkl")

with open(BASE + "label_map.json") as f:
    label_map = json.load(f)

print("Loaded. Classes:", le.classes_)

# ── Load features
X      = pd.read_csv(BASE + "X_features.csv").to_numpy(dtype=np.float32)
y_df   = pd.read_csv(BASE + "y_labels.csv")
y      = y_df["grade_encoded"].to_numpy(dtype=np.int32)

# ── Recreate same test split (same random_state = same split)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Predict
y_pred      = model.predict(X_test)
y_prob      = model.predict_proba(X_test)
confidence  = y_prob.max(axis=1)

# ─────────────────────────────────────────────
# 1. Accuracy + Classification Report
# ─────────────────────────────────────────────
print(f"\n── Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\n── Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ─────────────────────────────────────────────
# 2. Confusion Matrix
# ─────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
ax.set_title("Confusion Matrix")
plt.tight_layout(); plt.savefig(BASE + "confusion_matrix.png", dpi=150)
plt.show()

# ─────────────────────────────────────────────
# 3. Confidence Distribution per class
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
for i, cls in enumerate(le.classes_):
    mask = y_test == i
    axes[i].hist(confidence[mask], bins=20, color="#5DCAA5", edgecolor="white")
    axes[i].set_title(cls, fontsize=9)
    axes[i].set_xlabel("Confidence")
    axes[i].axvline(0.75, color="red", linestyle="--", linewidth=1, label="threshold")
axes[0].set_ylabel("Count")
axes[0].legend()
fig.suptitle("Confidence Distribution by Class", fontweight="bold")
plt.tight_layout(); plt.savefig(BASE + "confidence_dist.png", dpi=150)
plt.show()

# ─────────────────────────────────────────────
# 4. Routing simulation (how many go to LLM?)
# ─────────────────────────────────────────────
THRESHOLD = 0.75
high_conf = (confidence >= THRESHOLD).sum()
low_conf  = (confidence <  THRESHOLD).sum()

print(f"\n── Routing simulation (threshold={THRESHOLD}):")
print(f"   High confidence → direct  : {high_conf} ({high_conf/len(y_test)*100:.1f}%)")
print(f"   Low  confidence → LLM/RAG : {low_conf}  ({low_conf/len(y_test)*100:.1f}%)")

# ─────────────────────────────────────────────
# 5. Feature importance (top 20)
# ─────────────────────────────────────────────
feat_names = pd.read_csv(BASE + "X_features.csv").columns.tolist()
importances = model.feature_importances_
top_idx = np.argsort(importances)[-20:][::-1]

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh([feat_names[i] for i in top_idx][::-1],
        importances[top_idx][::-1], color="#378ADD")
ax.set_xlabel("Importance score")
ax.set_title("Top 20 Feature Importances")
plt.tight_layout(); plt.savefig(BASE + "feature_importance.png", dpi=150)
plt.show()