import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["KMP_DUPLICATE_LIB_OK"]  = "TRUE"

# ── SentenceTransformer MUST load before joblib
from sentence_transformers import SentenceTransformer
st_model = SentenceTransformer("all-MiniLM-L6-v2")

import json
import faiss
import pickle
import joblib
import numpy as np
import pandas as pd
import ollama
from xgboost import XGBClassifier
from datetime import datetime

BASE= "/Users/sambit_03/Desktop/Student Response/"
THRESHOLD_HIGH = 0.90   # very confident → direct, LLM writes feedback only
THRESHOLD_LOW  = 0.75   # uncertain → LLM acts as full judge (grade + feedback)


index = faiss.read_index(BASE + "faiss_index.bin")
with open(BASE + "metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

model_xgb = joblib.load(BASE + "xgb_model_local.pkl")
pca       = joblib.load(BASE + "pca.pkl")
scaler    = joblib.load(BASE + "scaler.pkl")
le        = joblib.load(BASE + "label_encoder.pkl")

print(f"  FAISS   : {index.ntotal} vectors")
print(f"  Metadata: {len(metadata)} records")
print(f"  Classes : {le.classes_}")
print(f"  ✅ All artifacts loaded\n")


def xgb_predict(ref_ans: str, student_ans: str) -> tuple:
    ref_emb = st_model.encode([ref_ans],     normalize_embeddings=True, convert_to_numpy=True)
    stu_emb = st_model.encode([student_ans], normalize_embeddings=True, convert_to_numpy=True)

    cos_sim = float((ref_emb * stu_emb).sum())

    combined        = np.vstack([ref_emb, stu_emb]).astype(np.float32)
    combined_scaled = scaler.transform(combined)
    combined_pca    = pca.transform(combined_scaled)

    X = np.hstack([
        combined_pca[0:1],
        combined_pca[1:2],
        [[cos_sim]]
    ]).astype(np.float32)

    proba      = model_xgb.predict_proba(X)[0]
    pred_idx   = int(proba.argmax())
    confidence = float(proba.max())
    grade      = le.classes_[pred_idx]

    return grade, confidence, proba

def retrieve_examples(question: str, ref_ans: str,
                      student_ans: str, k: int = 5) -> list:
    query = f"{question} [REF] {ref_ans} [STU] {student_ans}"
    qvec  = st_model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)

    scores, indices = index.search(qvec, k=k)

    return [
        {
            "question" : metadata[idx]["question"],
            "ref_ans"  : metadata[idx]["ref_ans"],
            "response" : metadata[idx]["response"],
            "grade"    : metadata[idx]["grade_label"],
            "score"    : round(float(score), 4)
        }
        for score, idx in zip(scores[0], indices[0])
    ]
def build_prompt(question: str, ref_ans: str, student_ans: str,
                 examples: list, xgb_grade: str,
                 xgb_confidence: float) -> str:

    ex_block = ""
    for i, ex in enumerate(examples, 1):
        ex_block += f"""
Example {i} (similarity={ex['score']:.3f}):
  Question  : {ex['question']}
  Reference : {ex['ref_ans']}
  Student   : {ex['response']}
  Grade     : {ex['grade']}
"""

    return f"""You are an expert student answer grader for science questions.
Grade the student answer using ONLY one of these four labels:
  - correct
  - partially_correct_incomplete
  - contradictory
  - irrelevant

DEFINITIONS:
  correct                      → answer matches reference in meaning
  partially_correct_incomplete → right track but missing key details
  contradictory                → similar vocabulary but states the opposite fact
  irrelevant                   → off-topic or unrelated to the question

PAY CLOSE ATTENTION to contradictory answers — high word overlap with reference but flipped meaning.

Similar graded examples:
{ex_block}
---
Now grade this:
  Question  : {question}
  Reference : {ref_ans}
  Student   : {student_ans}

Classifier predicted "{xgb_grade}" with {xgb_confidence:.0%} confidence.
Override only if examples clearly suggest a different label.

Respond ONLY with valid JSON, no explanation, no markdown:
{{"grade": "<label>", "feedback": "<one constructive sentence for the student>"}}"""

def call_llm(prompt: str) -> dict:
    response = ollama.chat(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0}
    )
    raw = response["message"]["content"].strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"grade": None, "feedback": raw}

def grade_answer(question: str, ref_ans: str, student_ans: str) -> dict:

    xgb_grade, confidence, proba = xgb_predict(ref_ans, student_ans)

    # Very high confidence → XGBoost grade is trusted
    # LLM still writes feedback
    if confidence >= THRESHOLD_HIGH:
        # LLM judges grade AND writes feedback (can override XGBoost)
        prompt = f"""You are an expert science answer grader.
A classifier graded this answer as: {xgb_grade} with {confidence:.0%} confidence.

Question  : {question}
Reference : {ref_ans}
Student   : {student_ans}

IMPORTANT: Check if the student answer contradicts the reference.
Same vocabulary but opposite meaning = contradictory. The classifier often misses this.

If you detect a contradiction override the grade to "contradictory".
Otherwise keep the classifier grade: {xgb_grade}

Grade using ONLY: correct / partially_correct_incomplete / contradictory / irrelevant

Respond ONLY with valid JSON, no explanation, no markdown:
{{"grade": "<label>", "feedback": "<one constructive sentence>"}}"""

        resp = ollama.chat(
            model="llama3.2:3b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0}
        )
        raw = resp["message"]["content"].strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"): raw = raw[4:]
        try:
            out         = json.loads(raw.strip())
            final_grade = out.get("grade", xgb_grade)
            feedback    = out.get("feedback", "")
        except Exception:
            final_grade = xgb_grade
            feedback    = raw

        route = "llm override" if final_grade != xgb_grade else "direct"
        return {"grade": final_grade, "confidence": confidence,
                "feedback": feedback, "route": route}

    # RAG + LLM judge — unchanged below this line
    query = f"{question} [REF] {ref_ans} [STU] {student_ans}"
    qvec  = st_model.encode([query], normalize_embeddings=True,
                             convert_to_numpy=True).astype(np.float32)
    scores, indices = index.search(qvec, k=5)
    examples = [{"response": metadata[i]["response"],
                 "grade":    metadata[i]["grade_label"],
                 "score":    float(s)}
                for s, i in zip(scores[0], indices[0])]

    ex_block = "\n".join([
        f"Example {i+1}: Student: {ex['response'][:80]}  Grade: {ex['grade']}"
        for i, ex in enumerate(examples)
    ])

    prompt = f"""You are an expert student answer grader.
Grade using ONLY: correct / partially_correct_incomplete / contradictory / irrelevant

Examples:
{ex_block}

Question  : {question}
Reference : {ref_ans}
Student   : {student_ans}

Classifier predicted "{xgb_grade}" with {confidence:.0%} confidence.
Override only if examples clearly suggest otherwise.

Respond ONLY with valid JSON:
{{"grade": "<label>", "feedback": "<one helpful sentence>"}}"""

    resp = ollama.chat(model="llama3.2:3b",
                       messages=[{"role":"user","content":prompt}],
                       options={"temperature":0.0})
    raw = resp["message"]["content"].strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
    try:
        out = json.loads(raw.strip())
    except Exception:
        out = {"grade": xgb_grade, "feedback": raw}

    return {"grade":      out.get("grade", xgb_grade),
            "confidence": confidence,
            "feedback":   out.get("feedback", ""),
            "route":      "llm"}

# if __name__ == "__main__":

#     test_cases = [
#         {
#             "label"      : "CORRECT — expect direct route",
#             "question"   : "What happens to pitch when frequency increases?",
#             "ref_ans"    : "The pitch of the sound increases when frequency increases.",
#             "student_ans": "The pitch gets higher as the frequency goes up."
#         },
#         {
#             "label"      : "CONTRADICTORY — expect LLM route",
#             "question"   : "What happens to pitch when frequency increases?",
#             "ref_ans"    : "The pitch of the sound increases when frequency increases.",
#             "student_ans": "The pitch gets lower when the frequency increases."
#         },
#         {
#             "label"      : "IRRELEVANT",
#             "question"   : "What happens to pitch when frequency increases?",
#             "ref_ans"    : "The pitch of the sound increases when frequency increases.",
#             "student_ans": "I have no idea about sound."
#         },
#         {
#             "label"      : "PARTIALLY CORRECT",
#             "question"   : "What happens to pitch when frequency increases?",
#             "ref_ans"    : "The pitch of the sound increases when frequency increases.",
#             "student_ans": "The sound changes when frequency changes."
#         },
#     ]

#     print("=" * 60)
#     print("PIPELINE TEST")
#     print("=" * 60)

#     for tc in test_cases:
#         print(f"\n── {tc['label']}")
#         result = grade_answer(tc["question"], tc["ref_ans"], tc["student_ans"])
#         print(f"   Student   : {tc['student_ans']}")
#         print(f"   Grade     : {result['grade']}")
#         print(f"   Confidence: {result['confidence']:.0%}")
#         print(f"   Route     : {result['route']}")
#         print(f"   Feedback  : {result['feedback']}")
#         if result["route"] == "llm":
#             print(f"   RAG hits  :")
#             for ex in result["examples"]:
#                 print(f"     [{ex['score']:.3f}] {ex['grade']:30s} → {ex['response'][:55]}")

if __name__ == "__main__":

    

    # ── Load test cases
    with open(BASE + "test_irrelevant.json") as f:
        test_cases = json.load(f)

    print(f"Loaded {len(test_cases)} irrelevant test cases\n")
    print("=" * 60)

    results = []

    for i, tc in enumerate(test_cases, 1):
        result = grade_answer(tc["question"], tc["ref_ans"], tc["student_ans"])

        correct = result["grade"] == tc["true_grade"]
        status  = "✅" if correct else "❌"

        print(f"[{i:02d}] {status}  predicted={result['grade']:30s}  "
              f"true={tc['true_grade']:12s}  conf={result['confidence']:.0%}  "
              f"route={result['route']}")

        results.append({
            "question"       : tc["question"],
            "ref_ans"        : tc["ref_ans"],
            "student_ans"    : tc["student_ans"],
            "true_grade"     : tc["true_grade"],
            "predicted_grade": result["grade"],
            "confidence"     : result["confidence"],
            "feedback"       : result["feedback"],
            "route"          : result["route"],
            "correct"        : correct
        })

    # ── Summary
    df = pd.DataFrame(results)
    total    = len(df)
    correct  = df["correct"].sum()
    accuracy = correct / total

    print(f"\n{'='*60}")
    print(f"IRRELEVANT CLASS TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Total        : {total}")
    print(f"  Correct      : {correct}")
    print(f"  Accuracy     : {accuracy:.0%}")
    print(f"\n  Predicted as :")
    print(df["predicted_grade"].value_counts().to_string())
    print(f"\n  Route breakdown:")
    print(df["route"].value_counts().to_string())

    # ── Misclassified cases
    misclassified = df[~df["correct"]]
    if len(misclassified) > 0:
        print(f"\n  Misclassified ({len(misclassified)} cases):")
        for _, row in misclassified.iterrows():
            print(f"\n    Student  : {row['student_ans'][:80]}")
            print(f"    Predicted: {row['predicted_grade']}  ({row['confidence']:.0%} conf, {row['route']})")
            print(f"    Feedback : {row['feedback']}")

    # ── Save results
    out_path = BASE + "irrelevant_test_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\n Saved → {out_path}")