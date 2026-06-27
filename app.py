"""
app.py — Streamlit UI for ASAG grading pipeline
Run: streamlit run app.py
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["KMP_DUPLICATE_LIB_OK"]  = "TRUE"

import streamlit as st

st.set_page_config(
    page_title="ASAG Grader",
    page_icon="🎓",
    layout="centered"
)

# ── Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #0F1117; color: #E2E8F0; }

/* Header */
.header-block {
    background: linear-gradient(135deg, #1E293B 0%, #0F1117 100%);
    border: 1px solid #1E3A5F;
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 28px;
    text-align: center;
}
.header-block h1 {
    font-size: 26px;
    font-weight: 600;
    color: #F1F5F9;
    margin: 0 0 6px 0;
    letter-spacing: -0.3px;
}
.header-block p {
    font-size: 14px;
    color: #64748B;
    margin: 0;
}

/* Input card */
.input-card {
    background: #1A1D27;
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 24px;
    margin-bottom: 20px;
}

/* Grade badge */
.grade-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.4px;
    text-transform: uppercase;
}
.grade-correct           { background: #052e16; color: #4ade80; border: 1px solid #166534; }
.grade-partially         { background: #1c1917; color: #fb923c; border: 1px solid #9a3412; }
.grade-contradictory     { background: #1c0606; color: #f87171; border: 1px solid #991b1b; }
.grade-irrelevant        { background: #1e1b4b; color: #a5b4fc; border: 1px solid #3730a3; }

/* Result card */
.result-card {
    background: #1A1D27;
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 24px;
    margin-top: 20px;
}
.result-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 6px;
}
.feedback-text {
    font-size: 15px;
    color: #CBD5E1;
    line-height: 1.65;
    border-left: 3px solid #2563EB;
    padding-left: 14px;
    margin-top: 4px;
}
.route-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 500;
}
.route-direct { background: #052e16; color: #4ade80; }
.route-llm    { background: #1c1917; color: #fb923c; }

/* Confidence bar */
.conf-bar-bg {
    background: #1E293B;
    border-radius: 6px;
    height: 8px;
    margin-top: 6px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 8px;
    border-radius: 6px;
    background: linear-gradient(90deg, #2563EB, #3B82F6);
}

/* History table */
.history-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid #1E293B;
    font-size: 13px;
}
.history-ans {
    flex: 1;
    color: #94A3B8;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* Streamlit widget overrides */
div[data-testid="stTextArea"] textarea {
    background: #0F1117 !important;
    border: 1px solid #1E293B !important;
    color: #E2E8F0 !important;
    border-radius: 8px !important;
    font-size: 14px !important;
}
div[data-testid="stTextArea"] textarea:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 2px rgba(37,99,235,0.2) !important;
}
div[data-testid="stButton"] button {
    background: #2563EB !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 10px 28px !important;
    width: 100%;
}
div[data-testid="stButton"] button:hover {
    background: #1D4ED8 !important;
}
label { color: #94A3B8 !important; font-size: 13px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Load pipeline (cached — loads once)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading grading pipeline...")
def load_pipeline():
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer("all-MiniLM-L6-v2")

    import faiss, pickle, joblib
    from xgboost import XGBClassifier

    BASE = "/Users/sambit_03/Desktop/Student Response/"

    index = faiss.read_index(BASE + "faiss_index.bin")
    with open(BASE + "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    model_xgb = joblib.load(BASE + "xgb_model_local.pkl")
    pca       = joblib.load(BASE + "pca.pkl")
    scaler    = joblib.load(BASE + "scaler.pkl")
    le        = joblib.load(BASE + "label_encoder.pkl")

    return st_model, index, metadata, model_xgb, pca, scaler, le


def grade_answer(question, ref_ans, student_ans):
    import json, numpy as np, ollama

    THRESHOLD_HIGH = 0.70
    st_model, index, metadata, model_xgb, pca, scaler, le = load_pipeline()

    # Embed
    ref_emb = st_model.encode([ref_ans],     normalize_embeddings=True, convert_to_numpy=True)
    stu_emb = st_model.encode([student_ans], normalize_embeddings=True, convert_to_numpy=True)
    cos_sim = float((ref_emb * stu_emb).sum())

    combined        = np.vstack([ref_emb, stu_emb]).astype(np.float32)
    combined_scaled = scaler.transform(combined)
    combined_pca    = pca.transform(combined_scaled)
    X = np.hstack([combined_pca[0:1], combined_pca[1:2], [[cos_sim]]]).astype(np.float32)

    proba      = model_xgb.predict_proba(X)[0]
    pred_idx   = int(proba.argmax())
    confidence = float(proba.max())
    xgb_grade  = le.classes_[pred_idx]

    if confidence >= THRESHOLD_HIGH:
        # LLM writes feedback only
        prompt = f"""You are a helpful science teacher. The student's answer is graded as: {xgb_grade}

Question  : {question}
Reference : {ref_ans}
Student   : {student_ans}

Write ONE short, constructive feedback sentence. No grade label. No preamble."""
        resp = ollama.chat(model="llama3.2:3b",
                           messages=[{"role":"user","content":prompt}],
                           options={"temperature":0.3})
        feedback = resp["message"]["content"].strip()
        return {"grade": xgb_grade, "confidence": confidence,
                "feedback": feedback, "route": "direct"}

    # RAG + LLM judge
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


def log_result(question, ref_ans, student_ans, result):
    import csv
    from datetime import datetime
    BASE     = "/Users/sambit_03/Desktop/Student Response/"
    LOG_PATH = BASE + "grading_log.csv"
    exists   = os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "timestamp","question","ref_ans","student_ans",
            "grade","confidence","feedback","route"
        ])
        if not exists: w.writeheader()
        w.writerow({
            "timestamp"  : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question"   : question, "ref_ans": ref_ans,
            "student_ans": student_ans, "grade": result["grade"],
            "confidence" : round(result["confidence"], 4),
            "feedback"   : result["feedback"], "route": result["route"]
        })


# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-block">
  <h1>🎓 Student Answer Grader</h1>
  <p>Two-stage grading · XGBoost + RAG · Llama 3.2</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs
tab_grade, tab_history = st.tabs(["Grade answer", "History"])

with tab_grade:

    question    = st.text_area("Question",         height=80,  placeholder="What happens to pitch when frequency increases?")
    ref_ans     = st.text_area("Reference answer", height=80,  placeholder="The pitch of the sound increases when frequency increases.")
    student_ans = st.text_area("Student answer",   height=80,  placeholder="Enter the student's response here...")

    grade_btn = st.button("Grade answer")

    if grade_btn:
        if not question.strip() or not ref_ans.strip() or not student_ans.strip():
            st.warning("Fill in all three fields before grading.")
        else:
            with st.spinner("Grading..."):
                result = grade_answer(question, ref_ans, student_ans)
                log_result(question, ref_ans, student_ans, result)
                st.session_state.history.insert(0, {
                    "student_ans": student_ans,
                    "result":      result
                })

            # Grade badge
            grade     = result["grade"]
            grade_cls = {
                "correct":                      "grade-correct",
                "partially_correct_incomplete": "grade-partially",
                "contradictory":                "grade-contradictory",
                "irrelevant":                   "grade-irrelevant"
            }.get(grade, "grade-irrelevant")

            grade_display = grade.replace("_", " ").title()
            route_cls     = "route-direct" if result["route"] == "direct" else "route-llm"
            route_label   = "⚡ Direct" if result["route"] == "direct" else "🤖 LLM judge"
            conf_pct      = int(result["confidence"] * 100)

            st.markdown(f"""
<div class="result-card">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:20px">
    <span class="grade-badge {grade_cls}">{grade_display}</span>
    <span class="route-pill {route_cls}">{route_label}</span>
  </div>

  <div class="result-label">Confidence</div>
  <div style="color:#E2E8F0;font-size:14px;font-weight:500">{conf_pct}%</div>
  <div class="conf-bar-bg">
    <div class="conf-bar-fill" style="width:{conf_pct}%"></div>
  </div>

  <div class="result-label" style="margin-top:20px">Feedback</div>
  <div class="feedback-text">{result['feedback']}</div>
</div>
""", unsafe_allow_html=True)


with tab_history:
    if not st.session_state.history:
        st.markdown("<p style='color:#475569;font-size:14px;padding-top:12px'>No grades yet — submit an answer to see history.</p>",
                    unsafe_allow_html=True)
    else:
        grade_colors = {
            "correct":                      "#4ade80",
            "partially_correct_incomplete": "#fb923c",
            "contradictory":                "#f87171",
            "irrelevant":                   "#a5b4fc"
        }
        for entry in st.session_state.history:
            g     = entry["result"]["grade"]
            color = grade_colors.get(g, "#94A3B8")
            conf  = int(entry["result"]["confidence"] * 100)
            label = g.replace("_"," ").title()
            route = "⚡" if entry["result"]["route"] == "direct" else "🤖"
            st.markdown(f"""
<div class="history-row">
  <span style="color:{color};font-weight:600;font-size:12px;min-width:160px">{label}</span>
  <span class="history-ans">{entry['student_ans'][:90]}</span>
  <span style="color:#475569;font-size:12px;min-width:40px">{conf}% {route}</span>
</div>
""", unsafe_allow_html=True)

        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()