"""
Student Pass / Fail Classification - Streamlit Frontend
--------------------------------------------------------
Run:  streamlit run app.py

Files needed in the same folder:
    model.pkl   -> your trained classifier (saved with joblib.dump)
    scaler.pkl  -> (optional) the scaler used during training

If model.pkl is not found, the app runs in DEMO mode with a simple
rule-based predictor so you can still test the UI.
"""

import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Plotly is optional: used only for the gauge chart
try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# ----------------------------------------------------------------------
# 1. CONFIGURATION  (change this section to match YOUR dataset)
# ----------------------------------------------------------------------
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

# Order of the features MUST match the order used while training the model.
# (name, label, min, max, default, step, help_text)
FEATURES = [
    ("study_hours",      "Study Hours per Day",      0.0,  12.0,  3.0, 0.5, "Average hours studied each day"),
    ("attendance",       "Attendance (%)",           0.0, 100.0, 75.0, 1.0, "Overall class attendance"),
    ("previous_marks",   "Previous Exam Marks (%)",  0.0, 100.0, 60.0, 1.0, "Marks in the last exam"),
    ("assignment_score", "Assignment Score (%)",     0.0, 100.0, 70.0, 1.0, "Average assignment marks"),
    ("sleep_hours",      "Sleep Hours per Day",      0.0,  12.0,  7.0, 0.5, "Average sleep per night"),
]

PASS_LABEL = 1          # value your model returns for "Pass"
PASS_THRESHOLD = 0.50   # probability above which student is called Pass

# ----------------------------------------------------------------------
# 2. PAGE SETUP + CUSTOM CSS
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Student Pass/Fail Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.4rem; font-weight: 800; text-align: center;
            background: linear-gradient(90deg, #4f46e5, #06b6d4);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 0;
        }
        .sub-title {
            text-align: center; color: #6b7280; font-size: 1.05rem;
            margin-top: 0; margin-bottom: 1.5rem;
        }
        .result-card {
            padding: 1.6rem; border-radius: 16px; text-align: center;
            color: white; box-shadow: 0 6px 18px rgba(0,0,0,0.15);
        }
        .pass-card { background: linear-gradient(135deg, #16a34a, #22c55e); }
        .fail-card { background: linear-gradient(135deg, #dc2626, #f97316); }
        .result-card h1 { margin: 0; font-size: 2.6rem; }
        .result-card p  { margin: 0.3rem 0 0 0; font-size: 1.1rem; opacity: 0.95; }
        .tip-box {
            background: #f1f5f9; border-left: 5px solid #4f46e5;
            padding: 0.8rem 1rem; border-radius: 8px; margin-bottom: 0.5rem;
            color: #1e293b;
        }
        div.stButton > button {
            width: 100%; border-radius: 10px; height: 3rem;
            font-weight: 700; font-size: 1.05rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# 3. MODEL LOADING
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    return model, scaler


model, scaler = load_artifacts()
DEMO_MODE = model is None
FEATURE_NAMES = [f[0] for f in FEATURES]


def predict(df: pd.DataFrame):
    """Return (labels, pass_probabilities) for a dataframe of features."""
    X = df[FEATURE_NAMES].astype(float).values

    if DEMO_MODE:
        # Simple weighted score, only for testing the UI
        score = (
            0.20 * (df["study_hours"] / 12)
            + 0.25 * (df["attendance"] / 100)
            + 0.30 * (df["previous_marks"] / 100)
            + 0.20 * (df["assignment_score"] / 100)
            + 0.05 * (1 - (df["sleep_hours"] - 7).abs() / 7).clip(0, 1)
        ).values
        proba = np.clip(score * 1.15, 0, 1)
    else:
        if scaler is not None:
            X = scaler.transform(X)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, list(model.classes_).index(PASS_LABEL)]
        else:
            proba = (model.predict(X) == PASS_LABEL).astype(float)

    labels = np.where(proba >= PASS_THRESHOLD, "Pass", "Fail")
    return labels, proba


def make_gauge(prob: float):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%"},
            title={"text": "Probability of Passing"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#4f46e5"},
                "steps": [
                    {"range": [0, 40], "color": "#fecaca"},
                    {"range": [40, 60], "color": "#fde68a"},
                    {"range": [60, 100], "color": "#bbf7d0"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "value": PASS_THRESHOLD * 100,
                },
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=60, b=10))
    return fig


def suggestions(row: dict):
    """Simple, rule-based improvement tips shown to the student."""
    tips = []
    if row["attendance"] < 75:
        tips.append("📅 Attendance 75% se kam hai. Classes regularly attend karein.")
    if row["study_hours"] < 3:
        tips.append("📚 Daily study hours badhayein (kam se kam 3-4 ghante).")
    if row["previous_marks"] < 50:
        tips.append("📝 Pichhle exam ke weak topics ko dobara revise karein.")
    if row["assignment_score"] < 60:
        tips.append("🗂️ Assignments time par aur dhyan se complete karein.")
    if row["sleep_hours"] < 6 or row["sleep_hours"] > 9:
        tips.append("😴 Sleep 6-9 ghante ke beech rakhein, isse focus behtar hota hai.")
    if not tips:
        tips.append("✅ Sab kuch achha chal raha hai. Isi tarah consistent rahein!")
    return tips


# ----------------------------------------------------------------------
# 4. SESSION STATE (prediction history)
# ----------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------------------------------------------------
# 5. SIDEBAR
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎓 Navigation")
    page = st.radio(
        "Go to",
        ["🏠 Home", "🔮 Single Prediction", "📂 Batch Prediction", "🕘 History", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    if DEMO_MODE:
        st.warning("Demo mode: `model.pkl` nahi mila.")
    else:
        st.success("Model loaded ✔")
    st.caption(f"Pass threshold: {int(PASS_THRESHOLD * 100)}%")

# ----------------------------------------------------------------------
# 6. HEADER
# ----------------------------------------------------------------------
st.markdown('<p class="main-title">🎓 Student Pass / Fail Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Machine Learning based classification of student performance</p>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# 7. PAGES
# ----------------------------------------------------------------------
# ---------- HOME ----------
if page == "🏠 Home":
    c1, c2, c3 = st.columns(3)
    c1.metric("Input Features", len(FEATURES))
    c2.metric("Output Classes", "Pass / Fail")
    c3.metric("Predictions Made", len(st.session_state.history))

    st.markdown("### How it works")
    st.markdown(
        """
        1. **Single Prediction** mein student ki details enter karein.
        2. Model result batata hai: **Pass** ya **Fail**, saath mein probability.
        3. **Batch Prediction** mein CSV upload karke poori class ka result nikaalein.
        4. **History** mein pichhli predictions dekhein aur download karein.
        """
    )

    st.markdown("### Features used")
    st.dataframe(
        pd.DataFrame(
            [{"Feature": f[1], "Min": f[2], "Max": f[3], "Description": f[6]} for f in FEATURES]
        ),
        use_container_width=True,
        hide_index=True,
    )

# ---------- SINGLE PREDICTION ----------
elif page == "🔮 Single Prediction":
    st.subheader("Enter Student Details")

    with st.form("predict_form"):
        student_name = st.text_input("Student Name (optional)", placeholder="e.g. Rahul Sharma")

        values = {}
        cols = st.columns(2)
        for i, (key, label, lo, hi, default, step, help_txt) in enumerate(FEATURES):
            with cols[i % 2]:
                values[key] = st.slider(label, float(lo), float(hi), float(default), float(step), help=help_txt)

        submitted = st.form_submit_button("🔮 Predict Result")

    if submitted:
        df_in = pd.DataFrame([values])
        labels, proba = predict(df_in)
        label, prob = labels[0], float(proba[0])

        st.markdown("---")
        left, right = st.columns([1, 1])

        with left:
            css = "pass-card" if label == "Pass" else "fail-card"
            emoji = "🎉" if label == "Pass" else "⚠️"
            name_txt = student_name if student_name else "Student"
            st.markdown(
                f"""
                <div class="result-card {css}">
                    <h1>{emoji} {label.upper()}</h1>
                    <p>{name_txt} ka predicted result</p>
                    <p><b>Confidence (Pass): {prob * 100:.1f}%</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("#### 💡 Suggestions")
            for t in suggestions(values):
                st.markdown(f'<div class="tip-box">{t}</div>', unsafe_allow_html=True)

        with right:
            if PLOTLY_OK:
                st.plotly_chart(make_gauge(prob), use_container_width=True)
            else:
                st.progress(prob, text=f"Probability of passing: {prob * 100:.1f}%")

            st.markdown("#### Input Summary")
            st.dataframe(
                pd.DataFrame({"Feature": [f[1] for f in FEATURES], "Value": [values[f[0]] for f in FEATURES]}),
                use_container_width=True,
                hide_index=True,
            )

        st.session_state.history.append(
            {
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Name": student_name or "-",
                **values,
                "Result": label,
                "Pass Probability (%)": round(prob * 100, 2),
            }
        )

# ---------- BATCH PREDICTION ----------
elif page == "📂 Batch Prediction":
    st.subheader("Upload CSV for Bulk Prediction")
    st.caption("CSV mein ye columns hone chahiye: " + ", ".join(f"`{n}`" for n in FEATURE_NAMES))

    template = pd.DataFrame([{f[0]: f[4] for f in FEATURES}])
    st.download_button(
        "⬇️ Download CSV Template",
        template.to_csv(index=False).encode("utf-8"),
        file_name="student_template.csv",
        mime="text/csv",
    )

    file = st.file_uploader("Upload your CSV file", type=["csv"])

    if file is not None:
        try:
            data = pd.read_csv(file)
        except Exception as e:
            st.error(f"CSV read nahi ho paayi: {e}")
            st.stop()

        missing = [c for c in FEATURE_NAMES if c not in data.columns]
        if missing:
            st.error(f"Ye columns missing hain: {missing}")
        else:
            data = data.dropna(subset=FEATURE_NAMES).reset_index(drop=True)
            labels, proba = predict(data)
            data["Predicted Result"] = labels
            data["Pass Probability (%)"] = (proba * 100).round(2)

            total = len(data)
            passed = int((data["Predicted Result"] == "Pass").sum())
            failed = total - passed

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Students", total)
            m2.metric("Predicted Pass", passed)
            m3.metric("Predicted Fail", failed)
            m4.metric("Pass Rate", f"{(passed / total * 100) if total else 0:.1f}%")

            tab1, tab2 = st.tabs(["📋 Results Table", "📊 Charts"])
            with tab1:
                st.dataframe(data, use_container_width=True)
            with tab2:
                st.bar_chart(data["Predicted Result"].value_counts())
                st.caption("Pass vs Fail count")

            st.download_button(
                "⬇️ Download Predictions",
                data.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )

# ---------- HISTORY ----------
elif page == "🕘 History":
    st.subheader("Prediction History (this session)")
    if st.session_state.history:
        hist = pd.DataFrame(st.session_state.history)
        st.dataframe(hist, use_container_width=True)
        col_a, col_b = st.columns(2)
        col_a.download_button(
            "⬇️ Download History",
            hist.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv",
        )
        if col_b.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("Abhi tak koi prediction nahi hui hai.")

# ---------- ABOUT ----------
else:
    st.subheader("About this Project")
    st.markdown(
        """
        **Project:** Student Pass / Fail Classification

        **Goal:** Student ke academic aur lifestyle factors se predict karna ki
        wo exam mein pass hoga ya fail.

        **Tech Stack**
        - Python, Pandas, NumPy
        - Scikit-learn (classification model)
        - Streamlit (frontend)
        - Plotly (charts)

        **Model Details**
        - Type: Binary classification (Pass = 1, Fail = 0)
        - Output: Class label + probability of passing

        > Is section mein apne model ka naam (Logistic Regression / Random Forest etc.),
        > accuracy, dataset ka source aur team members ke naam add kar dein.
        """
    )

st.markdown("---")
st.caption("Made with ❤️ using Streamlit")

