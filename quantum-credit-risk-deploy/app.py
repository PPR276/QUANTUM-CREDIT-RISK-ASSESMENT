import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC, VQC
from qiskit_algorithms.optimizers import COBYLA

st.set_page_config(page_title="Quantum Credit Risk Assessment", page_icon="🏦", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.8rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background: #172338; padding: 14px; border-radius: 10px;}
.risk-note {padding: 14px; border-radius: 8px; background: #172338; border-left: 4px solid #38bdf8;}
</style>
""", unsafe_allow_html=True)

st.title("🏦 Quantum Credit Risk Assessment")
st.caption("Hybrid quantum–classical credit risk demonstration using the German Credit dataset")
st.warning("Academic demonstration only — predictions are experimental and must not be used as the sole basis for real lending decisions.")

@st.cache_resource(show_spinner="Loading the German Credit dataset and training models. The first run may take a few minutes...")
def build_models():
    # Same source and general preprocessing pipeline as the supplied notebook.
    credit_data = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
    df = credit_data.frame.copy()
    df["target"] = df["class"].map({"good": 0, "bad": 1})
    X_raw = df.drop(columns=["class", "target"])
    y_raw = df["target"].astype(int).values
    X_encoded = pd.get_dummies(X_raw, drop_first=True)

    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X_encoded, y_raw, test_size=0.25, random_state=42, stratify=y_raw
    )
    train_good = X_train_full[y_train_full == 0].iloc[:105]
    train_bad = X_train_full[y_train_full == 1].iloc[:45]
    X_train_sub = pd.concat([train_good, train_bad])
    y_train = np.array([0] * len(train_good) + [1] * len(train_bad))
    test_good = X_test_full[y_test_full == 0].iloc[:35]
    test_bad = X_test_full[y_test_full == 1].iloc[:15]
    X_test_sub = pd.concat([test_good, test_bad])
    y_test = np.array([0] * len(test_good) + [1] * len(test_bad))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sub)
    X_test_scaled = scaler.transform(X_test_sub)
    pca = PCA(n_components=4, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    minmax = MinMaxScaler(feature_range=(0, np.pi))
    X_train = minmax.fit_transform(X_train_pca)
    X_test = minmax.transform(X_test_pca)

    classical_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Classical SVM (RBF)": SVC(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }
    models = {}
    predictions = {}
    metrics = []
    for name, model in classical_models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        models[name] = model
        predictions[name] = pred
        metrics.append({"Model": name, "Accuracy": accuracy_score(y_test, pred),
                        "F1-score": f1_score(y_test, pred, zero_division=0),
                        "Default recall": recall_score(y_test, pred, zero_division=0)})

    feature_map = ZZFeatureMap(feature_dimension=4, reps=2, entanglement="linear")
    qsvc = QSVC(quantum_kernel=FidelityQuantumKernel(feature_map=feature_map))
    start = time.time()
    qsvc.fit(X_train, y_train)
    qsvc_seconds = time.time() - start
    pred_qsvc = np.asarray(qsvc.predict(X_test)).ravel().astype(int)
    models["QSVC (Quantum Kernel)"] = qsvc
    predictions["QSVC (Quantum Kernel)"] = pred_qsvc
    metrics.append({"Model": "QSVC (Quantum Kernel)", "Accuracy": accuracy_score(y_test, pred_qsvc),
                    "F1-score": f1_score(y_test, pred_qsvc, zero_division=0),
                    "Default recall": recall_score(y_test, pred_qsvc, zero_division=0)})

    ansatz = RealAmplitudes(num_qubits=4, reps=2, entanglement="linear")
    vqc = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=COBYLA(maxiter=40))
    start = time.time()
    vqc.fit(X_train, y_train)
    vqc_seconds = time.time() - start
    pred_vqc = np.asarray(vqc.predict(X_test)).ravel().astype(int)
    models["VQC (Quantum Neural Net)"] = vqc
    predictions["VQC (Quantum Neural Net)"] = pred_vqc
    metrics.append({"Model": "VQC (Quantum Neural Net)", "Accuracy": accuracy_score(y_test, pred_vqc),
                    "F1-score": f1_score(y_test, pred_vqc, zero_division=0),
                    "Default recall": recall_score(y_test, pred_vqc, zero_division=0)})

    return {
        "df": df, "X_encoded": X_encoded, "scaler": scaler, "pca": pca, "minmax": minmax,
        "models": models, "predictions": predictions, "y_test": y_test,
        "X_train": X_train, "X_test": X_test, "y_train": y_train,
        "metrics": pd.DataFrame(metrics), "qsvc_seconds": qsvc_seconds, "vqc_seconds": vqc_seconds,
        "explained_variance": float(np.sum(pca.explained_variance_ratio_) * 100),
    }

with st.sidebar:
    st.header("Project overview")
    st.write("**Dataset:** German Credit (OpenML)")
    st.write("**Quantum register:** 4 qubits")
    st.write("**Feature map:** ZZFeatureMap")
    st.write("**Quantum classifiers:** QSVC and VQC")
    st.write("**VQC optimizer:** COBYLA, maxiter=40")
    st.caption("Models are trained once per running app instance and cached for subsequent interactions.")

try:
    with st.spinner("Preparing models…"):
        bundle = build_models()
except Exception as exc:
    st.error("The models could not be initialized. Check the deployment logs and package versions.")
    st.exception(exc)
    st.stop()

metrics = bundle["metrics"].copy()
metrics_display = metrics.copy()
for col in ["Accuracy", "F1-score", "Default recall"]:
    metrics_display[col] = metrics_display[col].map(lambda x: f"{x:.1%}" if col != "F1-score" else f"{x:.4f}")

st.subheader("Model benchmark")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Training records", len(bundle["X_train"]))
c2.metric("Test records", len(bundle["X_test"]))
c3.metric("PCA variance retained", f"{bundle['explained_variance']:.2f}%")
c4.metric("Quantum models", "2")
st.dataframe(metrics_display, use_container_width=True, hide_index=True)

left, right = st.columns(2)
with left:
    st.markdown("#### Accuracy, F1-score and default recall")
    plot_df = metrics.set_index("Model")[["Accuracy", "F1-score", "Default recall"]]
    st.bar_chart(plot_df, y=["Accuracy", "F1-score", "Default recall"])
with right:
    st.markdown("#### Expected loss on the test subset")
    losses = []
    for name, pred in bundle["predictions"].items():
        tn, fp, fn, tp = confusion_matrix(bundle["y_test"], pred, labels=[0, 1]).ravel()
        losses.append({"Model": name, "Expected loss ($)": int(fn * 5000 + fp * 500)})
    loss_df = pd.DataFrame(losses).set_index("Model")
    st.bar_chart(loss_df)
    st.caption("Illustrative cost assumptions from the notebook: $5,000 per missed default and $500 per wrongly rejected good borrower.")

st.divider()
st.subheader("Interactive borrower assessment")
st.write("Enter a sample borrower profile to run it through the trained models.")
with st.form("borrower_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        loan_amount = st.number_input("Loan amount (USD)", min_value=0.0, value=4000.0, step=500.0)
    with col2:
        duration_months = st.number_input("Loan duration (months)", min_value=1, max_value=120, value=24, step=1)
    with col3:
        age = st.number_input("Borrower age (years)", min_value=18, max_value=100, value=32, step=1)
    col4, col5 = st.columns(2)
    with col4:
        checking_status = st.selectbox("Checking account status", ["<0", "0<=X<200", ">=200", "no checking"])
    with col5:
        savings_status = st.selectbox("Savings account balance", ["<100", "100<=X<500", "500<=X<1000", ">=1000", "no known savings"])
    submitted = st.form_submit_button("Assess borrower", type="primary", use_container_width=True)

if submitted:
    row = pd.DataFrame(0, index=[0], columns=bundle["X_encoded"].columns)
    for col, val in [("duration", duration_months), ("credit_amount", loan_amount), ("age", age)]:
        if col in row.columns:
            row[col] = val
    for prefix, val in [("checking_status", checking_status), ("savings_status", savings_status)]:
        encoded_col = f"{prefix}_{val}"
        if encoded_col in row.columns:
            row[encoded_col] = 1
    cust_scaled = bundle["scaler"].transform(row)
    cust_pca = bundle["pca"].transform(cust_scaled)
    cust_angles = bundle["minmax"].transform(cust_pca)

    st.markdown("### Assessment report")
    m1, m2, m3 = st.columns(3)
    m1.metric("Loan amount", f"${loan_amount:,.0f}")
    m2.metric("Duration", f"{int(duration_months)} months")
    m3.metric("Borrower age", f"{int(age)} years")
    st.write("**Quantum rotation angles (θ₀, θ₁, θ₂, θ₃):**", ", ".join(f"{x:.3f}" for x in cust_angles[0]))

    verdicts = []
    for name in ["VQC (Quantum Neural Net)", "QSVC (Quantum Kernel)", "Random Forest"]:
        pred = int(np.asarray(bundle["models"][name].predict(cust_angles)).ravel()[0])
        verdict = "High risk — default predicted" if pred == 1 else "Lower risk — non-default predicted"
        verdicts.append({"Model": name, "Prediction": verdict})
    verdict_df = pd.DataFrame(verdicts)
    st.dataframe(verdict_df, use_container_width=True, hide_index=True)
    vqc_pred = int(np.asarray(bundle["models"]["VQC (Quantum Neural Net)"].predict(cust_angles)).ravel()[0])
    if vqc_pred == 1:
        st.error("VQC flags this profile as high risk. Consider manual review; this is not a definitive lending decision.")
    else:
        st.success("VQC classifies this profile as a non-default. Verify all details and apply normal underwriting checks.")

    st.info("This demonstration maps unprovided categorical attributes to the all-zero baseline of the notebook's one-hot encoding. A production system should collect all required dataset features and use a validated preprocessing pipeline.")

with st.expander("Confusion matrix details"):
    chosen = st.selectbox("Select model", list(bundle["predictions"].keys()))
    cm = confusion_matrix(bundle["y_test"], bundle["predictions"][chosen], labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=["Actual non-default", "Actual default"], columns=["Predicted non-default", "Predicted default"])
    st.dataframe(cm_df, use_container_width=True)

st.caption("Prototype for academic evaluation. The quantum classifiers use simulator-backed quantum circuits; this deployment does not claim execution on a physical quantum processor.")
