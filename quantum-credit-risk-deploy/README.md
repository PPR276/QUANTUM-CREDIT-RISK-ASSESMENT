# Quantum Machine Learning-Based Credit Risk Assessment

A Streamlit web application based on the supplied `quantum capstone bank loan approval.ipynb` notebook. It demonstrates a hybrid classical/quantum workflow for credit-risk classification using the German Credit dataset.

## Features

- Loads the German Credit dataset from OpenML.
- Applies one-hot encoding, a stratified 150-record training subset and 50-record test subset, standardization, PCA to four components, and angle scaling to `[0, π]`.
- Trains classical baselines (Logistic Regression, RBF SVM, Random Forest), QSVC, and VQC.
- Displays benchmark metrics, illustrative expected-loss comparisons, and confusion matrices.
- Offers an interactive borrower-input demo and model predictions.

## Run locally

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The first startup downloads the dataset and trains the models. It can take a few minutes. The trained models are cached for the current app process.

## Deploy with GitHub + Streamlit Community Cloud

1. Create a new GitHub repository, for example `quantum-credit-risk-assessment`.
2. Upload `app.py`, `requirements.txt`, and `README.md` from this folder to the repository root. You can also upload the original notebook as a project reference.
3. Open Streamlit Community Cloud and sign in with GitHub.
4. Choose **Create app**, select your repository and branch, and set the main file path to `app.py`.
5. Deploy and review the app logs if package installation or model initialization fails.

## Notes and limitations

- The dataset is downloaded from OpenML at runtime, so deployment requires outbound network access during first startup.
- Quantum kernel training can be relatively expensive on a shared free host. If startup times out, use a host with more resources or consider saving fitted model/preprocessing artifacts after a verified training run.
- The notebook uses a small 150/50 training/test subset for quantum simulation. Metrics can vary if quantum optimization is stochastic or package behavior differs.
- The dashboard is an academic prototype, not financial advice or a production credit-underwriting service. Do not use it as the sole basis for lending decisions.
- The manual-input demo has only a subset of the German Credit fields; unprovided categorical features are set to the all-zero baseline of the notebook's one-hot encoding. A production system should collect all required features and validate its preprocessing carefully.
