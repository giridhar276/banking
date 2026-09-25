"""Predict loan default with a pipeline loaded directly from MLflow."""
import pandas as pd
import streamlit as st
import mlflow
import mlflow.sklearn

DATA_FILE = "Banking_Loan_Default_Classification(2).csv"
EXPERIMENTS = {
    "Decision Tree": "Banking_Loan_Default_Decision_Tree",
    "Random Forest": "Banking_Loan_Default_Random_Forest",
}
MODEL_NAME = "loan_default_pipeline"
LABELS = {
    "age": "Age (years)",
    "monthly_income_inr": "Monthly income (INR)",
    "employment_type": "Employment type",
    "years_employed": "Years employed",
    "education_level": "Education level",
    "marital_status": "Marital status",
    "dependents": "Number of dependents",
    "residence_type": "Residence type",
    "years_at_address": "Years at current address",
    "loan_amount_inr": "Loan amount (INR)",
    "loan_term_months": "Loan term (months)",
    "loan_purpose": "Loan purpose",
    "existing_loans": "Number of existing loans",
    "monthly_debt_payments_inr": "Monthly debt payments (INR)",
    "credit_score": "Credit score",
    "late_payments_last_12m": "Late payments in past 12 months",
    "savings_balance_inr": "Savings balance (INR)",
    "account_tenure_years": "Years with the bank",
    "has_guarantor": "Has a guarantor?",
}

st.set_page_config(page_title="Banking Loan Default | MLflow", layout="wide")
st.title("Banking loan default prediction")
st.caption("The prediction uses a trained pipeline loaded from your local MLflow server.")

@st.cache_data
def read_reference_data():
    return pd.read_csv(DATA_FILE)

@st.cache_resource(show_spinner="Loading model from MLflow...")
def load_pipeline(tracking_uri, model_uri):
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.sklearn.load_model(model_uri)

try:
    reference = read_reference_data()
except Exception as error:
    st.error(f"Could not read {DATA_FILE}: {error}")
    st.stop()

features = reference.drop(columns="loan_default", errors="ignore")
with st.sidebar:
    st.header("MLflow model")
    tracking_uri = st.text_input("Tracking server", value="http://127.0.0.1:5000")
    algorithm = st.selectbox("Algorithm", list(EXPERIMENTS))
    st.caption("Select a completed run or paste the model URI shown in MLflow's model Overview.")

    try:
        mlflow.set_tracking_uri(tracking_uri)
        experiment = mlflow.get_experiment_by_name(EXPERIMENTS[algorithm])
        if experiment is None:
            raise ValueError(f"Experiment '{EXPERIMENTS[algorithm]}' was not found.")
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            max_results=1000,
            order_by=["start_time DESC"],
        )
        if runs.empty:
            raise ValueError("No completed runs are available in this experiment.")
        if "metrics.valid_f1" in runs:
            runs = runs[runs["metrics.valid_f1"].notna()].copy()
        if runs.empty:
            raise ValueError("No completed runs have validation F1 metrics.")
        run_names = runs.get("tags.mlflow.runName", pd.Series(index=runs.index, dtype=object))
        run_names = run_names.fillna("unnamed run")
        choices = [
            (row.run_id, str(run_names.loc[index]), float(row["metrics.valid_f1"]))
            for index, row in runs.iterrows()
        ]
        chosen = st.selectbox(
            "Completed run",
            choices,
            format_func=lambda item: f"{item[1]} | F1 {item[2]:.3f} | {item[0][:8]}",
        )
        default_uri = f"runs:/{chosen[0]}/{MODEL_NAME}"
        st.caption("The run URI works with these notebooks. In MLflow 3, the model URI from Overview is preferred.")
    except Exception as error:
        st.warning(f"Run list unavailable: {error}")
        default_uri = ""

    model_uri = st.text_input(
        "Model URI",
        value=default_uri,
        help="Paste an MLflow URI such as models:/m-... from the model Overview, or use the selected run URI.",
    ).strip()
    threshold = st.slider("Default probability threshold", 0.05, 0.95, 0.50, 0.05)
    st.caption("This is a classroom example. A bank must validate a threshold for its own costs and policy.")

if not model_uri:
    st.info("Start the MLflow server, select a completed run, or paste a model URI in the sidebar.")
    st.stop()

try:
    pipeline = load_pipeline(tracking_uri, model_uri)
except Exception as error:
    st.error(f"Could not load model from MLflow: {error}")
    st.info("Check that the server is running, the URI points to a saved model, and this environment has compatible packages.")
    st.stop()

st.subheader("Enter an applicant profile")
st.caption("Suggested values and dropdown choices come from the bundled example CSV. Enter a new profile below.")
values = {}
with st.form("loan_application"):
    columns = st.columns(2)
    for index, field in enumerate(features.columns):
        label = LABELS.get(field, field.replace("_", " ").title())
        with columns[index % 2]:
            if pd.api.types.is_numeric_dtype(features[field]):
                median = float(features[field].median())
                # All numeric inputs are floats, matching the updated MLflow notebook schema.
                values[field] = st.number_input(
                    label, min_value=0.0, value=median, step=1.0 if median < 100 else 100.0,
                    key=field,
                )
            else:
                choices = sorted(features[field].dropna().astype(str).unique().tolist())
                values[field] = st.selectbox(label, choices, key=field)
    submitted = st.form_submit_button("Predict loan default", type="primary")

if submitted:
    applicant = pd.DataFrame([values], columns=features.columns)
    try:
        probability = float(pipeline.predict_proba(applicant)[0, 1])
    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.stop()
    predicted = int(probability >= threshold)
    st.subheader("Prediction")
    st.metric("Estimated default probability", f"{probability:.1%}")
    if predicted:
        st.warning(f"At the {threshold:.0%} threshold: predicted default (class 1).")
    else:
        st.success(f"At the {threshold:.0%} threshold: predicted no default (class 0).")
    st.caption("Probability is a model estimate, not a lending decision. This app does not retrain the model.")
    with st.expander("Input values sent to the model"):
        st.dataframe(applicant.T.rename(columns={0: "Value"}), use_container_width=True)
