"""Simple banking prediction demo. Run: streamlit run banking_prediction_app.py"""

import os

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CSV_FILE = os.path.join(os.path.dirname(__file__), "banking_operations.csv")


def train_models():
    """Train both models on Amount; evaluate on held-out rows."""
    df = pd.read_csv(CSV_FILE)
    needed = {"Amount", "Balance_After_Transaction", "Status"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {', '.join(sorted(missing))}")

    X = df[["Amount"]]
    balance = df["Balance_After_Transaction"]
    status = df["Status"]
    train_rows, test_rows = train_test_split(
        df.index, test_size=0.20, random_state=42, stratify=status
    )

    balance_model = LinearRegression()
    balance_model.fit(X.loc[train_rows], balance.loc[train_rows])
    balance_predictions = balance_model.predict(X.loc[test_rows])
    balance_r2 = r2_score(balance.loc[test_rows], balance_predictions)

    status_model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, C=10.0)
    )
    status_model.fit(X.loc[train_rows], status.loc[train_rows])
    status_predictions = status_model.predict(X.loc[test_rows])
    status_accuracy = accuracy_score(status.loc[test_rows], status_predictions)

    return df, balance_model, status_model, balance_r2, status_accuracy


def main():
    import streamlit as st

    st.set_page_config(page_title="Banking Prediction Demo", page_icon="🏦")
    st.title("🏦 Banking Prediction Demo")
    st.write("Enter a transaction amount to estimate the balance and transaction status.")

    if not os.path.isfile(CSV_FILE):
        st.error("Place banking_operations(1).csv in the same folder as this app.")
        st.stop()

    try:
        df, balance_model, status_model, balance_r2, status_accuracy = train_models()
    except (ValueError, KeyError, pd.errors.ParserError) as error:
        st.error(f"Could not train the models: {error}")
        st.stop()

    amount = st.number_input(
        "Transaction amount (₹)", min_value=0, value=10000, step=500
    )

    if st.button("Predict"):
        single_record = pd.DataFrame({"Amount": [amount]})
        predicted_balance = balance_model.predict(single_record)[0]
        predicted_status = status_model.predict(single_record)[0]

        st.subheader("Prediction")
        st.metric("Estimated balance after transaction", f"₹{predicted_balance:,.2f}")
        st.metric("Predicted transaction status", predicted_status)

        if amount < df["Amount"].min() or amount > df["Amount"].max():
            st.warning("This amount is outside the range in the training data.")

    with st.expander("Model performance on 10 held-out records"):
        st.write(f"Linear Regression test R²: **{balance_r2:.2%}**")
        st.write(f"Logistic Regression test accuracy: **{status_accuracy:.0%}**")
        st.caption(
            "R² measures regression fit; accuracy measures correct status labels. "
            "This is a small, adjusted teaching dataset, so results do not "
            "represent production performance."
        )


if __name__ == "__main__":
    main()
