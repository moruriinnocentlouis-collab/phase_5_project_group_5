import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SACCO Financial Analysis",
    page_icon="",
    layout="wide"
)

st.title(" SACCO Financial Analysis and Surplus Prediction")
st.markdown("Upload your cleaned SACCO CSV files to analyse trends and predict financial surplus.")

# ── Constants ──────────────────────────────────────────────────────────────────
FEATURE_ORDER = [
    'Total Assets',
    'Total Deposit liabilities',
    'Total Equity',
    'Total Liabilities',
    'Cash & Cash Equivalent',
    'Financial Investments',
    'Gross Loan Portfolio',
    'Net Loan Portfolio',
    'Share Capital',
    'Statutory Reserve'
]
TARGET = "Current Year's Surplus"

# ── Helper functions ───────────────────────────────────────────────────────────
@st.cache_data
def load_and_process(files):
    """Load uploaded CSVs and reshape to long + model format."""
    df_list = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip()
        df = df.loc[:, ~df.columns.str.contains('Unnamed')]
        sacco_name = os.path.splitext(f.name)[0]
        df['SACCO'] = sacco_name
        df_list.append(df)

    df_all = pd.concat(df_list, ignore_index=True)
    valid_cols = ['Account_Name', 'SACCO'] + [
        col for col in df_all.columns if col.isdigit()
    ]
    df_all = df_all[[c for c in valid_cols if c in df_all.columns]]

    df_long = df_all.melt(
        id_vars=['Account_Name', 'SACCO'],
        var_name='Year',
        value_name='Value'
    )
    df_long['Value'] = (
        df_long['Value'].astype(str)
        .str.replace(',', '', regex=True)
        .str.replace('-', '', regex=True)
    )
    df_long['Value'] = pd.to_numeric(df_long['Value'], errors='coerce')

    df_model = df_long.pivot_table(
        index=['SACCO', 'Year'],
        columns='Account_Name',
        values='Value',
        aggfunc='sum'
    ).reset_index()

    return df_long, df_model


@st.cache_resource
def load_pretrained_model():
    """Load the model saved from the notebook, if it exists."""
    try:
        return joblib.load('random_forest_model.pkl')
    except FileNotFoundError:
        return None


def predict_surplus(model, input_data, features):
    """Apply the same log transform used during training, then predict."""
    input_df = pd.DataFrame([input_data])[features]
    input_log = np.log1p(input_df)
    pred_log = model.predict(input_log)
    return np.expm1(pred_log)[0]


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header(" Data Upload")
uploaded_files = st.sidebar.file_uploader(
    "Upload cleaned SACCO CSV files",
    type="csv",
    accept_multiple_files=True
)

# Load data
df_long, df_model = None, None
if uploaded_files:
    with st.spinner("Processing files..."):
        df_long, df_model = load_and_process(uploaded_files)
    st.sidebar.success(f" Loaded {len(uploaded_files)} SACCO files")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([" Visualisations", " Model Training", " Predict Surplus"])


# ════════════════════════════════════════════════════════
# TAB 1 — VISUALISATIONS
# ════════════════════════════════════════════════════════
with tab1:
    if df_long is None:
        st.info(" Upload your CSV files in the sidebar to begin.")
    else:
        # Summary metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Total SACCOs", df_long['SACCO'].nunique())
        c2.metric("Years Covered", df_long['Year'].nunique())
        c3.metric("Financial Accounts", df_long['Account_Name'].nunique())

        st.markdown("---")

        # Overall trend
        st.subheader("Overall SACCO Financial Trend")
        trend = df_long.groupby('Year')['Value'].sum().sort_index()
        fig, ax = plt.subplots(figsize=(10, 4))
        trend.plot(ax=ax, marker='o', color='#2196F3')
        ax.set_xlabel('Year'); ax.set_ylabel('Total Value (KES)')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig); plt.close()
        st.caption(
            "Financial performance remained relatively stable until 2017, followed by a significant "
            "surge between 2019 and 2022, indicating rapid sector expansion."
        )

        st.markdown("---")

        # Top & bottom SACCOs
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Top 10 SACCOs")
            top = df_long.groupby('SACCO')['Value'].sum().nlargest(10)
            fig, ax = plt.subplots(figsize=(6, 4))
            top.plot(kind='bar', ax=ax, color='#4CAF50')
            ax.set_xticklabels(top.index, rotation=40, ha='right', fontsize=8)
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig); plt.close()

        with col_b:
            st.subheader("Bottom 10 SACCOs")
            bottom = df_long.groupby('SACCO')['Value'].sum().nsmallest(10)
            fig, ax = plt.subplots(figsize=(6, 4))
            bottom.plot(kind='bar', ax=ax, color='#F44336')
            ax.set_xticklabels(bottom.index, rotation=40, ha='right', fontsize=8)
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig); plt.close()

        st.markdown("---")

        # Financial components
        st.subheader("Top Financial Components")
        top_acc = df_long.groupby('Account_Name')['Value'].sum().nlargest(10)
        fig, ax = plt.subplots(figsize=(10, 4))
        top_acc.plot(kind='bar', ax=ax, color='#9C27B0')
        ax.set_xticklabels(top_acc.index, rotation=40, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig); plt.close()

        st.markdown("---")

        # Loans vs Deposits
        st.subheader("Loans vs Deposits Over Time")
        pivot = df_long.pivot_table(index='Year', columns='Account_Name', values='Value')
        loan_col, dep_col = 'Gross Loan Portfolio', 'Total Deposit liabilities'
        available = [c for c in [loan_col, dep_col] if c in pivot.columns]
        if available:
            fig, ax = plt.subplots(figsize=(10, 4))
            pivot[available].sort_index().plot(ax=ax, marker='o')
            ax.set_ylabel('Value (KES)'); ax.grid(True, alpha=0.3)
            st.pyplot(fig); plt.close()
            st.caption(
                "Loans and deposits move in near-perfect lockstep, indicating a disciplined "
                "intermediation model. A widening gap from 2018 suggests a loan-to-deposit ratio above 100%."
            )


# ════════════════════════════════════════════════════════
# TAB 2 — MODEL TRAINING
# ════════════════════════════════════════════════════════
with tab2:
    if df_model is None:
        st.info(" Upload your CSV files in the sidebar first.")
    else:
        available_features = [f for f in FEATURE_ORDER if f in df_model.columns]

        if TARGET not in df_model.columns:
            st.error(f"Target column '{TARGET}' not found in data.")
        elif len(available_features) < 3:
            st.error("Not enough feature columns found. Check your CSV structure.")
        else:
            st.subheader("Model Configuration")
            col1, col2 = st.columns(2)
            with col1:
                selected_model = st.selectbox(
                    "Choose a model",
                    ["Random Forest", "Linear Regression", "Ridge Regression", "Lasso Regression"]
                )
            with col2:
                test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)

            if st.button(" Train Model", type="primary"):
                X = df_model[available_features].apply(pd.to_numeric, errors='coerce')
                y = pd.to_numeric(df_model[TARGET], errors='coerce')
                X_log = np.log1p(X.fillna(X.median()))
                y_log = np.log1p(y.fillna(y.median()))

                X_train, X_test, y_train, y_test = train_test_split(
                    X_log, y_log, test_size=test_size, random_state=42
                )

                model_map = {
                    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                    "Linear Regression": LinearRegression(),
                    "Ridge Regression": Ridge(alpha=1.0),
                    "Lasso Regression": Lasso(alpha=0.01),
                }
                model = model_map[selected_model]

                with st.spinner(f"Training {selected_model}..."):
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                r2  = r2_score(y_test, y_pred)

                # Store in session state so Predict tab can use it
                st.session_state['trained_model'] = model
                st.session_state['trained_features'] = available_features

                st.success(f" {selected_model} trained!")
                m1, m2, m3 = st.columns(3)
                m1.metric("MAE (log scale)", f"{mae:.4f}")
                m2.metric("MSE (log scale)", f"{mse:.4f}")
                m3.metric("R² Score", f"{r2:.4f}")

                # Feature importance / coefficients chart
                st.subheader("Feature Analysis")
                if selected_model == "Random Forest":
                    imp = pd.DataFrame({
                        'Feature': available_features,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance')
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.barh(imp['Feature'], imp['Importance'], color='#2196F3')
                    ax.set_title('Feature Importances')
                    ax.grid(axis='x', alpha=0.3)
                    st.pyplot(fig); plt.close()
                else:
                    coef = pd.DataFrame({
                        'Feature': available_features,
                        'Coefficient': model.coef_
                    }).sort_values('Coefficient')
                    colors = ['#F44336' if c < 0 else '#4CAF50' for c in coef['Coefficient']]
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.barh(coef['Feature'], coef['Coefficient'], color=colors)
                    ax.axvline(0, color='black', linewidth=0.8)
                    ax.set_title('Feature Coefficients')
                    ax.grid(axis='x', alpha=0.3)
                    st.pyplot(fig); plt.close()

                # Download trained model
                joblib.dump(model, '/tmp/sacco_model.pkl')
                with open('/tmp/sacco_model.pkl', 'rb') as f:
                    st.download_button(
                        " Download Trained Model (.pkl)",
                        data=f,
                        file_name='sacco_model.pkl',
                        mime='application/octet-stream'
                    )


# ════════════════════════════════════════════════════════
# TAB 3 — PREDICT SURPLUS
# ════════════════════════════════════════════════════════
with tab3:
    # Use model trained in this session OR the one saved from the notebook
    model = st.session_state.get('trained_model', load_pretrained_model())
    features = st.session_state.get('trained_features', FEATURE_ORDER)

    if model is None:
        st.info("Train a model in the **Model Training** tab, or make sure `random_forest_model.pkl` is in the same folder as `app.py`.")
    else:
        model_source = "notebook-trained model" if 'trained_model' not in st.session_state else "just-trained model"
        st.subheader(f"Predict Surplus — using {model_source}")
        st.markdown("Enter SACCO financial figures below (KES):")

        defaults = {
            'Total Assets': 800_000_000,
            'Total Deposit liabilities': 500_000_000,
            'Total Equity': 150_000_000,
            'Total Liabilities': 650_000_000,
            'Cash & Cash Equivalent': 80_000_000,
            'Financial Investments': 120_000_000,
            'Gross Loan Portfolio': 400_000_000,
            'Net Loan Portfolio': 350_000_000,
            'Share Capital': 100_000_000,
            'Statutory Reserve': 50_000_000,
        }

        input_data = {}
        cols = st.columns(2)
        for i, feat in enumerate(features):
            with cols[i % 2]:
                input_data[feat] = st.number_input(
                    feat,
                    value=float(defaults.get(feat, 0)),
                    step=1_000_000.0,
                    format="%.0f"
                )

        if st.button(" Predict Surplus", type="primary"):
            prediction = predict_surplus(model, input_data, features)
            st.success("Prediction complete.")
            st.metric(
                label="Predicted Current Year's Surplus",
                value=f"KES {prediction:,.0f}"
            )
            assets = input_data.get('Total Assets', 1)
            roa = prediction / assets * 100 if assets > 0 else 0
            st.caption(f"Return on Assets (proxy): **{roa:.2f}%**")
