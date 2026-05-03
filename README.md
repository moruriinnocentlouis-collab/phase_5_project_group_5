SACCO FINANCIAL ANALYSIS AND SURPLUS PREDICTION
================================================
Phase 5 Group Project | Group 5
GitHub: https://github.com/moruriinnocentlouis-collab/phase_5_project_group_5.git


TABLE OF CONTENTS
-----------------
1.  Project Overview
2.  Repository Structure
3.  Data
4.  Setup and Installation
5.  Running the Analysis
6.  Exploratory Data Analysis
7.  Modelling
8.  Feature Importance
9.  The Streamlit Application
10. Reproducibility
11. Conclusions


------------------------------------------------------------------------
1. PROJECT OVERVIEW
------------------------------------------------------------------------

Savings and Credit Cooperative Organizations (SACCOs) are the primary
source of affordable credit for millions of Kenyans. Despite their
importance, most SACCO managers have no reliable method to forecast
whether their institution will end the financial year with a surplus or
a deficit. Without this forecast, decisions on dividends, loan portfolio
limits, and reserve allocation are made largely on judgment rather than
data.

This project investigates whether a SACCO's Current Year Surplus can be
predicted from ten standard balance sheet indicators. Nine machine
learning models were trained and evaluated on financial data from 52
deposit-taking SACCOs regulated by SASRA (SACCO Societies Regulatory
Authority of Kenya) over the period 2007 to 2023. The best-performing
model was deployed as an interactive web application using Streamlit.

Research Question
-----------------
Can ten standard balance sheet numbers predict a SACCO's Current Year
Surplus with sufficient accuracy to support financial planning decisions?

Objectives
----------
- Analyse the financial performance trends of 52 SACCOs over 16 years.
- Identify the key financial indicators that drive surplus.
- Build and compare nine regression models, from baseline linear
  regression to advanced ensemble and deep learning approaches.
- Select the best-performing model and deploy it as an interactive
  web application.

Why This Matters
----------------
Accurate surplus prediction enables SACCO management to:
- Plan dividend distribution and reserve allocation before year-end.
- Set safe loan portfolio growth targets.
- Meet SASRA capital adequacy requirements proactively.
- Identify institutions at risk of underperformance early.


------------------------------------------------------------------------
2. REPOSITORY STRUCTURE
------------------------------------------------------------------------

    sacco_project_sat.ipynb     Main analysis notebook covering data
                                loading, cleaning, EDA, modelling,
                                and deployment.

    app.py                      Streamlit web application, generated
                                by running the notebook.

    requirements.txt            Pinned Python dependencies for
                                reproducible installation.

    stacking_ensemble.pkl       Saved best-performing model, generated
                                by the notebook.

    data_clean/                 Folder containing one cleaned CSV file
                                per SACCO (52 files total).

    README.txt                  This document.


------------------------------------------------------------------------
3. DATA
------------------------------------------------------------------------

Source
------
The data was sourced from the SASRA annual supervision reports. SASRA
publishes audited financial statements for all regulated deposit-taking
SACCOs in Kenya on an annual basis.

To access the data, download the annual supervision reports from the
SASRA publications page at:
    https://www.sasra.go.ke/index.php/publications/supervision-reports

Download reports for the years 2007 to 2023. Extract the relevant
financial figures for each SACCO into individual CSV files following the
structure described below, and place them in the data_clean/ folder.

Dataset Summary
---------------
    Scope     52 deposit-taking SACCOs regulated by SASRA
    Period    2007 to 2023 (16 years)
    Format    One CSV file per SACCO in data_clean/
    Target    Current Year's Surplus
    Records   9,724 observations after reshaping to long format

CSV File Structure
------------------
Each file must contain:
- An Account_Name column listing financial line items
  (e.g. Total Assets, Gross Loan Portfolio).
- Year columns with numeric headers (e.g. 2019, 2020, 2021, 2022).
- One file per SACCO, named after the institution (e.g. stima.csv).

Data Preprocessing
------------------
The following steps were applied before modelling:

1. Zero values were treated as missing rather than legitimate
   observations and replaced with NaN.

2. Missing values were imputed using the column median to handle
   outliers robustly.

3. The dataset was reshaped from wide format to long format using the
   melt() function, then back to wide format for modelling using a
   pivot table.

4. A log transformation (log1p) was applied to all features and the
   target variable to reduce right-skew in financial data. This step
   alone improved the baseline R2 from 0.48 to 0.79.

5. An 80/20 train-test split was used throughout with random_state=42
   for reproducibility.


------------------------------------------------------------------------
4. SETUP AND INSTALLATION
------------------------------------------------------------------------

Prerequisites
-------------
- Python 3.11 (recommended)
- Git

Clone the Repository
--------------------
    git clone https://github.com/moruriinnocentlouis-collab/phase_5_project_group_5.git
    cd phase_5_project_group_5

Install Dependencies
--------------------
All dependencies are pinned to ensure reproducibility:

    pip install -r requirements.txt

If running on Google Colab or a fresh environment, the notebook also
installs XGBoost and LightGBM via subprocess during execution, so those
do not need to be installed separately.

Dependencies
------------
    streamlit       1.40.1
    pandas          2.2.3
    numpy           1.26.4
    matplotlib      3.9.2
    scikit-learn    1.5.2
    joblib          1.4.2
    xgboost         (installed during notebook run)
    lightgbm        (installed during notebook run)


------------------------------------------------------------------------
5. RUNNING THE ANALYSIS
------------------------------------------------------------------------

Open sacco_project_sat.ipynb in Jupyter and run all cells from top to
bottom without restarting the kernel between sections.

Part 1 -- Analysis
------------------
- Data loading and combining 52 SACCO CSV files.
- Missing value handling and data cleaning.
- Reshaping and pivoting to create the modelling dataset.
- Exploratory data analysis with visualisations.
- Baseline and advanced model training and evaluation.
- Feature importance analysis.
- Final model selection.

Part 2 -- Deployment
--------------------
- Environment and dependency verification.
- Writing app.py to disk using the %%writefile magic command.
- Saving the best-performing model as stacking_ensemble.pkl.
- Launching the Streamlit application.

Note: Do not restart the kernel between Part 1 and Part 2. The
deployment section depends on objects (rf, df_long, df_model) created
during the analysis.


------------------------------------------------------------------------
6. EXPLORATORY DATA ANALYSIS
------------------------------------------------------------------------

Key findings from the analysis of 52 SACCOs over 16 years:

Financial Trend
---------------
Sector performance was broadly flat from 2007 to 2016, followed by
gradual growth from 2017 to 2019. A significant surge occurred between
2019 and 2022, with total financial values nearly tripling across the
sector. A sharp decline is observed in 2023, which is likely
attributable to incomplete data collection rather than a reversal in
sector performance. Overall, the long-term trend shows strong growth.

Market Concentration
--------------------
The market is highly concentrated at the top. Stima SACCO holds
significantly more total value than any other institution in the
dataset. There is also a substantial valuation gap between the
bottom-ranked SACCOs, suggesting considerable variation in size and
scale across the sector.

Loan-Driven Business Model
---------------------------
Total Assets, Gross Loan Portfolio, and Total Liabilities move in
near-perfect synchronisation throughout the full period. The sector's
core activity is credit intermediation, with lending growth directly
fuelled by deposit growth. Total Equity and Cash Equivalents are
proportionally lower, reflecting a business model deeply focused on
leverage.

Funding Gap
-----------
From approximately 2018, the Gross Loan Portfolio began to consistently
exceed Total Deposit Liabilities. By 2023, loans reached approximately
KES 5 billion while deposits sat at approximately KES 4.5 billion,
implying a loan-to-deposit ratio above 100 percent. This is a signal of
stretched liquidity that warrants ongoing monitoring by management and
regulators.

Capital Structure as the Key Driver
-------------------------------------
Across all ensemble models, Total Liabilities and Total Equity rank as
the top two predictors of surplus. This finding challenges the intuition
that loan portfolio size is the primary driver of financial performance.
How a SACCO is financed matters more than how much it lends.


------------------------------------------------------------------------
7. MODELLING
------------------------------------------------------------------------

Approach
--------
Nine models were trained and evaluated using the same log-transformed
dataset and an 80/20 train-test split. R2 (coefficient of determination)
was the primary selection criterion.

Model Results
-------------
    Model                                Category       R2      Notes
    -----------------------------------------------------------------------
    Linear Regression (baseline)         Baseline       0.48    Starting point before preprocessing
    Linear Regression (log-transformed)  Baseline       0.79    Log scaling made the key difference
    Ridge Regression                     Regularised    0.79    L2 regularisation, no improvement
    Lasso Regression                     Regularised    0.78    L1 regularisation, feature selection
    Random Forest                        Ensemble       0.82    Strong non-linear performance
    XGBoost                              Ensemble       0.81    Gradient boosting, consistent results
    LightGBM                             Ensemble       0.77    Leaf-wise splitting, less consistent
    Neural Network (MLP)                 Deep Learning  0.76    128-64-32 architecture, ReLU activation
    Stacking Ensemble        [SELECTED]  Ensemble       0.82    Combines RF, Ridge, Lasso via meta-model

Note: R2 values are measured on the log-transformed test set. A value
of 0.82 means the model correctly explains 82 percent of the variation
in SACCO surplus across held-out test observations.

Why the Stacking Ensemble Was Selected
---------------------------------------
The Stacking Ensemble achieved the highest R2 of 0.82 and was selected
as the final model for three reasons:

1. It matched the accuracy of Random Forest while being more consistent,
   as it combines three different modelling approaches rather than
   relying on a single algorithm.

2. It uses the complementary strengths of its base learners. Random
   Forest captures non-linear patterns. Ridge Regression adds
   coefficient stability. Lasso Regression performs implicit feature
   selection. The Linear Regression meta-learner then learns the optimal
   weighting of their predictions using 5-fold cross-validation.

3. It is more robust to the characteristics of SACCO financial data,
   which include skewed distributions, varied institution sizes, and
   occasional missing values.

Architecture of the Stacking Ensemble
--------------------------------------
Level 0 base learners:
    - Random Forest Regressor (100 trees, random_state=42)
    - Ridge Regression (alpha=1.0)
    - Lasso Regression (alpha=0.01)

Level 1 meta-learner:
    - Linear Regression trained on out-of-fold predictions from the
      Level 0 models
    - 5-fold cross-validation used throughout to prevent data leakage


------------------------------------------------------------------------
8. FEATURE IMPORTANCE
------------------------------------------------------------------------

Feature importances were derived from the Random Forest model. The key
finding is that capital structure -- specifically Total Liabilities and
Total Equity -- predicts SACCO surplus more strongly than lending volume
alone.

    Feature                      Importance    Interpretation
    -----------------------------------------------------------------------
    Total Liabilities                  22%    Strongest predictor of surplus
    Total Equity                       19%    Capital adequacy and financial strength
    Financial Investments              14%    Diversified income beyond lending
    Gross Loan Portfolio               13%    Primary lending activity
    Total Assets                       11%    Overall scale of the institution
    Net Loan Portfolio                  8%    Loans net of provisions
    Total Deposit Liabilities           6%    Member savings funding loan activity
    Cash and Cash Equivalent            4%    Short-term liquidity position
    Share Capital                       2%    Member equity contributions
    Statutory Reserve                   1%    Regulatory reserve requirement


------------------------------------------------------------------------
9. THE STREAMLIT APPLICATION
------------------------------------------------------------------------

Running Locally
---------------
After running the notebook, launch the application with:

    streamlit run app.py

Then open a browser at http://localhost:8501.
Public link: https://phase5projectgroup5-awkluggptkz7vty9tsahtk.streamlit.app/

Hosting on Streamlit Community Cloud
--------------------------------------
1. Push the repository to GitHub, ensuring app.py and requirements.txt
   are in the root directory.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click New app and enter:
      Repository: moruriinnocentlouis-collab/phase_5_project_group_5
      Branch:     main
      Main file:  app.py
4. Click Deploy. Streamlit reads requirements.txt, installs packages,
   and serves the application at a public URL.

Application Features
--------------------
Model Training
    Retrain any of the nine models directly in the application. Set the
    test/train split size, view feature importance charts, and download
    the trained model as a .pkl file.

Predict Surplus
    Enter values for the ten balance sheet indicators. The application
    applies the same log transformation used during training, passes the
    values to the saved model, and returns the predicted Current Year
    Surplus along with a proxy return-on-assets figure.


------------------------------------------------------------------------
10. REPRODUCIBILITY
------------------------------------------------------------------------

This project addresses the three standard components of reproducible
data science:

1. Published Code
   All code is version-controlled on GitHub at the repository linked at
   the top of this document. Every step from data loading to deployment
   is documented in the notebook with accompanying explanations.

2. Pinned Environment
   requirements.txt specifies exact package versions. Running
   pip install -r requirements.txt reproduces the environment used
   during development. XGBoost and LightGBM are installed via subprocess
   calls within the notebook itself, ensuring availability even if not
   pre-installed in the base environment.

3. Data Access
   The 52 cleaned SACCO CSV files are stored in the data_clean/ folder
   of the repository. The original data was sourced from SASRA annual
   supervision reports. Instructions for downloading and structuring the
   source data are in Section 3 of this document.


------------------------------------------------------------------------
11. CONCLUSIONS
------------------------------------------------------------------------

The analysis demonstrates that a SACCO's Current Year Surplus can be
predicted from ten standard balance sheet figures with 82 percent
accuracy. The following conclusions were drawn:

1. The Stacking Ensemble is the best-performing model, achieving
   R2 = 0.82 by combining Random Forest, Ridge Regression, and Lasso
   Regression through a Linear Regression meta-model. Model diversity
   outperforms any single algorithm on this dataset.

2. Capital structure is a stronger predictor of surplus than lending
   volume. Total Liabilities and Total Equity consistently rank as the
   top two features across all ensemble models, regardless of which
   algorithm is used.

3. The sector experienced significant growth between 2019 and 2022,
   with a loan-to-deposit ratio exceeding 100 percent from 2018
   onwards. This represents a potential liquidity risk that SACCO
   managers and SASRA should monitor.

4. The Streamlit web application makes the prediction tool accessible
   to any SACCO manager or analyst without requiring programming
   knowledge.


------------------------------------------------------------------------
This project was developed for academic purposes as part of a Phase 5
data science programme.
------------------------------------------------------------------------
