import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, log_loss, confusion_matrix,
    classification_report, roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score
)

# ======================================
# PAGE CONFIG
# ======================================

st.set_page_config(
    page_title="ITD105 Lab Exercise 2",
    layout="wide"
)


# === Simple global styling ===
ACCENT_COLOR = "#2563eb"  # blue accent

st.markdown(f"""
    <style>
    /* Sidebar background and text */
    [data-testid="stSidebar"] {{
        background-color: #0f172a;   /* dark navy */
        color: white;
    }}
    [data-testid="stSidebar"] * {{
        color: white !important;
    }}

    /* Main page background */
    .stApp {{
        background-color: #f5f5f5;
    }}

    /* Buttons with uniform accent color */
    .stButton>button {{
        background-color: {ACCENT_COLOR};
        color: white;
        border-radius: 999px;
        border: none;
        padding: 0.4rem 1.2rem;
        font-weight: 600;
    }}
    .stButton>button:hover {{
        opacity: 0.9;
    }}

    /* Radio options in sidebar */
    div[role="radiogroup"] > label {{
        padding: 0.5rem 0.75rem;
        border-radius: 999px;
        margin-bottom: 0.25rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    div[role="radiogroup"] > label:hover {{
        background-color: rgba(255, 255, 255, 0.1);
    }}
    </style>
""", unsafe_allow_html=True)
# ======================================



# === Sidebar navigation ===
with st.sidebar:
    st.markdown("### Model Evaluation")

    # default page stored in session_state
    if "nav" not in st.session_state:
        st.session_state["nav"] = "🏠 Home"

    page = st.radio(
        "Navigation",
        ["🏠 Home", "📊 Classification", "📈 Regression"],
        index=["🏠 Home", "📊 Classification", "📈 Regression"].index(st.session_state["nav"]),
        label_visibility="collapsed"
    )

    # keep session_state in sync when user clicks in sidebar
    st.session_state["nav"] = page
# ======================================



# ======================================
# PART 1 – FRAMINGHAM: LOADING + PREPROCESSING
# ======================================

@st.cache_data
def load_and_preprocess_framingham():
    """
    Load the Framingham dataset and apply the exact preprocessing
    described in the notebook:
      1. Drop ID + other outcome/label + time-to-event columns
      2. Drop HDLC and LDLC (too many missing values)
      3. Drop rows with missing values in key columns (incl. educ & ANYCHD)
      4. Drop PREVAP and CURSMOKE (highly correlated with other features)
      5. Define X (features) and y (ANYCHD)
    """
    df_raw = pd.read_csv("Framingham Dataset.csv")
    info = {}
    info["raw_shape"] = df_raw.shape

    # Work on a copy
    df = df_raw.copy()

    # === 1. Drop ID + other outcome/label + time-to-event columns ===
    cols_to_drop = [
        # ID
        "RANDID", "educ",

        # Other outcome/label columns (future events)
        "DEATH", "ANGINA", "HOSPMI", "MI_FCHD",
        "STROKE", "CVD", "HYPERTEN",
        # NOTE: ANYCHD is NOT dropped here; it's our target
        # and is handled separately below.

        # Time-to-event columns (future information)
        "TIMEAP", "TIMEMI", "TIMEMIFC", "TIMECHD",
        "TIMESTRK", "TIMECVD", "TIMEDTH", "TIMEHYP",
        "TIME", "PERIOD"   # sometimes present as general time
    ]
    cols_to_drop_existing = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop_existing)
    info["dropped_leakage_cols"] = cols_to_drop_existing
    info["shape_after_leakage_drop"] = df.shape

    # === 2. Drop HDLC and LDLC (too many missing values) ===
    lipids_to_drop = ["HDLC", "LDLC"]
    lipids_existing = [c for c in lipids_to_drop if c in df.columns]
    df = df.drop(columns=lipids_existing)
    info["dropped_lipids_cols"] = lipids_existing
    info["shape_after_lipids_drop"] = df.shape

    # === 3. Drop rows with missing values in key columns ===
    cols_required_complete = [
        "SEX",
        "TOTCHOL",
        "AGE",
        "SYSBP",
        "DIABP",
        "CURSMOKE",
        "CIGPDAY",
        "BMI",
        "DIABETES",
        "BPMEDS",
        "HEARTRTE",
        "GLUCOSE",
        "educ",
        "PREVCHD",
        "PREVAP",
        "PREVMI",
        "PREVSTRK",
        "PREVHYP",
        "PERIOD",
        "ANYCHD",  # include target as well
    ]
    cols_required_complete = [c for c in cols_required_complete if c in df.columns]

    before_rows = df.shape[0]
    df = df.dropna(subset=cols_required_complete)
    after_rows = df.shape[0]

    info["rows_before_dropna"] = before_rows
    info["rows_after_dropna"] = after_rows
    info["rows_removed_dropna"] = before_rows - after_rows
    info["rows_removed_pct"] = round(
        (before_rows - after_rows) / before_rows * 100, 2
    )
    info["shape_after_dropna"] = df.shape
    info["cols_required_complete"] = cols_required_complete

    # === 4. Drop PREVAP and CURSMOKE (highly correlated) ===
    corr_drop_cols = ["PREVAP", "CURSMOKE"]
    corr_drop_existing = [c for c in corr_drop_cols if c in df.columns]
    df = df.drop(columns=corr_drop_existing)
    info["dropped_corr_cols"] = corr_drop_existing
    info["shape_after_corr_drop"] = df.shape

    # === 5. Define features X and target y ===
    target_col = "ANYCHD"
    if target_col not in df.columns:
        raise KeyError(
            f"Expected target column '{target_col}' not found. Columns: {list(df.columns)}"
        )

    X = df.drop(columns=[target_col])
    y = df[target_col]

    info["target_col"] = target_col
    info["feature_columns"] = X.columns.tolist()
    info["X_shape"] = X.shape
    info["y_shape"] = y.shape

    return df, X, y, info

# ======================================
# PART 1 – MODEL TRAINING (REPLICATING NOTEBOOK)
# ======================================

@st.cache_resource
def train_part1_models():
    """Train the classification models for Part 1 using the same
    logic as in the notebook:
      - Model A: single train/test split
      - Model B: 10-fold stratified cross-validation
    """
    df, X, y, info = load_and_preprocess_framingham()

    # ---------- Model A: Train/Test split ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train_scaled, y_train)

    y_proba_test = logreg.predict_proba(X_test_scaled)[:, 1]
    y_pred_test = (y_proba_test >= 0.5).astype(int)

    acc_tt = accuracy_score(y_test, y_pred_test)
    ll_tt = log_loss(y_test, y_proba_test)
    auc_tt = roc_auc_score(y_test, y_proba_test)
    cm_tt = confusion_matrix(y_test, y_pred_test)
    report_tt = classification_report(y_test, y_pred_test, output_dict=True)
    fpr_tt, tpr_tt, _ = roc_curve(y_test, y_proba_test)

    metrics_train_test = {
        "accuracy": acc_tt,
        "log_loss": ll_tt,
        "auc": auc_tt,
        "confusion_matrix": cm_tt,
        "classification_report": report_tt,
        "fpr": fpr_tt,
        "tpr": tpr_tt,
        "y_test": y_test,
        "y_proba_test": y_proba_test,
    }

    # pipeline version of the same model for easy use later (interactive prediction)
    pipeline_tt = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("log_reg", LogisticRegression(max_iter=1000)),
    ])
    pipeline_tt.fit(X_train, y_train)

    # ---------- Model B: 10-fold stratified cross-validation ----------
    cv_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("log_reg", LogisticRegression(max_iter=1000)),
    ])

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    cv_accuracy = cross_val_score(cv_pipeline, X, y, cv=kf, scoring="accuracy")
    cv_logloss = cross_val_score(cv_pipeline, X, y, cv=kf, scoring="neg_log_loss")
    cv_auc = cross_val_score(cv_pipeline, X, y, cv=kf, scoring="roc_auc")

    # Out-of-fold predictions (for confusion matrix and ROC under CV)
    y_proba_cv = cross_val_predict(cv_pipeline, X, y, cv=kf, method="predict_proba")[:, 1]
    y_pred_cv = (y_proba_cv >= 0.5).astype(int)

    acc_cv_full = accuracy_score(y, y_pred_cv)
    logloss_cv_full = log_loss(y, y_proba_cv)
    auc_cv_full = roc_auc_score(y, y_proba_cv)
    cm_cv = confusion_matrix(y, y_pred_cv)
    report_cv = classification_report(y, y_pred_cv, output_dict=True)
    fpr_cv, tpr_cv, _ = roc_curve(y, y_proba_cv)

    metrics_cv = {
        # fold-averaged metrics
        "accuracy_mean": cv_accuracy.mean(),
        "accuracy_std": cv_accuracy.std(),
        "logloss_mean": -cv_logloss.mean(),
        "logloss_std": cv_logloss.std(),
        "auc_mean": cv_auc.mean(),
        "auc_std": cv_auc.std(),
        # full OOF metrics
        "accuracy_full": acc_cv_full,
        "logloss_full": logloss_cv_full,
        "auc_full": auc_cv_full,
        "confusion_matrix": cm_cv,
        "classification_report": report_cv,
        "fpr": fpr_cv,
        "tpr": tpr_cv,
    }

    return metrics_train_test, metrics_cv, pipeline_tt, info

# ======================================
# PART 2 – ENERGY DATA: LOADING + PREPROCESSING
# ======================================

@st.cache_data
def load_and_preprocess_energy():
    """Load the Energy Efficiency dataset and apply the same preprocessing
    used in the notebook: simply renaming columns; no rows are dropped."""
    df_raw = pd.read_excel("ENBenergy_efficiency_2012_data.xlsx", sheet_name=0)
    info = {}
    info["raw_shape"] = df_raw.shape

    df = df_raw.rename(columns={
        "X1": "Relative_Compactness",
        "X2": "Surface_Area",
        "X3": "Wall_Area",
        "X4": "Roof_Area",
        "X5": "Overall_Height",
        "X6": "Orientation",
        "X7": "Glazing_Area",
        "X8": "Glazing_Area_Distribution",
        "Y1": "Heating_Load",
        "Y2": "Cooling_Load",
    })

    info["after_rename_shape"] = df.shape

    # In the lab notebook, they verified there are no missing values
    info["missing_values"] = df.isnull().sum().to_dict()

    # Features and target exactly as in the notebook
    feature_cols = [
        "Relative_Compactness",
        "Surface_Area",
        "Wall_Area",
        "Roof_Area",
        "Overall_Height",
        "Orientation",
        "Glazing_Area",
        "Glazing_Area_Distribution",
    ]
    target_col = "Heating_Load"

    X = df[feature_cols]
    y = df[target_col]

    info["feature_columns"] = feature_cols
    info["X_shape"] = X.shape
    info["y_shape"] = y.shape

    return df, X, y, info

# ======================================
# PART 2 – MODEL TRAINING (REPLICATING NOTEBOOK)
# ======================================

@st.cache_resource
def train_part2_models():
    """Train the regression models for Part 2 using the same logic
    as in the notebook:
      - Model A: single train/test split
      - Model B: repeated random splits
    """
    df, X, y, info = load_and_preprocess_energy()

    # ----- Common pipeline -----
    linreg_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])

    # ---------- Model A: Single train/test split ----------
    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    linreg_pipeline.fit(X_train_a, y_train_a)
    y_pred_a = linreg_pipeline.predict(X_test_a)

    mse_a = mean_squared_error(y_test_a, y_pred_a)
    mae_a = mean_absolute_error(y_test_a, y_pred_a)
    r2_a = r2_score(y_test_a, y_pred_a)

    metrics_model_a = {
        "mse": mse_a,
        "mae": mae_a,
        "r2": r2_a,
        "y_test": y_test_a,
        "y_pred": y_pred_a,
    }

    # ---------- Model B: Repeated random train/test splits ----------
    n_repeats = 10
    mse_list = []
    mae_list = []
    r2_list = []
    last_y_test_b = None
    last_y_pred_b = None
    last_pipeline_b = None

    for seed in range(n_repeats):
        X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )

        pipe_b = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ])

        pipe_b.fit(X_train_b, y_train_b)
        y_pred_b = pipe_b.predict(X_test_b)

        mse_list.append(mean_squared_error(y_test_b, y_pred_b))
        mae_list.append(mean_absolute_error(y_test_b, y_pred_b))
        r2_list.append(r2_score(y_test_b, y_pred_b))

        # remember last run for plotting
        last_y_test_b = y_test_b
        last_y_pred_b = y_pred_b
        last_pipeline_b = pipe_b

    metrics_model_b = {
        "mse_mean": float(np.mean(mse_list)),
        "mse_std": float(np.std(mse_list)),
        "mae_mean": float(np.mean(mae_list)),
        "mae_std": float(np.std(mae_list)),
        "r2_mean": float(np.mean(r2_list)),
        "r2_std": float(np.std(r2_list)),
        "y_test_last": last_y_test_b,
        "y_pred_last": last_y_pred_b,
    }

    # We'll use the last pipeline from Model B for the interactive demo
    return metrics_model_a, metrics_model_b, last_pipeline_b, info

# ======================================
# HOME PAGE
# ======================================

if page == "🏠 Home":
    # === HERO SECTION ===
    left_col, right_col = st.columns([4, 3])

    with left_col:
        st.markdown("#### Welcome to")
        st.markdown("## **Model Evaluation Explorer**")
        st.markdown("""
        This app lets you interact with two machine learning workflows:

        - **Classification:** Predicting 10-year heart disease risk (Framingham dataset)  
        - **Regression:** Predicting building heating load (Energy Efficiency dataset)  

        You can see how the data was **cleaned**, how the models were **trained**,
        and how different **evaluation strategies** (train/test split, cross-validation,
        repeated random splits) affect the performance metrics.
        """)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Explore Classification"):
                # hack: change page by rerunning with a session state key
                st.session_state["nav"] = "📊 Classification"
        with c2:
            if st.button("Explore Regression"):
                st.session_state["nav"] = "📈 Regression"

    with right_col:
        try:
            st.markdown("### ")
            st.image("hero.png", use_container_width=True)
        except Exception:
            st.info("Add an image file named `hero.png` in the app folder to show a hero illustration here.")

    st.markdown("---")

    # === STEP-BY-STEP SECTION ===
    st.markdown("### How this app is structured")

    step1, step2, step3, step4 = st.columns(4)

    with step1:
        st.markdown("#### 1️⃣ Data")
        st.markdown("""
        - Load raw CSV / Excel  
        - Inspect shape & columns  
        - Check for missing values
        """)

    with step2:
        st.markdown("#### 2️⃣ Preprocessing")
        st.markdown("""
        - Drop ID & leakage columns  
        - Handle missing values  
        - Select features & target
        """)

    with step3:
        st.markdown("#### 3️⃣ Modeling")
        st.markdown("""
        - Train/Test split  
        - 10-fold cross-validation  
        - Repeated random splits
        """)

    with step4:
        st.markdown("#### 4️⃣ Evaluation & Demo")
        st.markdown("""
        - Compare metrics (Accuracy, AUC, MSE, R²)  
        - Interpret results (Q3 & Q4)  
        - Try interactive predictions
        """)

    st.markdown("---")
    st.markdown("""
    👉 Use the **sidebar** to switch between the **Classification** and **Regression** sections
    and explore each workflow in more detail.
    """)


# ======================================
# PART 1 PAGE – CLASSIFICATION
# ======================================

elif page == "📊 Classification":
    st.title("Classification: Framingham Heart Study")

    df_fram, X_fram, y_fram, info_fram = load_and_preprocess_framingham()
    metrics_tt, metrics_cv, clf_part1, info_model = train_part1_models()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview",
        "Dataset & Preprocessing",
        "Models & Metrics",
        "Try the Classifier",
    ])

    # ---------- Tab 1: Model Overview ----------
    with tab1:
        st.subheader("Model Overview")

        # 1. Metrics explanation – card style
        st.markdown(f"""
        <div style="
            background-color: #ffffff;
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
            box-shadow: 0 2px 10px rgba(15,23,42,0.08);
            border-left: 4px solid {ACCENT_COLOR};
            margin-bottom: 1rem;
        ">
        <h4 style="margin-top:0;">Performance metrics used in this app</h4>
        <ul>
            <li><b>Accuracy</b> – proportion of patients that are correctly classified
                as having CHD or not. Accuracy around <b>0.79–0.80</b> means roughly
                8 out of 10 patients are classified correctly.</li>
            <li><b>Log Loss</b> – measures how well predicted <em>probabilities</em>
                match the true labels, punishing very confident mistakes.
                Lower log loss means better calibrated probabilities.</li>
            <li><b>ROC AUC</b> – probability that a randomly chosen CHD patient
                receives a higher risk score than a non-CHD patient.
                Values around <b>0.77</b> indicate good discriminative power.</li>
            <li><b>Confusion Matrix</b> – breaks predictions into:<br>
                • True Negatives (correct non-CHD)<br>
                • False Positives (non-CHD predicted as CHD)<br>
                • False Negatives (CHD predicted as non-CHD)<br>
                • True Positives (correct CHD)<br>
                It shows the trade-off between catching true CHD cases
                (recall/sensitivity) and avoiding false alarms (specificity).</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        # 2. Model options – side-by-side cards
        col_a, col_b = st.columns(2)

        with col_b:
            st.markdown(f"""
            <div style="
                background-color: #ffffff;
                border-radius: 16px;
                padding: 1.25rem 1.5rem;
                box-shadow: 0 2px 10px rgba(15,23,42,0.08);
                margin-bottom: 1rem;
            ">
            <h4 style="margin-top:0;">Model B – Leave-One-Out Cross-Validation</h4>
            <ul>
                <li>The dataset is treated as having <b>N folds</b>, where N is the number of patients.</li>
                <li>In each round, the model is trained on <b>N − 1 patients</b> and validated on the single left-out patient.</li>
                <li>Every patient is used once as a validation case and almost every time as part of the training set.</li>
                <li>Metrics are averaged across all N runs, giving mean Accuracy, Log Loss, and ROC AUC,
                    but each run is based on just one validation patient.</li>
                <li>This behaves like an extreme form of K-fold CV, using nearly all data for training but at a higher computational cost.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)


        with col_a:
            st.markdown(f"""
            <div style="
                background-color: #ffffff;
                border-radius: 16px;
                padding: 1.25rem 1.5rem;
                box-shadow: 0 2px 10px rgba(15,23,42,0.08);
                margin-bottom: 1rem;
            ">
            <h4 style="margin-top:0;">Model A – K-Fold Cross-Validation</h4>
            <ul>
                <li>The dataset is split into <b>10 folds</b>.</li>
                <li>In each round, 9 folds are used for training and 1 fold for validation.</li>
                <li>Every patient is used once as validation and multiple times for training.</li>
                <li>Metrics are averaged across the 10 folds, giving mean Accuracy, Log Loss, and ROC AUC,
                    plus a sense of variability across folds.</li>
                <li>This mimics training and testing on many different splits, without wasting data.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        # 3. Why focus on the cross-validated model
        st.markdown(f"""
        <div style="
            background-color: #0f172a;
            color: #ffffff;
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
            margin-top: 0.5rem;
        ">
        <h4 style="margin-top:0;">This app focuses on the K-Fold model, Why?</h4>
        <ul>
            <li><b>More stable than Leave-One-Out.</b><br>
                Leave-One-Out Cross-Validation (LOOCV) validates on a single patient at a time,
                which can make results very sensitive to outliers. In 10-fold CV, each validation
                fold contains many patients, giving a smoother and more stable estimate of performance.</li>
            <li><b>Still uses every patient for validation.</b><br>
                Just like LOOCV, every patient appears in a validation set exactly once, but 10-fold CV
                does this with only 10 model fits instead of N fits. This keeps the evaluation thorough
                without being unnecessarily expensive.</li>
            <li><b>More efficient for this dataset.</b><br>
                With LOOCV, the model would need to be refit as many times as there are patients.
                For medical models that take time to train and tune, 10-fold CV is a much more
                practical choice while giving very similar insights.</li>
            <li><b>Aligned with common practice.</b><br>
                Using 5- or 10-fold Cross-Validation is standard in applied machine learning.
                It balances bias, variance, and computation, and is widely accepted when reporting
                model performance for potential real-world deployment.</li>
        </ul>
        <p style="margin-bottom:0;">
            Because of these reasons, the <b>10-fold cross-validated logistic regression</b>
            is treated as the main model in this app. The next tab shows its metrics in detail.
        </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")



    # ---------- Tab 2: Dataset & Preprocessing ----------
    with tab2:
        st.subheader("Framingham Heart Study – Dataset & Preprocessing")

        # === 1. High-level description ===
        top_left, top_right = st.columns([2, 1])

        with top_left:
            st.markdown("""
            The **Framingham Heart Study** is a long-term cardiovascular study that has
            followed thousands of participants from the town of Framingham, Massachusetts.

            In this app, we focus on predicting **`ANYCHD`** – whether a person will
            develop **any coronary heart disease (CHD)** during follow-up
            (myocardial infarction, angina, coronary insufficiency, or fatal CHD).

            This is naturally a **binary classification** problem:

            - `ANYCHD = 1` → CHD event occurred  
            - `ANYCHD = 0` → no CHD event

            The dataset is well-suited for this project because it contains:
            - clinically meaningful baseline risk factors (age, blood pressure, cholesterol, smoking, diabetes, etc.), and  
            - a large sample size (over **9,000** complete records after cleaning),
            which is ideal for training and evaluating a **logistic regression** model.
            """)

        with top_right:
            st.markdown(f"""
            <div style="
                background-color: #ffffff;
                border-radius: 16px;
                padding: 1rem 1.25rem;
                box-shadow: 0 2px 10px rgba(15,23,42,0.08);
                border-left: 4px solid {ACCENT_COLOR};
            ">
            <h4 style="margin-top:0; margin-bottom:0.5rem;">Basic Info</h4>
            <p style="margin:0;">
                <b>Rows:</b> {info_fram["y_shape"][0]}<br>
                <b>Features:</b> {info_fram["X_shape"][1]}<br>
                <b>Target:</b> {info_fram["target_col"]}
            </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # === 2. Preprocessing pipeline (card-style steps) ===
        st.markdown("### Preprocessing pipeline")

        step1, step2, step3 = st.columns(3)

        with step1:
            st.markdown("#### 1️⃣ Remove ID, outcome, and time-to-event columns")
            st.markdown("""
            We drop columns that either:
            - have **no medical meaning** as features (ID), or  
            - contain **future information** (outcomes and event times).

            **Dropped columns include for example:**
            """)
            st.code(", ".join(info_fram["dropped_leakage_cols"]), language="text")
            st.markdown(
                f"Shape after this step: **{info_fram['shape_after_leakage_drop'][0]} × {info_fram['shape_after_leakage_drop'][1]}**"
            )

        with step2:
            st.markdown("#### 2️⃣ Handle missing values (HDLC/LDLC + incomplete rows)")
            st.markdown("""
            1. **Drop `HDLC` and `LDLC`**  
            These lipid variables have ~74% missing values, which makes
            imputation unreliable, so we remove them entirely.

            2. **Drop rows with missing values** in key baseline variables  
            (sex, cholesterol, blood pressures, smoking, BMI, diabetes,
            meds, glucose, education, previous CHD/stroke, PERIOD, ANYCHD).
            """)
            st.markdown(
                f"- Rows before: **{info_fram['rows_before_dropna']}**  \n"
                f"- Rows after: **{info_fram['rows_after_dropna']}**  \n"
                f"- Removed: **{info_fram['rows_removed_dropna']}** "
                f"({info_fram['rows_removed_pct']}%)"
            )
            st.markdown(
                f"Shape after this step: **{info_fram['shape_after_dropna'][0]} × {info_fram['shape_after_dropna'][1]}**"
            )

        with step3:
            st.markdown("#### 3️⃣ Reduce redundancy (correlated features)")
            st.markdown("""
            Correlation analysis showed:

            - `PREVCHD` and `PREVAP` are highly correlated (angina is part of CHD history).  
            - `CURSMOKE` and `CIGPDAY` are highly correlated
            (non-smokers have `CIGPDAY = 0`, smokers have `CIGPDAY > 0`).

            To reduce redundancy and multicollinearity, we keep the more informative
            variables and **drop**:
            """)
            st.code(", ".join(info_fram["dropped_corr_cols"]), language="text")
            st.markdown(
                f"Final cleaned shape: **{info_fram['shape_after_corr_drop'][0]} × {info_fram['shape_after_corr_drop'][1]}**"
            )

        st.markdown("---")

        st.markdown("### Modeling pipeline used in training")

        st.markdown(f"""
        <div style="
            background-color: #ffffff;
            border-radius: 16px;
            padding: 1rem 1.25rem;
            box-shadow: 0 2px 10px rgba(15,23,42,0.08);
            border-left: 4px solid {ACCENT_COLOR};
            margin-bottom: 1rem;
        ">
        <h4 style="margin-top:0;">StandardScaler + LogisticRegression</h4>
        <p style="margin-bottom:0.5rem;">
            The model is built inside a <code>Pipeline</code> that performs:
        </p>
        <ul style="margin-top:0.25rem;">
            <li><b>StandardScaler</b> – centers each feature (mean 0) and scales it
                to unit variance. This ensures that features with larger numeric
                ranges do not dominate the regression just
                because of their scale.</li>
            <li><b>LogisticRegression</b> – fits a standard logistic regression model to
                predict <code>ANYCHD</code>.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        # === 3. Raw vs Cleaned dataset preview ===
        st.markdown("### Raw vs. Cleaned dataset")

        raw_col, clean_col = st.columns(2)

        with raw_col:
            st.markdown("#### Raw dataset (before preprocessing)")
            raw_df_preview = pd.read_csv("Framingham Dataset.csv").head()
            st.dataframe(raw_df_preview, use_container_width=True)
            st.caption(f"Raw shape: {info_fram['raw_shape'][0]} rows × {info_fram['raw_shape'][1]} columns")

        with clean_col:
            st.markdown("#### Cleaned dataset (after all steps)")
            st.dataframe(df_fram.head(), use_container_width=True)
            st.caption(
                f"Cleaned shape: {info_fram['X_shape'][0]} rows × {info_fram['X_shape'][1] + 1} columns "
                f"(including target `{info_fram['target_col']}`)"
            )

        st.markdown("### Final feature set used for logistic regression")
        st.write(info_fram["feature_columns"])
        st.caption("These columns form the feature matrix **X**. The target vector **y** is `ANYCHD`.")


    # ---------- Tab 3: Models & Metrics ----------
    with tab3:

        # === 1. Quick data summary ===
        top_left, top_right = st.columns([2, 1])

        with top_left:
            st.markdown("#### Dataset snapshot")
            st.dataframe(df_fram.head(), use_container_width=True)

        with top_right:
            st.markdown("#### ")
            st.markdown(f"""
            <div style="
                background-color: #ffffff;
                border-radius: 16px;
                padding: 1rem 1.25rem;
                box-shadow: 0 2px 10px rgba(15,23,42,0.08);
                border-left: 4px solid {ACCENT_COLOR};
            ">
            <h4 style="margin-top:0; margin-bottom:0.5rem;">Framingham CHD dataset</h4>
            <p style="margin:0;">
                <b>Rows (after cleaning):</b> {info_fram["X_shape"][0]}<br>
                <b>Features:</b> {info_fram["X_shape"][1]}<br>
                <b>Target:</b> {info_fram["target_col"]}
            </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # === 2. Features and target definition ===
        st.markdown("### Features and target used in this model")

        feat_col, target_col = st.columns([3, 1])

        with feat_col:
            
            st.markdown(f"""
            <div style="
                background-color: #ffffff;
                border-radius: 16px;
                padding: 1rem 1.25rem;
                box-shadow: 0 2px 10px rgba(15,23,42,0.08);
                margin-bottom: 0.75rem;
            ">
            <h4 style="margin-top:0;">Feature matrix <code>X</code></h4>
            <p style="margin-bottom:0.25rem;">
                The model uses the following baseline risk factors as inputs:
            </p>
            <ul style="margin-top:0.25rem;">
            """ + "".join([f"<li>{c}</li>" for c in info_fram["feature_columns"]]) + """
            </ul>
            </div>
            """, unsafe_allow_html=True)

        with target_col:
            st.markdown(f"""
            <div style="
                background-color: #0f172a;
                color: #ffffff;
                border-radius: 16px;
                padding: 1rem 1.25rem;
                box-shadow: 0 2px 10px rgba(15,23,42,0.08);
            ">
            <h4 style="margin-top:0;">Target <code>y</code></h4>
            <p style="margin-bottom:0.25rem;">
                <b>{info_fram["target_col"]}</b>
            </p>
            <p style="font-size:0.9rem; margin-bottom:0;">
                0 = no coronary heart disease during follow-up<br>
                1 = at least one CHD event (any CHD)
            </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # === 3. Cross-validated model summary (Model B) ===
        st.markdown("### 10-Fold Cross-Validated Logistic Regression")

        # high-level metrics card
        st.markdown(f"""
        <div style="
            background-color: #ffffff;
            border-radius: 16px;
            padding: 1rem 1.25rem;
            box-shadow: 0 2px 10px rgba(15,23,42,0.08);
            border-left: 4px solid {ACCENT_COLOR};
            margin-bottom: 1rem;
        ">
        <h4 style="margin-top:0;">Overall performance (averaged over 10 folds)</h4>
        <ul>
            <li><b>Classification Accuracy (mean ± std):</b>
                {metrics_cv['accuracy_mean']:.3f} ± {metrics_cv['accuracy_std']:.3f}</li>
            <li><b>Logarithmic Loss (mean ± std):</b>
                {metrics_cv['logloss_mean']:.3f} ± {metrics_cv['logloss_std']:.3f}</li>
            <li><b>Area Under ROC Curve (AUC, mean ± std):</b>
                {metrics_cv['auc_mean']:.3f} ± {metrics_cv['auc_std']:.3f}</li>
        </ul>
        <p style="font-size:0.9rem; margin-bottom:0;">
            These values summarize how the logistic regression model performs across
            multiple train/validation splits, giving a more stable estimate than a
            single train/test partition.
        </p>
        </div>
        """, unsafe_allow_html=True)

        # === 4. Detailed confusion matrix + ROC (based on out-of-fold predictions) ===
        cm_cv = metrics_cv["confusion_matrix"]
        report_cv = metrics_cv["classification_report"]
        fpr_cv = metrics_cv["fpr"]
        tpr_cv = metrics_cv["tpr"]
        auc_full = metrics_cv["auc_full"]

        mat_col, roc_col = st.columns(2)

        with mat_col:
            st.markdown("#### Confusion matrix (out-of-fold predictions)")
            cm_df = pd.DataFrame(
                cm_cv,
                index=["Actual 0 (no CHD)", "Actual 1 (CHD)"],
                columns=["Pred 0 (no CHD)", "Pred 1 (CHD)"],
            )
            st.dataframe(cm_df, use_container_width=True)

            st.markdown("""
            <p style="font-size:0.9rem;">
            This matrix is computed using out-of-fold predictions from the 10-fold
            cross-validation procedure. It summarizes how often the model correctly
            distinguishes CHD vs non-CHD, and where it makes false positive or
            false negative errors across the whole dataset.
            </p>
            """, unsafe_allow_html=True)

        with roc_col:
            st.markdown("#### ROC curve (out-of-fold predictions)")
            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr_cv, tpr_cv, label=f"AUC = {auc_full:.2f}")
            ax_roc.plot([0, 1], [0, 1], linestyle="--")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title("ROC Curve – 10-Fold Cross-Validation (OOF)")
            ax_roc.legend()
            st.pyplot(fig_roc)

        st.markdown("---")

        # === 5. Classification report table ===
        st.markdown("### Classification report (precision, recall, F1-score)")

        report_df = pd.DataFrame(report_cv).T
        st.dataframe(report_df, use_container_width=True)

        st.caption(
            "The classification report shows precision, recall, F1-score and support "
            "for each class (0 = no CHD, 1 = CHD), as well as macro and weighted averages."
        )


    # ---------- Tab 4: Interactive prediction ----------
    with tab4:
        st.subheader("Try the CHD Risk Classifier")


        # === 1. Brief reminder of model performance ===
        st.markdown(f"""
        <div style="
            background-color: #ffffff;
            border-radius: 16px;
            padding: 1rem 1.25rem;
            box-shadow: 0 2px 10px rgba(15,23,42,0.08);
            border-left: 4px solid {ACCENT_COLOR};
            margin-bottom: 1rem;
        ">
        <h4 style="margin-top:0;">How reliable are these predictions?</h4>
        <p style="margin-bottom:0.5rem;">
            The classifier shown here is the <b>10-fold cross-validated logistic regression</b>
            trained on the cleaned Framingham dataset.
        </p>
        <ul style="margin-top:0.25rem; margin-bottom:0.5rem;">
            <li><b>Accuracy (mean ± std):</b> {metrics_cv['accuracy_mean']:.3f} ± {metrics_cv['accuracy_std']:.3f}</li>
            <li><b>Logarithmic Loss (mean ± std):</b> {metrics_cv['logloss_mean']:.3f} ± {metrics_cv['logloss_std']:.3f}</li>
            <li><b>ROC AUC (mean ± std):</b> {metrics_cv['auc_mean']:.3f} ± {metrics_cv['auc_std']:.3f}</li>
        </ul>
        <p style="font-size:0.9rem; margin-bottom:0;">
            These numbers show that the model is not accurate and but still has the ability
            to discriminate between CHD and non-CHD cases. It is <b>not 100% correct</b>.
            Some predictions will still be wrong, especially for borderline cases. The
            probability shown below should be interpreted as an <b>estimated risk</b>,
            not a guaranteed outcome.
        </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("Adjust the patient characteristics below to see how the estimated 10-year CHD risk changes.")

        # === 2. Short descriptions for each feature ===
        feature_descriptions = {
            "SEX": "Patient sex (numeric code; typically 1 = male, 2 = female).",
            "TOTCHOL": "Total cholesterol in mg/dL.",
            "AGE": "Age in years at baseline.",
            "SYSBP": "Systolic blood pressure in mmHg.",
            "DIABP": "Diastolic blood pressure in mmHg.",
            "CIGPDAY": "Number of cigarettes smoked per day.",
            "BMI": "Body Mass Index (kg/m²).",
            "DIABETES": "Diabetes indicator (0 = no diabetes, 1 = has diabetes).",
            "BPMEDS": "On blood pressure medication (0 = no, 1 = yes).",
            "HEARTRTE": "Heart rate (beats per minute).",
            "GLUCOSE": "Casual blood glucose (mg/dL).",
            "PREVCHD": "History of coronary heart disease at baseline (0/1).",
            "PREVMI": "History of myocardial infarction (heart attack) at baseline (0/1).",
            "PREVSTRK": "History of stroke at baseline (0/1).",
            "PREVHYP": "History of hypertension at baseline (0/1).",
        }

        # === 3. Two-column layout for inputs (widgets chosen per feature type) ===
        col_left, col_right = st.columns(2)
        user_input = {}

        # Columns that are truly binary 0/1
        binary_cols = ["DIABETES", "BPMEDS", "PREVCHD", "PREVMI", "PREVSTRK", "PREVHYP"]

        # Discrete coded variables with a small set of integer categories
        coded_discrete_cols = ["SEX",]

        # Real-world plausible ranges for continuous features: (min, max, default)
        real_ranges = {
            "AGE": (30.0, 90.0, 55.0),
            "TOTCHOL": (100.0, 350.0, 220.0),    # mg/dL
            "SYSBP": (80.0, 220.0, 130.0),       # mmHg
            "DIABP": (40.0, 130.0, 80.0),        # mmHg
            "CIGPDAY": (0.0, 60.0, 0.0),         # cigarettes/day
            "BMI": (15.0, 50.0, 26.0),           # kg/m^2
            "HEARTRTE": (40.0, 130.0, 70.0),     # bpm
            "GLUCOSE": (50.0, 300.0, 90.0),      # mg/dL
        }

        for i, col_name in enumerate(info_fram["feature_columns"]):
            target_col = col_left if i % 2 == 0 else col_right

            with target_col:
                desc = feature_descriptions.get(col_name, "")
                if desc:
                    st.markdown(f"**{col_name}**  \n<small>{desc}</small>", unsafe_allow_html=True)
                else:
                    st.markdown(f"**{col_name}**")

                # Get unique values for this column (helpful for discrete vars)
                unique_vals = sorted(df_fram[col_name].dropna().unique().tolist())

                # 1) Binary 0/1 → radio buttons
                if col_name in binary_cols and set(unique_vals).issubset({0, 1}):
                    options = [0, 1]
                    labels = [f"0 – No", f"1 – Yes"]
                    chosen = st.radio(
                        f"Select {col_name}",
                        options=options,
                        index=1 if 1 in unique_vals else 0,
                        format_func=lambda v: labels[options.index(v)],
                        horizontal=True,
                    )
                    user_input[col_name] = chosen

                # 2) Discrete coded variables → selectbox with actual codes
                elif col_name in coded_discrete_cols:
                    options = unique_vals
                    user_input[col_name] = st.selectbox(
                        f"Select {col_name}",
                        options=options,
                        index=0,
                    )

                # 3) Continuous numeric features → slider
                else:
                    # If we defined a real-world range, use it; otherwise fallback to data
                    if col_name in real_ranges:
                        col_min, col_max, col_default = real_ranges[col_name]
                    else:
                        col_min = float(df_fram[col_name].min())
                        col_max = float(df_fram[col_name].max())
                        col_default = float(df_fram[col_name].mean())

                    # If all values are integers, step = 1; otherwise step = 0.1
                    all_int = all(float(v).is_integer() for v in unique_vals)
                    step_val = 1.0 if all_int else 0.1

                    user_input[col_name] = st.slider(
                        label=f"{col_name}",
                        min_value=float(col_min),
                        max_value=float(col_max),
                        value=float(col_default),
                        step=step_val,
                    )

        st.markdown("")



        # === 4. Prediction + result card ===
        if st.button("Predict CHD risk"):
            input_df = pd.DataFrame([user_input])
            proba = clf_part1.predict_proba(input_df)[:, 1][0]
            pred_class = int(proba >= 0.5)

            risk_text = "higher estimated risk" if proba >= 0.5 else "lower estimated risk"

            st.markdown(f"""
            <div style="
                background-color: #0f172a;
                color: #ffffff;
                border-radius: 16px;
                padding: 1rem 1.25rem;
                box-shadow: 0 2px 10px rgba(15,23,42,0.12);
                margin-top: 0.5rem;
            ">
            <h4 style="margin-top:0;">Predicted 10-year CHD risk</h4>
            <p style="font-size:1.1rem; margin-bottom:0.5rem;">
                Estimated probability of developing CHD within 10 years:
                <b>{proba:.2%}</b>
            </p>
            <p style="margin-bottom:0.5rem;">
                Predicted class: <b>{pred_class}</b>
                <p style="font-size:0.9rem;">(0 = no CHD event, 1 = CHD event)</p>
            </p>
            <p style="font-size:0.9rem; margin-bottom:0.35rem;">
                This corresponds to a <b>{risk_text}</b> based on the selected
                combination of risk factors.
            </p>
            <p style="font-size:0.85rem; margin-bottom:0;">
                Remember that the model's accuracy is about {metrics_cv['accuracy_mean']:.0%},
                so there is still a chance of incorrect predictions. The output is meant
                to illustrate how this logistic regression model behaves, not to be used
                as a real medical diagnosis.
            </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Extra explanation: possible real-life uses
        st.markdown(f"""
        <div style="
            background-color: #ffffff;
            border-radius: 16px;
            padding: 1rem 1.25rem;
            box-shadow: 0 2px 10px rgba(15,23,42,0.08);
            margin-top: 0.75rem;
        ">
        <h4 style="margin-top:0;">How could this type of model be used in real life?</h4>
        <ul>
            <li><b>Risk stratification:</b> estimate which patients might have
                a higher risk of developing coronary heart disease and may
                benefit from closer monitoring or lifestyle counselling.</li>
            <li><b>Decision support:</b> help clinicians prioritize additional
                diagnostic tests for patients with higher predicted risk.</li>
            <li><b>Patient education:</b> show how changing factors such as
                blood pressure, smoking, or BMI could shift estimated risk
                in a simplified way.</li>
        </ul>
        <p style="font-size:0.85rem; margin-bottom:0;">
            In this web app, the model is used <b>only for educational purposes</b>
            to demonstrate how a logistic regression risk model works. It is
            <b>not a medical device</b> and should not be used to make real
            clinical decisions or replace advice from a health professional.
        </p>
        </div>
        """, unsafe_allow_html=True)

        


# ======================================
# PART 2 PAGE – REGRESSION
# ======================================

else:  # "Part 2 – Regression"
    st.title("Regression: Energy Efficiency (Heating Load)")

    df_energy, X_energy, y_energy, info_energy = load_and_preprocess_energy()
    metrics_a, metrics_b, reg_part2, info_model2 = train_part2_models()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview",
        "Dataset & Preprocessing",
        "Models & Metrics",
        "Try the Regressor",
    ])

    # ---------- Tab 1: Model Overview ----------
    with tab1:
        st.subheader("Model Overview")

        # 1. Metrics explanation – card style
        st.markdown(f"""
        <div style="
            background-color: #ffffff;
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
            box-shadow: 0 2px 10px rgba(15,23,42,0.08);
            border-left: 4px solid {ACCENT_COLOR};
            margin-bottom: 1rem;
        ">
        <h4 style="margin-top:0;">Performance metrics used in this app</h4>
        <ul>
            <li><b>Mean Squared Error (MSE)</b> – average of the squared differences
                between predicted and actual heating load values. Larger errors are
                penalized more strongly because the difference is squared. Lower MSE
                indicates better overall fit.</li>
            <li><b>Mean Absolute Error (MAE)</b> – average of the absolute differences
                between predicted and actual values, in the same units as the target.
                For example, an MAE around 2.2 means predictions are off by about
                2.2 units of heating load on average.</li>
            <li><b>R-squared (R²)</b> – proportion of the variance in heating load that
                is explained by the model. Values close to 1.0 mean the model captures
                most of the variation in the target. An R² around 0.91 means the model
                explains about 91% of the variability in heating load.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        # 2. Model options – side-by-side cards
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown(f"""
            <div style="
                background-color: #ffffff;
                border-radius: 16px;
                padding: 1.25rem 1.5rem;
                box-shadow: 0 2px 10px rgba(15,23,42,0.08);
                margin-bottom: 1rem;
            ">
            <h4 style="margin-top:0;">Model A – Single Train/Test Split</h4>
            <ul>
                <li>The dataset is split once into a <b>training set</b> (80%) and a
                    <b>test set</b> (20%).</li>
                <li>A linear regression model is trained on the training set and evaluated
                    once on the test set.</li>
                <li>Produces a single set of metrics (MSE, MAE, R²) for that particular split.</li>
                <li>Simple and fast, but the reported performance depends on this single
                    random partition; a “lucky” or “unlucky” split can slightly change the
                    metrics.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown(f"""
            <div style="
                background-color: #ffffff;
                border-radius: 16px;
                padding: 1.25rem 1.5rem;
                box-shadow: 0 2px 10px rgba(15,23,42,0.08);
                margin-bottom: 1rem;
            ">
            <h4 style="margin-top:0;">Model B – Repeated Random Train/Test Splits</h4>
            <ul>
                <li>The dataset is repeatedly split into training (80%) and test (20%)
                    using different random seeds (for example, 10 different splits).</li>
                <li>For each split, the same linear regression model is trained and
                    evaluated on the corresponding test set.</li>
                <li>This produces a <b>distribution</b> of MSE, MAE, and R² values across
                    all runs.</li>
                <li>The app summarizes these results using the <b>mean ± standard deviation</b>
                    of each metric, showing both typical performance and how stable it is
                    across different random partitions.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        # 3. Why focus on Model B (repeated random splits)
        st.markdown(f"""
        <div style="
            background-color: #0f172a;
            color: #ffffff;
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
            margin-top: 0.5rem;
        ">
        <h4 style="margin-top:0;">This app focuses on Repeated Random Train/Test Splits, Why?</h4>
        <ul>
            <li><b>Similar average performance.</b><br>
                In this dataset, the average MSE, MAE, and R² of Model B are very close
                to the metrics from Model A's single train/test split. Neither model is
                clearly better in terms of mean performance alone.</li>
            <li><b>Additional robustness information.</b><br>
                Model A only shows what happens on one particular split. Model B shows
                how the metrics change across many random splits and reports the
                <b>standard deviation</b> for each metric. Low standard deviations
                indicate that the model’s performance is stable across different
                partitions of the data.</li>
            <li><b>More convincing evaluation.</b><br>
                It is more informative to report results such as
                “MSE ≈ 9 with a standard deviation around 0.7, MAE ≈ 2.2, R² ≈ 0.91 ± 0.008”
                than to report only a single value from one split. This shows both
                typical accuracy and variability.</li>
            <li><b>Aligned with robust model assessment.</b><br>
                Repeated random train/test splits are conceptually similar to
                cross-validation in that they test the model on multiple partitions.
                This reduces the risk that the reported performance is just a fluke
                of one particularly easy or difficult split.</li>
        </ul>
        <p style="margin-bottom:0;">
            Because of these reasons, the <b>repeated-split linear regression model</b>
            (Model B) is treated as the main evaluation in this app. The next tab shows
            its metrics in detail.
        </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        

    # ---------- Tab 2: Dataset & Preprocessing ----------
    with tab2:
        st.subheader("Energy Efficiency Dataset – Dataset & Preprocessing")

        # === 1. High-level description ===
        top_left, top_right = st.columns([2, 1])

        with top_left:
            st.markdown("""
            The **Energy Efficiency** dataset (UCI repository) contains information
            about simulated residential buildings and their energy performance.

            Each row represents a building configuration described by 8 input
            features (geometry and glazing properties), and two target variables:

            - **Heating_Load (Y1)** – energy needed to heat the building  
            - **Cooling_Load (Y2)** – energy needed to cool the building  

            In this app, the focus is on **predicting Heating_Load** using the 8
            building features as inputs. This is a classic **regression problem**,
            because the target is a continuous numeric value.

            This dataset is well-suited for this project because:
            - All features are numeric and already clean.  
            - The relationships between geometry, glazing, and heating load are
            meaningful and smooth, which fits a linear regression model.  
            - The dataset is small enough to train quickly, but rich enough to show
            how evaluation with repeated random splits works.
            - **This can help explore how design choices affect energy demand, which is relevant to sustainable and environmentally-friendly building design**.
            """)

        with top_right:
            st.markdown(f"""
            <div style="
                background-color: #ffffff;
                border-radius: 16px;
                padding: 1rem 1.25rem;
                box-shadow: 0 2px 10px rgba(15,23,42,0.08);
                border-left: 4px solid {ACCENT_COLOR};
            ">
            <h4 style="margin-top:0; margin-bottom:0.5rem;">Basic Info</h4>
            <p style="margin:0;">
                <b>Rows:</b> {info_energy["X_shape"][0]}<br>
                <b>Features:</b> {info_energy["X_shape"][1]}<br>
                <b>Target:</b> Heating_Load
            </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # === 2. Preprocessing pipeline (card-style steps) ===
        st.markdown("### Preprocessing pipeline")

        step1, step2, step3 = st.columns(3)

        with step1:
            st.markdown("#### 1️⃣ Load data and rename columns")
            st.markdown("""
            - Read the original Excel file from the UCI dataset.  
            - Rename technical column names (`X1`–`X8`, `Y1`, `Y2`) into more
            descriptive names.
            """)
            st.code(
                '"X1" → "Relative_Compactness"\n'
                '"X2" → "Surface_Area"\n'
                '"X3" → "Wall_Area"\n'
                '"X4" → "Roof_Area"\n'
                '"X5" → "Overall_Height"\n'
                '"X6" → "Orientation"\n'
                '"X7" → "Glazing_Area"\n'
                '"X8" → "Glazing_Area_Distribution"\n'
                '"Y1" → "Heating_Load"\n'
                '"Y2" → "Cooling_Load"',
                language="text",
            )
            st.markdown(
                f"Raw shape: **{info_energy['raw_shape'][0]} × {info_energy['raw_shape'][1]}**  \n"
                f"After renaming: **{info_energy['after_rename_shape'][0]} × {info_energy['after_rename_shape'][1]}**"
            )

        with step2:
            st.markdown("#### 2️⃣ Check for missing values")
            st.markdown("""
            According to the dataset description, there should be **no missing
            values**. This is verified explicitly by checking `isnull().sum()` for
            each column.
            """)

            mv_series = pd.Series(info_energy["missing_values"])
            st.dataframe(mv_series.to_frame("Missing values"), use_container_width=True)
            st.caption("All columns have 0 missing values, so no rows needed to be dropped.")

        with step3:
            st.markdown("#### 3️⃣ Select features and target")
            st.markdown("""
            The regression model uses the 8 building properties as features **X**
            and `Heating_Load` as the target **y**.
            """)
            st.markdown("**Features (X):**")
            st.write(info_energy["feature_columns"])
            st.markdown("""
            **Target (y):** `Heating_Load`  

            All features are numeric, so they can directly be scaled and used in
            a Linear Regression model.
            """)

        st.markdown("---")

        # === 3. Scaling + model pipeline explanation (used later in training) ===
        st.markdown("### Modeling pipeline used in training")

        st.markdown(f"""
        <div style="
            background-color: #ffffff;
            border-radius: 16px;
            padding: 1rem 1.25rem;
            box-shadow: 0 2px 10px rgba(15,23,42,0.08);
            border-left: 4px solid {ACCENT_COLOR};
            margin-bottom: 1rem;
        ">
        <h4 style="margin-top:0;">StandardScaler + LinearRegression</h4>
        <p style="margin-bottom:0.5rem;">
            The model is built inside a <code>Pipeline</code> that performs:
        </p>
        <ul style="margin-top:0.25rem;">
            <li><b>StandardScaler</b> – centers each feature (mean 0) and scales it
                to unit variance. This ensures that features with larger numeric
                ranges (e.g., surface area) do not dominate the regression just
                because of their scale.</li>
            <li><b>LinearRegression</b> – fits a standard linear regression model to
                predict <code>Heating_Load</code> as a weighted combination of the
                scaled features.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        # === 4. Dataset preview (raw vs processed/renamed) ===
        st.markdown("### Dataset preview")

        raw_col, clean_col = st.columns(2)

        with raw_col:
            st.markdown("#### Raw dataset")
            df_raw_preview = pd.read_excel("ENBenergy_efficiency_2012_data.xlsx", sheet_name=0).head()
            st.dataframe(df_raw_preview, use_container_width=True)
            st.caption(f"Raw shape: {info_energy['raw_shape'][0]} rows × {info_energy['raw_shape'][1]} columns")

        with clean_col:
            st.markdown("#### Processed dataset")
            st.dataframe(df_energy.head(), use_container_width=True)
            st.caption(
                f"Processed shape: {info_energy['after_rename_shape'][0]} rows × {info_energy['after_rename_shape'][1]} columns"
            )

        st.markdown("""
        Even though no rows were dropped, this tab shows that the dataset was
        checked for missing values, renamed for clarity, and that the feature and
        target columns were explicitly selected before modeling.
        """)


    # ---------- Tab 3: Models & Metrics ----------
    with tab3:
        st.subheader("Models & Metrics – Repeated Random Train/Test Linear Regression")

        # === 1. Quick data summary ===
        top_left, top_right = st.columns([2, 1])

        with top_left:
            st.markdown("#### Dataset snapshot")
            st.dataframe(df_energy.head(), use_container_width=True)

        with top_right:
            st.markdown("#### ")
            st.markdown(f"""
            <div style="
                background-color: #ffffff;
                border-radius: 16px;
                padding: 1rem 1.25rem;
                box-shadow: 0 2px 10px rgba(15,23,42,0.08);
                border-left: 4px solid {ACCENT_COLOR};
            ">
            <h4 style="margin-top:0; margin-bottom:0.5rem;">Energy Efficiency dataset</h4>
            <p style="margin:0;">
                <b>Rows:</b> {info_energy["X_shape"][0]}<br>
                <b>Features:</b> {info_energy["X_shape"][1]}<br>
                <b>Target:</b> Heating_Load
            </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # === 2. Features and target definition ===
        st.markdown("### Features and target used in this model")

        feat_col, target_col = st.columns([3, 1])

        with feat_col:
            st.markdown(f"""
            <div style="
                background-color: #ffffff;
                border-radius: 16px;
                padding: 1rem 1.25rem;
                box-shadow: 0 2px 10px rgba(15,23,42,0.08);
                margin-bottom: 0.75rem;
            ">
            <h4 style="margin-top:0;">Feature matrix <code>X</code></h4>
            <p style="margin-bottom:0.25rem;">
                The model uses the following building properties as input features:
            </p>
            <ul style="margin-top:0.25rem;">
            """ + "".join([f"<li>{c}</li>" for c in info_energy["feature_columns"]]) + """
            </ul>
            </div>
            """, unsafe_allow_html=True)

        with target_col:
            st.markdown(f"""
            <div style="
                background-color: #0f172a;
                color: #ffffff;
                border-radius: 16px;
                padding: 1rem 1.25rem;
                box-shadow: 0 2px 10px rgba(15,23,42,0.08);
            ">
            <h4 style="margin-top:0;">Target <code>y</code></h4>
            <p style="margin-bottom:0.25rem;">
                <b>Heating_Load</b>
            </p>
            <p style="font-size:0.9rem; margin-bottom:0;">
                Continuous energy consumption needed to heat the building
                (arbitrary units from the original simulation).
            </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # === 3. Repeated-split model summary (Model B) ===
        st.markdown("### Repeated Random Train/Test Linear Regression (Model B)")

        # high-level metrics card
        st.markdown(f"""
        <div style="
            background-color: #ffffff;
            border-radius: 16px;
            padding: 1rem 1.25rem;
            box-shadow: 0 2px 10px rgba(15,23,42,0.08);
            border-left: 4px solid {ACCENT_COLOR};
            margin-bottom: 1rem;
        ">
        <h4 style="margin-top:0;">Overall performance (averaged over 10 random splits)</h4>
        <ul>
            <li><b>Mean Squared Error (MSE, mean ± std):</b>
                {metrics_b['mse_mean']:.3f} ± {metrics_b['mse_std']:.3f}</li>
            <li><b>Mean Absolute Error (MAE, mean ± std):</b>
                {metrics_b['mae_mean']:.3f} ± {metrics_b['mae_std']:.3f}</li>
            <li><b>R-squared (R², mean ± std):</b>
                {metrics_b['r2_mean']:.3f} ± {metrics_b['r2_std']:.3f}</li>
        </ul>
        <p style="font-size:0.9rem; margin-bottom:0;">
            These values summarize how the linear regression model performs across
            multiple random train/test splits. Reporting the mean and standard
            deviation gives both the typical error and how stable the model is
            when the dataset is partitioned in different ways.
        </p>
        </div>
        """, unsafe_allow_html=True)

        # === 4. Detailed view: Actual vs Predicted + residuals (from last split) ===
        y_test_last = metrics_b["y_test_last"]
        y_pred_last = metrics_b["y_pred_last"]

        scatter_col, resid_col = st.columns(2)

        with scatter_col:
            st.markdown("#### Actual vs Predicted (last random split)")

            fig_scatter, ax_scatter = plt.subplots()
            ax_scatter.scatter(y_test_last, y_pred_last, alpha=0.7)
            line_min = min(y_test_last.min(), y_pred_last.min())
            line_max = max(y_test_last.max(), y_pred_last.max())
            ax_scatter.plot([line_min, line_max], [line_min, line_max], linestyle="--")
            ax_scatter.set_xlabel("Actual Heating_Load")
            ax_scatter.set_ylabel("Predicted Heating_Load")
            ax_scatter.set_title("Actual vs Predicted – last random split")
            st.pyplot(fig_scatter)

            st.markdown("""
            <p style="font-size:0.9rem;">
            Points close to the dashed line indicate accurate predictions.
            The overall cloud of points near the line is consistent with the
            high R² value reported above.
            </p>
            """, unsafe_allow_html=True)

        with resid_col:
            st.markdown("#### Residuals (last random split)")

            residuals = y_test_last - y_pred_last
            fig_resid, ax_resid = plt.subplots()
            ax_resid.hist(residuals, bins=15)
            ax_resid.set_xlabel("Residual (Actual − Predicted)")
            ax_resid.set_ylabel("Count")
            ax_resid.set_title("Residual distribution – last random split")
            st.pyplot(fig_resid)

            st.markdown("""
            <p style="font-size:0.9rem;">
            Residuals show how far predictions are from the true values.
            A roughly centered distribution (around 0) with no extreme outliers
            supports that the linear regression model is fitting the data
            reasonably well.
            </p>
            """, unsafe_allow_html=True)

    # ---------- Tab 4: Try the Regressor ----------
    with tab4:
        st.subheader("Try the Heating Load Regressor")

        # === 1. Brief reminder of model performance ===
        st.markdown(f"""
        <div style="
            background-color: #ffffff;
            border-radius: 16px;
            padding: 1rem 1.25rem;
            box-shadow: 0 2px 10px rgba(15,23,42,0.08);
            border-left: 4px solid {ACCENT_COLOR};
            margin-bottom: 1rem;
        ">
        <h4 style="margin-top:0;">How reliable are these predictions?</h4>
        <p style="margin-bottom:0.5rem;">
            The regressor used here is the <b>linear regression model with repeated random train/test splits</b>
            trained on the Energy Efficiency dataset.
        </p>
        <ul style="margin-top:0.25rem; margin-bottom:0.5rem;">
            <li><b>MSE (mean ± std):</b> {metrics_b['mse_mean']:.3f} ± {metrics_b['mse_std']:.3f}</li>
            <li><b>MAE (mean ± std):</b> {metrics_b['mae_mean']:.3f} ± {metrics_b['mae_std']:.3f}</li>
            <li><b>R² (mean ± std):</b> {metrics_b['r2_mean']:.3f} ± {metrics_b['r2_std']:.3f}</li>
        </ul>
        <p style="font-size:0.9rem; margin-bottom:0;">
            These values show that the model usually predicts Heating_Load quite accurately
            (R² around {metrics_b['r2_mean']:.2f}), but it is still <b>not perfect</b>. Take note that the dataset only contains around 700+ rows.
            Some building configurations will have larger errors than others. The predicted
            value below should be interpreted as an <b>approximate estimate</b> for this
            simplified linear model, not an exact engineering calculation.
        </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("Adjust the building characteristics below to see how the estimated heating load changes.")

        # === 2. Short descriptions for each feature ===
        feature_descriptions_reg = {
            "Relative_Compactness": "Compactness of the building shape (higher = more compact).",
            "Surface_Area": "Total exterior surface area of the building.",
            "Wall_Area": "Total area of the exterior walls.",
            "Roof_Area": "Total roof surface area.",
            "Overall_Height": "Overall height of the building.",
            "Orientation": "Cardinal orientation (encoded as integers, e.g., 2–5).",
            "Glazing_Area": "Fraction of the facade area that is glazed (windows).",
            "Glazing_Area_Distribution": "Distribution of glazing on the building’s facades.",
        }

        # === 3. Two-column layout for inputs ===
        col_left, col_right = st.columns(2)
        user_input_reg = {}

        # Features from the preprocessing info (same order as training)
        feature_cols_reg = info_energy["feature_columns"]

        # For this dataset, Orientation and Glazing_Area_Distribution are discrete codes
        coded_discrete_reg = ["Orientation", "Glazing_Area_Distribution"]

        for i, col_name in enumerate(feature_cols_reg):
            target_col = col_left if i % 2 == 0 else col_right

            # Categorical code → human-readable labels
            orientation_labels = {
                2: "2 – North-facing",
                3: "3 – East-facing",
                4: "4 – South-facing",
                5: "5 – West-facing",
            }

            glazing_dist_labels = {
                0: "0 – Uniform glazing",
                1: "1 – North side only",
                2: "2 – East side only",
                3: "3 – South side only",
                4: "4 – West side only",
                5: "5 – No glazing",
            }


            with target_col:
                desc = feature_descriptions_reg.get(col_name, "")
                
                if desc:
                    st.markdown(f"**{col_name}**  \n<small>{desc}</small>", unsafe_allow_html=True)
                else:
                    st.markdown(f"**{col_name}**")


                # unique values from dataset (useful for discrete coded vars)
                unique_vals = sorted(df_energy[col_name].dropna().unique().tolist())



                # 1) Discrete coded variables → selectbox with meaningful labels
                if col_name in coded_discrete_reg:
                    options = unique_vals

                    if col_name == "Orientation":
                        user_input_reg[col_name] = st.selectbox(
                            f"Select {col_name}",
                            options=options,
                            index=0,
                            format_func=lambda v: orientation_labels.get(v, str(v)),
                        )

                    elif col_name == "Glazing_Area_Distribution":
                        user_input_reg[col_name] = st.selectbox(
                            f"Select {col_name}",
                            options=options,
                            index=0,
                            format_func=lambda v: glazing_dist_labels.get(v, str(v)),
                        )


                # 2) Continuous numeric features → slider using data range
                else:
                    col_min = float(df_energy[col_name].min())
                    col_max = float(df_energy[col_name].max())
                    col_default = float(df_energy[col_name].mean())

                    # If all values are integers, step = 1; otherwise step = 0.01
                    all_int = all(float(v).is_integer() for v in unique_vals)
                    step_val = 1.0 if all_int else 0.01

                    user_input_reg[col_name] = st.slider(
                        label=f"{col_name}",
                        min_value=float(col_min),
                        max_value=float(col_max),
                        value=float(col_default),
                        step=step_val,
                    )

        st.markdown("")

        # === 4. Prediction + result card ===
        if st.button("Predict heating load"):
            input_df_reg = pd.DataFrame([user_input_reg])
            # Use Model B pipeline for prediction
            y_hat = reg_part2.predict(input_df_reg)[0]

            # Optional: rough context from dataset
            y_min = float(y_energy.min())
            y_max = float(y_energy.max())
            y_mean = float(y_energy.mean())

            st.markdown(f"""
            <div style="
                background-color: #0f172a;
                color: #ffffff;
                border-radius: 16px;
                padding: 1rem 1.25rem;
                box-shadow: 0 2px 10px rgba(15,23,42,0.12);
                margin-top: 0.5rem;
            ">
            <h4 style="margin-top:0;">Predicted heating load</h4>
            <p style="font-size:1.1rem; margin-bottom:0.5rem;">
                Estimated <b>Heating_Load</b> for this building configuration:
                <b>{y_hat:.2f}</b>
            </p>
            <p style="font-size:0.9rem; margin-bottom:0.5rem;">
                For reference, in the dataset the Heating_Load values range roughly from
                {y_min:.2f} to {y_max:.2f}, with an average around {y_mean:.2f}.
            </p>
            <p style="font-size:0.85rem; margin-bottom:0;">
                This estimate is produced by the repeated-split linear regression model
                (Model B) with an R² of about {metrics_b['r2_mean']:.2f}. It is useful
                for understanding how the model reacts to different feature values, but
                it should not be treated as an exact engineering design calculation.
            </p>
            </div>
            """, unsafe_allow_html=True)

        
        st.markdown("---")
        # Extra explanation: possible real-life uses
        st.markdown(f"""
        <div style="
            background-color: #ffffff;
            border-radius: 16px;
            padding: 1rem 1.25rem;
            box-shadow: 0 2px 10px rgba(15,23,42,0.08);
            margin-top: 0.75rem;
        ">
        <h4 style="margin-top:0;">How could this type of model be used in real life?</h4>
        <ul>
            <li><b>Early design exploration:</b> quickly estimate how changes in
                geometry or glazing might affect heating demand in a simulated
                building, before running more detailed simulations.</li>
            <li><b>What-if analysis:</b> compare different design options
                (e.g., more compact shape, different glazing area) to see which
                ones tend to reduce predicted heating load.</li>
            <li><b>Educational tools:</b> help students and engineers build
                intuition about how high-level building parameters influence
                energy performance.</li>
        </ul>
        <p style="font-size:0.85rem; margin-bottom:0;">
            In this web app, the model is a <b>simplified linear regression</b>
            trained on a small benchmark dataset. The predictions are intended
            for <b>learning and demonstration only</b>, not as a substitute for
            full building energy simulations or professional engineering design.
        </p>
        </div>
        """, unsafe_allow_html=True)



