# 🏦 TermTrack — Bank Term Deposit Subscription Predictor

> An end-to-end machine learning system that predicts whether a bank customer will subscribe to a term deposit — from raw data exploration to a deployed Streamlit web application.

---

## 📌 Project Overview

Banks run telephone marketing campaigns to promote term deposit products. Most calls fail — agents waste time, budgets are burned, and customers receive irrelevant outreach.

**TermTrack** solves this by predicting, before the call is made, whether a specific customer is likely to subscribe. By targeting only high-probability customers, banks can:

- Improve campaign ROI
- Reduce operational costs from unsuccessful calls
- Enhance customer experience through personalised outreach

---

## 🎯 Problem Formulation

| | |
|---|---|
| **Type** | Supervised Binary Classification |
| **Input** | Customer demographics, financial profile, campaign history |
| **Output** | Will the customer subscribe? (Yes = 1 / No = 0) |
| **Challenge** | Severe class imbalance — 88% No, 12% Yes |
| **Primary Metric** | PR-AUC (Precision-Recall AUC) |

---

## 📁 Project Structure

```
TermTrack/
│
├── app/                    # Streamlit application
├── datasets/               # Original dataset
├── models/                 # Pickle files
├── notebooks/              # Exploratory Data Analysis & model building
├── reports/                # Data Visualizations
├── requirements.txt        # Dependencies
├── .python-version         # Python version to run the application
└── README.md               # Project documentation
```

---

## 📊 Dataset

- **Source:** [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- **Records:** 45,211 rows × 17 columns
- **Features:** 7 numerical, 10 categorical
- **Target:** `y` — whether the client subscribed to a term deposit

| Feature | Type | Description |
|---|---|---|
| `age` | Numerical | Client age |
| `job` | Categorical | Type of job |
| `marital` | Categorical | Marital status |
| `education` | Categorical | Education level |
| `has_credit` | Binary | Has credit default? |
| `balance` | Numerical | Average yearly account balance (€) |
| `housing_loan` | Binary | Has housing loan? |
| `personal_loan` | Binary | Has personal loan? |
| `contact` | Categorical | Contact communication type |
| `month` | Categorical | Last contact month |
| `day` | Numerical | Last contact day of month |
| `duration` | Numerical | Last contact duration (seconds) |
| `total_calls` | Numerical | Number of contacts in this campaign |
| `pdays` | Numerical | Days since last contact (0 = never) |
| `p_total_calls` | Numerical | Contacts in previous campaigns |
| `p_y` | Categorical | Outcome of previous campaign |

---

## 🔬 Methodology

### 1. Exploratory Data Analysis
- Numerical distributions by target class — `duration` showed the clearest separation
- Categorical subscription rates — students (28%), retired (22%), March/Sep/Oct/Dec calls highest
- Target imbalance identified: **88% No vs 12% Yes**

### 2. Data Preprocessing
- Replaced `pdays = -1` (never contacted) with `0`
- Cleaned inconsistent job label `admin.` → `admin`
- Applied **Box-Cox transformation** to right-skewed features: `duration`, `total_calls`, `pdays`, `p_total_calls`
- Converted `category` dtype to `str` before split to prevent dtype mismatch at inference

### 3. Train-Test Split
- 80/20 stratified split preserving the 88:12 class ratio
- Test set never used during training or calibration

### 4. Feature Scaling & Encoding

| Feature Group | Encoder | Reason |
|---|---|---|
| Numerical | `RobustScaler` | Handles outliers in `balance`, `pdays` |
| Binary | Passthrough | Already 0/1 |
| `education`, `p_y` | `OrdinalEncoder` | Natural ordering exists |
| `job`, `contact`, `marital` | `TargetEncoder` | No natural order; target rates vary widely |

### 5. Feature Selection
- **Correlation heatmap** on encoded features — identified `pdays` / `p_total_calls` overlap
- **Mutual Information** — `duration` ranked highest; `day` dropped (near-zero MI)
- Final feature count: **9 features**

### 6. Class Imbalance Handling
- **SMOTEENN** (SMOTE + Edited Nearest Neighbours) applied inside the pipeline
- Runs after encoding, per CV fold — zero data leakage
- Never touches test data

### 7. Model Selection

| Model | Notes |
|---|---|
| Logistic Regression | Linear baseline |
| Random Forest | Robust, parallelisable |
| Gradient Boosting | Shallow trees, moderate speed |
| XGBoost | `scale_pos_weight` for imbalance |
| LightGBM | Fastest, best on large tabular data |

Primary metric: **PR-AUC** (not ROC-AUC — misleadingly optimistic on imbalanced data)

### 8. Hyperparameter Tuning
- `RandomizedSearchCV` — 40 iterations
- `StratifiedKFold(5)` — preserves class ratio per fold
- Parameters tuned: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, regularisation terms
- Pipeline param prefix: `model__` (e.g. `model__learning_rate`)

### 9. Probability Calibration
- Raw model probabilities were miscalibrated after SMOTEENN resampling
- Applied **Isotonic Calibration** (`CalibratedClassifierCV`, `method='isotonic'`, `cv='prefit'`)
- Calibrated on original (non-resampled) training data to align with real class distribution
- Optimal decision threshold found by scanning F1-Yes across `[0.05, 0.80]`

---

## 📈 Results

| Metric | Score |
|---|---|
| **ROC-AUC** | 0.92 |
| **PR-AUC** | 0.59 |
| **F1-Macro** | 0.66 |
| **Baseline PR-AUC** (random) | 0.12 |

> PR-AUC of 0.59 is **~5× above the random baseline** of 0.12 on a severely imbalanced dataset.

---

## 💾 Model Bundle

Saved via `cloudpickle`:

```python
model_bundle = {
    'preprocessor'    : preprocessor_final,   # Fitted ColumnTransformer
    'calibrated_model': calibrated_model,      # Isotonic calibrated model
    'keep_idx'        : keep_idx,              # Feature selection indices
    'best_model_name' : best_name,
    'keep_cols'       : KEEP_COLS,
    'feature_names'   : feature_names,
    'lambda_values'   : lambda_values,         # Box-Cox lambdas per feature
    'TRANSFORM_COLS'  : TRANSFORM_COLS,        # Columns needing Box-Cox
    'threshold'       : float(best_t),         # Optimal decision threshold
    'metrics'         : { 'f1_macro', 'roc_auc', 'pr_auc' }
}
```

### Inference Pipeline (Streamlit)

```
Raw Input
    ↓
Box-Cox Transform (using saved lambdas)
    ↓
ColumnTransformer (RobustScaler + OrdinalEncoder + TargetEncoder)
    ↓
Feature Selection (keep_idx slice)
    ↓
Calibrated Model → P(subscribe)
    ↓
Threshold Decision → Yes / No
```

---

## 🚀 Running the App

### Prerequisites

```bash
pip install streamlit cloudpickle pandas numpy scikit-learn lightgbm xgboost imbalanced-learn
```

### Launch

```bash
streamlit run streamlit.py
```

---

## 🖥️ Streamlit Application

The web app accepts all 16 customer features across three input sections:

- **Client Demographics** — age, job, marital status, education, credit default, balance, loans
- **Contact Information** — contact type, month, day, call duration
- **Campaign History** — total calls, days since last contact, previous calls, previous outcome

**Output:**
- ✅ / ❌ subscription prediction
- Live probability score with decision threshold
- Model performance metrics (F1-Macro, ROC-AUC, PR-AUC)

---

## ⚠️ Known Limitations

| Limitation | Detail |
|---|---|
| **Duration leakage** | `duration` is a post-call feature — unknown before the call. For pre-call prospecting, retrain without it |
| **Temporal validity** | Model trained on historical data; campaign patterns may shift over time |
| **Threshold sensitivity** | Optimal threshold was found on test set — may need recalibration on new data distributions |

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data & EDA | `pandas`, `numpy`, `matplotlib`, `seaborn` |
| Preprocessing | `scikit-learn` — `RobustScaler`, `OrdinalEncoder`, `TargetEncoder`, `ColumnTransformer` |
| Imbalance | `imbalanced-learn` — `SMOTEENN`, `ImbPipeline` |
| Modelling | `XGBoost`, `LightGBM`, `scikit-learn` |
| Calibration | `scikit-learn` — `CalibratedClassifierCV` |
| Serialisation | `cloudpickle` |
| Deployment | `Streamlit` |

---

## 📌 Key Engineering Decisions

1. **PR-AUC over ROC-AUC** — ROC-AUC is misleadingly optimistic on 88:12 imbalanced data
2. **Mixed encoding strategy** — encoder choice based on data semantics, not convenience
3. **SMOTEENN inside the pipeline** — resampling per fold, never touching test data
4. **Isotonic calibration** — mandatory when training on resampled data and evaluating on real distribution
5. **cloudpickle over pickle** — required for serializing closures and custom transformer classes
6. **dtype consistency** — converting `category` → `str` before split prevents silent inference mismatch

---

*TermTrack — Predict smarter. Call better.*
