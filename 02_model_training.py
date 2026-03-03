import pandas as pd
import numpy as np
import os
import sys

# ML imports
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix, classification_report
from sklearn.calibration import calibration_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
DATA_PATH = '/home/UserData/ljx/beidabingli/队列符合277.xlsx'
OUTPUT_DIR = '/home/UserData/ljx/beidabingli/results'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Load Data
print("Loading data...")
df = pd.read_excel(DATA_PATH)

# Basic Cleanup
# Drop utility columns
drop_cols = ['Unnamed: 26', 'number', 'Biopsydate', '病理扫片']
df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns])

# --- Feature Engineering & Handling Missing Values ---

# 1. Define Variable Sets
target_vars = ['label1', 'label2']

# Clinical Core (Low missing, high value)
# Note: Choose MAP over SBP/DBP. Choose GFR over Scr.
clinical_core_vars = [
    'age', 'gender', 
    'baseline GFR', # vs baseline Scr
    'baseline UTP', 
    'MAP',          # vs SBP, DBP
    'Alb',          # Albumin, usually important
    'RASB',
    '尿酸'          # Uric acid, ~9 missing, manageable
]

# Clinical Extended (High missing, potential value)
# '前驱感染', '肉眼血尿' have ~50% missing. 
# '血尿（RBC）' ~25% missing
# 'Hb' ~9% missing
# 'IgA', 'C3' ~25% missing
clinical_ext_vars = ['Hb', '血尿（RBC）', 'IgA', 'C3'] 

# Check binary high missing vars
# If they are mostly 1s, maybe Nan is 0? Or true missing?
# For safety in "Robust" model, we might exclude them or create binary flag.
# Let's excluded them from "Core" but add to "Extended" with imputation.
clinical_high_missing_vars = ['前驱感染', '肉眼血尿']

# MEST-C
mest_vars = ['M', 'E', 'S', 'T', 'C']

# 2. Imputation Strategy
# For continuous vars: KNN Imputation (better than Median for clinical)
# For categorical: Mode (Frequent)
# For now, we will do a global imputation for simplicity in this script, 
# although strictly it should be inside CV. For N=277, doing it inside CV is better but code is complex.
# We'll use a Pipeline in CV to be rigorous.

# --- Prepare Experiment Data Structure ---
# We need to test:
# Task 1: Label1 (6mo) | Task 2: Label2 (12mo)
# FeatSet 1: Clinical Core
# FeatSet 2: Clinical Core + MEST-C

experiments = [
    {'name': 'ShortTerm_Clinical', 'label': 'label1', 'feats': clinical_core_vars},
    {'name': 'ShortTerm_Clin+MEST', 'label': 'label1', 'feats': clinical_core_vars + mest_vars},
    {'name': 'LongTerm_Clinical', 'label': 'label2', 'feats': clinical_core_vars},
    {'name': 'LongTerm_Clin+MEST', 'label': 'label2', 'feats': clinical_core_vars + mest_vars},
]

# --- Helper Functions ---

def get_calibration_curve_plot(y_true, y_prob, title, filename):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=5)
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def run_experiment(df, exp_config):
    name = exp_config['name']
    label = exp_config['label']
    feat_cols = exp_config['feats']
    
    print(f"\n=== Running {name} ===")
    
    # Filter rows where target is not null (should be none, but safe check)
    data = df.dropna(subset=[label]).copy()
    X = data[feat_cols]
    y = data[label]
    
    # Preprocessing Pipeline
    # Identify numeric and categorical columns
    # M, E, S are 0/1 (binary/cat). T, C are 0/1/2 (ordinal/cat).
    # Clinical Core: Gender (cat), RASB (cat), others numeric.
    
    # Heuristic for cat vs num
    cat_cols = [c for c in feat_cols if c in ['gender', 'RASB'] + mest_vars]
    num_cols = [c for c in feat_cols if c not in cat_cols]
    
    # Num transformer: Impute Median -> Scale
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Cat transformer: Impute Mode -> (Optional: OneHot). 
    # MEST are ordinal-ish, but OneHot is safer for Linear Models. 
    # But given small N, OneHot might explode dims (T=0,1,2 -> 3 cols). 
    # Let's keep them as numeric (ordinal) for T/C if we assume linearity, or OneHot.
    # For robust plan: SimpleImputer(most_frequent) -> Leave as is (treated as numeric for ElasticNet if ordinal).
    # Actually, T and C (0,1,2) have order. 
    # Let's treat all as numeric for simplicity in Logistic Regression unless strictly categorical.
    # Gender is 1/2. Need to change to 0/1 or OneHot.
    
    # Custom Preprocessor to handle this:
    # Just use SimpleImputer for everything (Median for all is okay if coded numerically).
    # Gender 1/2 -> Let's Standardization will make it -1/1. Okay for LR.
    
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Models
    # 1. Logistic Regression (Elastic Net)
    lr = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=5000, class_weight='balanced', random_state=42)
    pipeline_lr = Pipeline([
        ('preprocessor', preprocessor),
        ('model', lr)
    ])
    
    # 2. XGBoost (Tree)
    # Monotonic constraints? No.
    # Max depth small for small data.
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=3, 
        learning_rate=0.05, 
        eval_metric='logloss',
        random_state=42
    )
    # XGB handles missing, but pipeline scaler helps interpretation? XGB doesn't need scaling.
    # But Imputation is needed for fairness comparison? XGB handles NaN.
    # Let's fit XGB directly on X (with simple cleaned X).
    
    # Evaluation
    # 5-fold Stratified CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    metrics = {
        'LR_AUC': [], 'LR_Brier': [], 
        'XGB_AUC': [], 'XGB_Brier': []
    }
    
    all_y_true = []
    all_lr_probs = []
    all_xgb_probs = []
    
    fold = 0
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # LR
        pipeline_lr.fit(X_train, y_train)
        y_pred_lr = pipeline_lr.predict_proba(X_val)[:, 1]
        
        # XGB
        # Simple impute for XGB just to be safe (or let it handle nan)
        # Let's use the preprocessor logic but without scaler for XGB? 
        # Actually XGB works with NaN. Standardization doesn't hurt.
        # Let's use same parsed data.
        X_train_proc = preprocessor.fit_transform(X_train)
        X_val_proc = preprocessor.transform(X_val) 
        # Note: Preprocessor returns numpy array, losing col names. 
        # XGB likes col names for SHAP.
        # Let's recreate DF
        X_train_xgb = pd.DataFrame(X_train_proc, columns=feat_cols)
        X_val_xgb = pd.DataFrame(X_val_proc, columns=feat_cols)
        
        xgb_model.fit(X_train_xgb, y_train)
        y_pred_xgb = xgb_model.predict_proba(X_val_xgb)[:, 1]
        
        # Log metrics
        metrics['LR_AUC'].append(roc_auc_score(y_val, y_pred_lr))
        metrics['LR_Brier'].append(brier_score_loss(y_val, y_pred_lr))
        metrics['XGB_AUC'].append(roc_auc_score(y_val, y_pred_xgb))
        metrics['XGB_Brier'].append(brier_score_loss(y_val, y_pred_xgb))
        
        all_y_true.extend(y_val)
        all_lr_probs.extend(y_pred_lr)
        all_xgb_probs.extend(y_pred_xgb)
        
        fold += 1

    # Report
    print(f"Results for {name}:")
    print(f"  LR  - Avg AUC: {np.mean(metrics['LR_AUC']):.4f} (+/- {np.std(metrics['LR_AUC']):.4f})")
    print(f"  LR  - Avg Brier: {np.mean(metrics['LR_Brier']):.4f}")
    print(f"  XGB - Avg AUC: {np.mean(metrics['XGB_AUC']):.4f} (+/- {np.std(metrics['XGB_AUC']):.4f})")
    print(f"  XGB - Avg Brier: {np.mean(metrics['XGB_Brier']):.4f}")
    
    # Plots
    get_calibration_curve_plot(all_y_true, all_lr_probs, f"Calibration LR - {name}", f"calib_lr_{name}.png")
    get_calibration_curve_plot(all_y_true, all_xgb_probs, f"Calibration XGB - {name}", f"calib_xgb_{name}.png")
    
    # Final Full Model for Interpretation (LR)
    pipeline_lr.fit(X, y)
    coefs = pipeline_lr.named_steps['model'].coef_[0]
    coef_df = pd.DataFrame({'Feature': feat_cols, 'Coef': coefs}).sort_values(by='Coef', ascending=False)
    print("\n  Top Coefficients (LR):")
    print(coef_df.to_string())
    
    # Save SHAP for XGB (trained on full data)
    try:
        X_proc = preprocessor.fit_transform(X)
        X_xgb_full = pd.DataFrame(X_proc, columns=feat_cols)
        xgb_model.fit(X_xgb_full, y)
        
        # Use Generic Explainer (Model Agnostic) to avoid XGBoost/SHAP version conflict
        # Wrap predict_proba to return only the positive class probability
        def model_predict(data_asarray):
            data_df = pd.DataFrame(data_asarray, columns=feat_cols)
            return xgb_model.predict_proba(data_df)[:, 1]

        # Use a background sample for speed and stability
        background = shap.maskers.Independent(X_xgb_full, max_samples=100)
        explainer = shap.Explainer(model_predict, background)
        
        shap_values = explainer(X_xgb_full)
        
        # Debug: Check if SHAP values are non-zero
        print(f"  SHAP values shape: {shap_values.shape}")
        print(f"  Max SHAP value: {np.max(np.abs(shap_values.values))}")

        # Fix Chinese Font Issue by temporary renaming
        col_mapping = {
            '尿酸': 'Uric Acid', '性别': 'Gender', '前驱感染': 'Infection',
            '肉眼血尿': 'Gross Hemat.', '血尿（RBC）': 'RBC',
            'baseline GFR': 'eGFR', 'baseline UTP': 'UTP'
        }
        display_names = [col_mapping.get(c, c) for c in feat_cols]

        # IMPORTANT: Use raw SHAP arrays for stable plotting.
        # summary_plot can render blank in some headless setups; use a custom bar plot instead.
        shap_vals_array = shap_values.values
        mean_abs_shap = np.mean(np.abs(shap_vals_array), axis=0)
        
        # Sort by importance
        sorted_idx = np.argsort(mean_abs_shap)
        sorted_features = [display_names[i] for i in sorted_idx]
        sorted_values = mean_abs_shap[sorted_idx]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(sorted_features, sorted_values, color='#4C72B0')
        ax.set_xlabel('Mean |SHAP value|')
        ax.set_title(f'SHAP Feature Importance (XGBoost) - {name}')
        fig.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, f"shap_summary_{name}.png")
        fig.savefig(out_path)
        print(f"  SHAP plot saved: {out_path} (bytes={os.path.getsize(out_path)})")
        plt.close(fig)
    except Exception as e:
        print(f"  [Warning] SHAP plot failed: {e}")

# Run all
for exp in experiments:
    run_experiment(df_clean, exp)

print("\nAll experiments completed. Results saved to ./results")
