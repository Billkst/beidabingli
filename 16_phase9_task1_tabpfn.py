import os
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import joblib

# Try importing TabPFN
try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    print("TabPFN is not installed. Please install it using `pip install tabpfn`.")

DATA_PATH = '/home/UserData/ljx/beidabingli/队列符合277.xlsx'
RESULT_DIR = '/home/UserData/ljx/beidabingli/results_phase9'
os.makedirs(RESULT_DIR, exist_ok=True)

# Feature definitions
core_vars = ['age', 'gender', 'baseline GFR', 'baseline UTP', 'MAP', 'Alb', 'RASB', '尿酸']
mest_vars = ['M', 'E', 'S', 'T', 'C']
high_missing_vars = ['前驱感染', '肉眼血尿', 'IgA', 'C3', '血尿（RBC）', 'Hb']

def load_and_preprocess(label_col):
    _df = pd.read_excel(DATA_PATH)
    DROP_COLS = ['Unnamed: 26', 'number', 'Biopsydate', '病理扫片']
    df = _df.drop(columns=[c for c in DROP_COLS if c in _df.columns])
    
    # Select features based on task
    if label_col == 'label1': # Short term
        feature_cols = core_vars + high_missing_vars
    else: # Long term
        feature_cols = core_vars + mest_vars + high_missing_vars
    
    data = df.dropna(subset=[label_col]).copy()
    X = data[feature_cols]
    y = data[label_col].astype(int)
    
    return X, y

def add_features(X_in):
    X = X_in.copy()
    # Manual feature engineering (same as previous phases)
    for col in high_missing_vars:
        if col in X.columns:
            X[f'{col}_missing'] = X[col].isna().astype(int)

    if 'IgA' in X.columns and 'C3' in X.columns:
        X['IgA_C3_ratio'] = X['IgA'] / X['C3'].replace(0, np.nan)

    if 'baseline UTP' in X.columns and 'S' in X.columns:
        X['UTP_x_S'] = X['baseline UTP'] * X['S']
        
    # Get dummies
    X = pd.get_dummies(X, columns=['gender'] if 'gender' in X.columns else [], drop_first=True)
    
    return X

def run_tabpfn_experiment():
    if not TABPFN_AVAILABLE:
        return

    results = []
    
    for label_col, task_name in [('label1', 'ShortTerm'), ('label2', 'LongTerm')]:
        print(f"Processing {task_name} ({label_col})...")
        X_raw, y = load_and_preprocess(label_col)
        
        # Add basic features first
        X_eng = add_features(X_raw)
        
        # MICE Imputation
        imputer = IterativeImputer(max_iter=10, random_state=42)
        X_imputed = pd.DataFrame(imputer.fit_transform(X_eng), columns=X_eng.columns)
        
        # Scaling is not strictly necessary for TabPFN but good practice
        # TabPFN handles arbitrary scaling, but let's standardise
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        # Validation scheme: Repeated 5x5
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
        
        fold_aucs = []
        fold_briers = []
        fold_sens = []
        fold_spec = []

        for i, (train_idx, val_idx) in enumerate(rskf.split(X_scaled, y)):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # TabPFN - n_estimators replaces N_ensemble_configurations in newer versions
            # Use local checkpoint
            ckpt_path = '/home/UserData/ljx/beidabingli/tabpfn-v2.5-classifier-v2.5_default.ckpt'
            clf = TabPFNClassifier(device='cuda', n_estimators=32, model_path=ckpt_path)
            clf.fit(X_train, y_train)
            
            # Predict
            y_pred_proba = clf.predict_proba(X_val)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            auc = roc_auc_score(y_val, y_pred_proba)
            brier = brier_score_loss(y_val, y_pred_proba)
            
            # For Sens/Spec, we use Youden index threshold or just 0.5? 
            # Phase 8 used Youden. Let's stick to simple 0.5 for now or calculate best.
            # Using 0.5 for stability in reporting loop, or youden?
            # Let's simple check recall (Sens) and TNR (Spec) at 0.5 (TabPFN is calibrated)
            sens = recall_score(y_val, y_pred)
            spec = recall_score(y_val, y_pred, pos_label=0)
            
            fold_aucs.append(auc)
            fold_briers.append(brier)
            fold_sens.append(sens)
            fold_spec.append(spec)
            
            if i % 5 == 0:
                print(f"  Fold {i} AUC: {auc:.4f}")

        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)
        print(f"Task {task_name} Result: AUC={mean_auc:.4f} +/- {std_auc:.4f}")
        
        results.append({
            'Task': task_name,
            'Model': 'TabPFN',
            'AUC_Mean': mean_auc,
            'AUC_Std': std_auc,
            'Brier': np.mean(fold_briers),
            'Sensitivity': np.mean(fold_sens),
            'Specificity': np.mean(fold_spec)
        })

    # Save results
    res_df = pd.DataFrame(results)
    res_path = os.path.join(RESULT_DIR, 'phase9_task1_tabpfn_results.csv')
    res_df.to_csv(res_path, index=False)
    print(f"Results saved to {res_path}")

if __name__ == "__main__":
    run_tabpfn_experiment()
