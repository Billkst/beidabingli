import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import matplotlib.pyplot as plt

import dca_utils

DATA_PATH = '/home/UserData/ljx/beidabingli/队列符合277.xlsx'
OUTPUT_DIR = '/home/UserData/ljx/beidabingli/results_phase8_xgb_mono'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load Data
def load_data():
    _df = pd.read_excel(DATA_PATH)
    DROP_COLS = ['Unnamed: 26', 'number', 'Biopsydate', '病理扫片']
    df = _df.drop(columns=[c for c in DROP_COLS if c in _df.columns])
    return df

core_vars = ['age', 'gender', 'baseline GFR', 'baseline UTP', 'MAP', 'Alb', 'RASB', '尿酸']
mest_vars = ['M', 'E', 'S', 'T', 'C']
high_missing_vars = ['前驱感染', '肉眼血尿', 'IgA', 'C3', '血尿（RBC）', 'Hb']

def add_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in high_missing_vars:
        if col in X.columns:
            X[f'{col}_missing'] = X[col].isna().astype(int)
    if 'IgA' in X.columns and 'C3' in X.columns:
        X['IgA_C3_ratio'] = X['IgA'] / X['C3'].replace(0, np.nan)
    if 'baseline UTP' in X.columns and 'S' in X.columns:
        X['UTP_x_S'] = X['baseline UTP'] * X['S']
    return X

def get_monotone_constraints(features):
    # Mapping: Variable -> Constraint
    # 1: Increasing, -1: Decreasing, 0: No constraint
    constraints_map = {
        'baseline GFR': 1,
        'baseline UTP': -1,
        'MAP': -1,
        'age': -1,
        'Alb': 1
    }
    
    constraints = []
    for f in features:
        # Check partial match or exact match
        # The exact names must match input features to XGBoost
        c = constraints_map.get(f, 0)
        constraints.append(c)
        
    return tuple(constraints)

def run_experiment():
    df = load_data()
    
    experiments = [
        {'name': 'ShortTerm_XGB_Mono', 'label': 'label1'},
        {'name': 'LongTerm_XGB_Mono', 'label': 'label2'}
    ]
    
    # Feature set: Use Core + MEST (Best configuration generally)
    base_features = core_vars + mest_vars
    # All raw columns needed for engineering
    all_needed = list(dict.fromkeys(base_features + high_missing_vars))
    
    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for exp in experiments:
        name = exp['name']
        label = exp['label']
        
        print(f"\nRunning {name}...")
        
        data = df.dropna(subset=[label]).copy()
        X = data[all_needed]
        y = data[label]
        
        # Pre-process to get feature names after engineering
        # We need to do this outside pipeline first to determine constraints tuple length/order
        X_sample = add_features(X.iloc[:5])
        feature_names = X_sample.columns.tolist()
        monotonic_tuple = get_monotone_constraints(feature_names)
        
        print(f"  Constraints: {monotonic_tuple}")
        
        aucs = []
        briers = []
        all_y_true = []
        all_y_prob = []
        
        # Pipeline construction needs to handle feature engineering then model
        # But XGBoost constraints need to match column order.
        # KNNImputer preserves order.
        
        for i, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Manual pipeline to ensure we know column order
            X_train_eng = add_features(X_train)
            X_val_eng = add_features(X_val)
            
            imputer = KNNImputer(n_neighbors=5)
            X_train_imp = imputer.fit_transform(X_train_eng)
            X_val_imp = imputer.transform(X_val_eng)
            
            model = xgb.XGBClassifier(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=3,
                eval_metric='logloss',
                use_label_encoder=False,
                monotone_constraints=monotonic_tuple,
                n_jobs=1,
                random_state=42
            )
            
            model.fit(X_train_imp, y_train)
            y_prob = model.predict_proba(X_val_imp)[:, 1]
            
            aucs.append(roc_auc_score(y_val, y_prob))
            briers.append(brier_score_loss(y_val, y_prob))
            all_y_true.extend(y_val)
            all_y_prob.extend(y_prob)
            
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        mean_brier = np.mean(briers)
        
        youden = dca_utils.get_youden_metrics(all_y_true, all_y_prob)
        
        print(f"  AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
        
        results.append({
            'Experiment': name,
            'AUC_Mean': mean_auc,
            'AUC_Std': std_auc,
            'Brier': mean_brier,
            'Sensitivity': youden['sensitivity'],
            'Specificity': youden['specificity']
        })
        
        # Plot DCA
        fig, ax = plt.subplots(figsize=(8, 6))
        dca_utils.plot_dca(np.array(all_y_true), np.array(all_y_prob), name, ax=ax)
        plt.savefig(os.path.join(OUTPUT_DIR, f"dca_{name}.png"))
        plt.close()
        
    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, 'phase8_xgb_mono_results.csv'), index=False)
    print("Saved XGB Monotonic results.")

if __name__ == '__main__':
    run_experiment()
