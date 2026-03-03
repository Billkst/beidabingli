import os
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
import matplotlib.pyplot as plt

import dca_utils

DATA_PATH = '/home/UserData/ljx/beidabingli/队列符合277.xlsx'
OUTPUT_DIR = '/home/UserData/ljx/beidabingli/results_phase8_lightgbm'
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
    
    # Rename columns to replace special characters for LightGBM
    # LightGBM doesn't like brackets/parentheses in column names (e.g. "血尿（RBC）")
    new_cols = {}
    for c in X.columns:
        if '（' in c or '）' in c or '(' in c or ')' in c:
            new_name = c.replace('（', '_').replace('）', '_').replace('(', '_').replace(')', '_')
            new_cols[c] = new_name
    X = X.rename(columns=new_cols)
    return X

def run_experiment():
    df = load_data()
    
    experiments = [
        {'name': 'ShortTerm_LGBM', 'label': 'label1', 'features': core_vars + mest_vars},
        {'name': 'LongTerm_LGBM', 'label': 'label2', 'features': core_vars + mest_vars}
    ]
    
    for exp in experiments:
        exp['features'] = list(dict.fromkeys(exp['features'] + high_missing_vars))
        
    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for exp in experiments:
        name = exp['name']
        label = exp['label']
        feat_cols = exp['features']
        
        print(f"\nRunning {name}...")
        
        data = df.dropna(subset=[label]).copy()
        X = data[feat_cols]
        y = data[label]
        
        X_eng = add_features(X)
        
        aucs = []
        briers = []
        all_y_true = []
        all_y_prob = []
        
        for i, (train_idx, val_idx) in enumerate(cv.split(X_eng, y)):
            X_train, X_val = X_eng.iloc[train_idx], X_eng.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = LGBMClassifier(
                n_estimators=200,
                learning_rate=0.03,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=1,
                verbose=-1
            )
            
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_val)[:, 1]
            
            aucs.append(roc_auc_score(y_val, y_prob))
            briers.append(brier_score_loss(y_val, y_prob))
            all_y_true.extend(y_val)
            all_y_prob.extend(y_prob)
            
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        mean_brier = np.mean(briers)
        
        youden_metrics = dca_utils.get_youden_metrics(all_y_true, all_y_prob)
        
        print(f"  AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
        print(f"  Brier: {mean_brier:.4f}")
        
        results.append({
            'Experiment': name,
            'AUC_Mean': mean_auc,
            'AUC_Std': std_auc,
            'Brier': mean_brier,
            'Sensitivity': youden_metrics['sensitivity'],
            'Specificity': youden_metrics['specificity'],
            'Best_Threshold': youden_metrics['best_threshold']
        })
        
        # Plot DCA
        fig, ax = plt.subplots(figsize=(8, 6))
        dca_utils.plot_dca(np.array(all_y_true), np.array(all_y_prob), name, ax=ax)
        plt.savefig(os.path.join(OUTPUT_DIR, f"dca_{name}.png"))
        plt.close()
        
    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, 'phase8_lightgbm_results.csv'), index=False)
    print("Saved LightGBM results.")

if __name__ == '__main__':
    run_experiment()
