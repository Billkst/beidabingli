import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
import matplotlib.pyplot as plt

# Using the created dca_utils
import dca_utils

DATA_PATH = '/home/UserData/ljx/beidabingli/队列符合277.xlsx'
OUTPUT_DIR = '/home/UserData/ljx/beidabingli/results_phase8_catboost'
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
    # Missing indicators
    for col in high_missing_vars:
        if col in X.columns:
            X[f'{col}_missing'] = X[col].isna().astype(int)
            
    # Ratio IgA/C3
    if 'IgA' in X.columns and 'C3' in X.columns:
        X['IgA_C3_ratio'] = X['IgA'] / X['C3'].replace(0, np.nan)
        
    # Interaction UTP * S
    if 'baseline UTP' in X.columns and 'S' in X.columns:
        X['UTP_x_S'] = X['baseline UTP'] * X['S']
        
    return X

def run_experiment():
    df = load_data()
    
    experiments = [
        {'name': 'ShortTerm_CatBoost', 'label': 'label1', 'features': core_vars + mest_vars},
        {'name': 'LongTerm_CatBoost', 'label': 'label2', 'features': core_vars + mest_vars}
    ]
    
    # Extend features with high missing cols to allow feature engineering
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
        
        # Feature Engineering
        X_eng = add_features(X)
        
        # CatBoost handles NaNs naturally, but we added features manually too.
        # We need to identify categorical features for CatBoost
        # S, T, C, M, E are categorical? 
        # In this dataset they are numerically coded 0/1 or 0/1/2
        # CatBoost works well if we specify cat_features indices.
        # But 'add_features' returns a dataframe, so we can find names.
        
        # Potential categorical columns: 'gender', 'RASB', 'M','E','S','T','C', '_missing' cols
        # However, for simplicity and fair comparison with XGB (which treated them as numeric), 
        # we will let CatBoost treat them as auto or numeric unless strictly string.
        # Actually CatBoost is great with explicit cat features. 
        # But to be safe and consistent with previous phases where we treated them as numeric/ordinal, we proceed as is.
        
        aucs = []
        briers = []
        all_y_true = []
        all_y_prob = []
        
        for i, (train_idx, val_idx) in enumerate(cv.split(X_eng, y)):
            X_train, X_val = X_eng.iloc[train_idx], X_eng.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = CatBoostClassifier(
                iterations=500,
                learning_rate=0.03,
                depth=4,
                loss_function='Logloss',
                verbose=False,
                random_seed=42,
                allow_writing_files=False
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
        
        # Youden stats
        youden_metrics = dca_utils.get_youden_metrics(all_y_true, all_y_prob)
        
        print(f"  AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
        print(f"  Brier: {mean_brier:.4f}")
        print(f"  Sens: {youden_metrics['sensitivity']:.4f}, Spec: {youden_metrics['specificity']:.4f}")
        
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
        
    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, 'phase8_catboost_results.csv'), index=False)
    print("Saved CatBoost results.")

if __name__ == '__main__':
    run_experiment()
