import os
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import matplotlib.pyplot as plt

import dca_utils

DATA_PATH = '/home/UserData/ljx/beidabingli/队列符合277.xlsx'
OUTPUT_DIR = '/home/UserData/ljx/beidabingli/results_phase8_mice_impute'
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def run_experiment():
    df = load_data()
    experiments = [
        {'name': 'ShortTerm_MICE', 'label': 'label1'},
        {'name': 'LongTerm_MICE', 'label': 'label2'}
    ]
    feat_cols = list(dict.fromkeys(core_vars + mest_vars + high_missing_vars))
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    
    for exp in experiments:
        name_prefix = exp['name']
        label = exp['label']
        
        data = df.dropna(subset=[label]).copy()
        X = data[feat_cols]
        y = data[label]
        
        # MICE needs numeric data generally? 
        # IterativeImputer works on float metrics. 
        # Categoricals (Gender, MEST) need to be handled carefuly or just treated as numeric (0/1).
        # We assume numeric encoding here.
        
        models = {
            'XGB': xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42, n_jobs=1),
            'SVM': SVC(probability=True, kernel='rbf', class_weight='balanced', random_state=42),
            'LR': LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=5000, class_weight='balanced', random_state=42)
        }
        
        for m_name, model in models.items():
            full_name = f"{name_prefix}_{m_name}"
            print(f"Running {full_name}...")
            
            aucs, briers = [], []
            all_y_true, all_y_prob = [], []
            
            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Manual Pipeline
                # 1. Feature Engineering (creates NaNs or keeps them)
                X_train_eng = add_features(X_train)
                X_val_eng = add_features(X_val)
                
                # 2. MICE Imputation
                imputer = IterativeImputer(max_iter=10, random_state=42)
                X_train_imp = imputer.fit_transform(X_train_eng)
                X_val_imp = imputer.transform(X_val_eng)
                
                # 3. Scale (Only for SVM/LR)
                if m_name in ['SVM', 'LR']:
                    scaler = StandardScaler()
                    X_train_ready = scaler.fit_transform(X_train_imp)
                    X_val_ready = scaler.transform(X_val_imp)
                else:
                    X_train_ready = X_train_imp
                    X_val_ready = X_val_imp
                    
                model.fit(X_train_ready, y_train)
                y_prob = model.predict_proba(X_val_ready)[:, 1]
                
                aucs.append(roc_auc_score(y_val, y_prob))
                briers.append(brier_score_loss(y_val, y_prob))
                all_y_true.extend(y_val)
                all_y_prob.extend(y_prob)
                
            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)
            mean_brier = np.mean(briers)
            youden = dca_utils.get_youden_metrics(all_y_true, all_y_prob)
            
            print(f"  AUC: {mean_auc:.4f} +/- {std_auc:.4f} | Brier: {mean_brier:.4f}")
            
            results.append({
                'Experiment': full_name,
                'AUC_Mean': mean_auc,
                'AUC_Std': std_auc,
                'Brier': mean_brier,
                'Sensitivity': youden['sensitivity'],
                'Specificity': youden['specificity']
            })
            
            # Plot DCA
            fig, ax = plt.subplots(figsize=(8, 6))
            dca_utils.plot_dca(np.array(all_y_true), np.array(all_y_prob), full_name, ax=ax)
            plt.savefig(os.path.join(OUTPUT_DIR, f"dca_{full_name}.png"))
            plt.close()

    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, 'phase8_mice_impute_results.csv'), index=False)

if __name__ == '__main__':
    run_experiment()
