import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.pipeline import  make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline 
import matplotlib.pyplot as plt

import dca_utils

DATA_PATH = '/home/UserData/ljx/beidabingli/队列符合277.xlsx'
OUTPUT_DIR = '/home/UserData/ljx/beidabingli/results_phase8_class_imbalance'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    _df = pd.read_excel(DATA_PATH)
    DROP_COLS = ['Unnamed: 26', 'number', 'Biopsydate', '病理扫片']
    df = _df.drop(columns=[c for c in DROP_COLS if c in _df.columns])
    return df

core_vars = ['age', 'gender', 'baseline GFR', 'baseline UTP', 'MAP', 'Alb', 'RASB', '尿酸']
mest_vars = ['M', 'E', 'S', 'T', 'C']
high_missing_vars = ['前驱感染', '肉眼血尿', 'IgA', 'C3', '血尿（RBC）', 'Hb']
feat_cols = list(dict.fromkeys(core_vars + mest_vars + high_missing_vars))

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
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
        {'name': 'ShortTerm', 'label': 'label1'},
        {'name': 'LongTerm', 'label': 'label2'}
    ]
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    
    for exp in experiments:
        name = exp['name']
        label = exp['label']
        
        data = df.dropna(subset=[label]).copy()
        X = data[feat_cols]
        y = data[label]
        
        # Calculate positive ratio to determine weights
        pos_ratio = y.mean()
        neg_pos_ratio = (1 - pos_ratio) / pos_ratio
        
        print(f"\nTask {name} (Pos Ratio: {pos_ratio:.2f})")
        
        strategies = [
            {
                'strat': 'Balanced_Weight',
                'model': make_pipeline(FeatureEngineer(), KNNImputer(n_neighbors=5), 
                         xgb.XGBClassifier(scale_pos_weight=neg_pos_ratio, eval_metric='logloss', use_label_encoder=False, random_state=42, n_jobs=1))
            },
            {
                'strat': 'No_Weight',
                'model': make_pipeline(FeatureEngineer(), KNNImputer(n_neighbors=5), 
                         xgb.XGBClassifier(scale_pos_weight=1, eval_metric='logloss', use_label_encoder=False, random_state=42, n_jobs=1))
            },
            {
                'strat': 'Undersampling',
                # ImbPipeline allows sampling steps
                'model': ImbPipeline([
                    ('fe', FeatureEngineer()),
                    ('imputer', KNNImputer(n_neighbors=5)),
                    ('sampler', RandomUnderSampler(random_state=42)),
                    ('model', xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42, n_jobs=1))
                ])
            }
        ]
        
        for s in strategies:
            strat_name = s['strat']
            model = s['model']
            full_name = f"{name}_{strat_name}"
            
            aucs = []
            briers = []
            all_y_true = []
            all_y_prob = []
            
            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                y_prob = model.predict_proba(X_val)[:, 1]
                
                aucs.append(roc_auc_score(y_val, y_prob))
                briers.append(brier_score_loss(y_val, y_prob))
                all_y_true.extend(y_val)
                all_y_prob.extend(y_prob)
                
            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)
            mean_brier = np.mean(briers)
            youden = dca_utils.get_youden_metrics(all_y_true, all_y_prob)
            
            print(f"  {strat_name} | AUC: {mean_auc:.4f} | Brier: {mean_brier:.4f}")
            
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
            
    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, 'phase8_class_imbalance_results.csv'), index=False)

if __name__ == '__main__':
    run_experiment()
