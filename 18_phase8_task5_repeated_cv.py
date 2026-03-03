import os
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb
import matplotlib.pyplot as plt

import dca_utils

DATA_PATH = '/home/UserData/ljx/beidabingli/队列符合277.xlsx'
OUTPUT_DIR = '/home/UserData/ljx/beidabingli/results_phase8_repeated_cv'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    _df = pd.read_excel(DATA_PATH)
    DROP_COLS = ['Unnamed: 26', 'number', 'Biopsydate', '病理扫片']
    df = _df.drop(columns=[c for c in DROP_COLS if c in _df.columns])
    return df

core_vars = ['age', 'gender', 'baseline GFR', 'baseline UTP', 'MAP', 'Alb', 'RASB', '尿酸']
mest_vars = ['M', 'E', 'S', 'T', 'C']
high_missing_vars = ['前驱感染', '肉眼血尿', 'IgA', 'C3', '血尿（RBC）', 'Hb']

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

def get_voting_ensemble():
    # Estimators must solve imputation themselves or use pipeline
    fe = FeatureEngineer()
    
    pipe_lr = make_pipeline(fe, KNNImputer(n_neighbors=5), StandardScaler(), 
                           LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=3000, class_weight='balanced', random_state=42))
    
    pipe_svm = make_pipeline(fe, KNNImputer(n_neighbors=5), StandardScaler(),
                            SVC(probability=True, kernel='rbf', class_weight='balanced', random_state=42))
    
    pipe_xgb = make_pipeline(fe, KNNImputer(n_neighbors=5),
                            xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, eval_metric='logloss', use_label_encoder=False, random_state=42, n_jobs=1))
    
    pipe_rf = make_pipeline(fe, KNNImputer(n_neighbors=5),
                           RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=1))
    
    return VotingClassifier(estimators=[
        ('lr', pipe_lr), ('svm', pipe_svm), ('xgb', pipe_xgb), ('rf', pipe_rf)
    ], voting='soft')

def run_experiment():
    df = load_data()
    experiments = [
        {'name': 'ShortTerm', 'label': 'label1'},
        {'name': 'LongTerm', 'label': 'label2'}
    ]
    feat_cols = list(dict.fromkeys(core_vars + mest_vars + high_missing_vars))
    
    results = []
    
    # Repeated 5-Fold, 5 times = 25 runs
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    
    for exp in experiments:
        name_prefix = exp['name']
        label = exp['label']
        
        data = df.dropna(subset=[label]).copy()
        X = data[feat_cols]
        y = data[label]
        
        # Models: Single XGB vs Voting
        # Note: We reconstruct fresh models inside loop or clone, but Voting is complex.
        # Ideally fit/predict in loop.
        
        models_def = {
            'XGB_Single': make_pipeline(FeatureEngineer(), KNNImputer(), xgb.XGBClassifier(n_estimators=100, max_depth=3, eval_metric='logloss', use_label_encoder=False, random_state=42, n_jobs=1)),
            'Voting': get_voting_ensemble()
        }
        
        for m_name, model_obj in models_def.items():
            full_name = f"{name_prefix}_{m_name}"
            print(f"Running {full_name} (Repeated CV)...")
            
            aucs = []
            briers = []
            all_y_true = []
            all_y_prob = []
            
            for i, (train_idx, val_idx) in enumerate(rkf.split(X, y)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model_obj.fit(X_train, y_train)
                y_prob = model_obj.predict_proba(X_val)[:, 1]
                
                auc = roc_auc_score(y_val, y_prob)
                brier = brier_score_loss(y_val, y_prob)
                
                aucs.append(auc)
                briers.append(brier)
                all_y_true.extend(y_val)
                all_y_prob.extend(y_prob)
                
                if i % 5 == 0:
                    print(f"  Iter {i}: {auc:.3f}")
            
            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)
            mean_brier = np.mean(briers)
            youden = dca_utils.get_youden_metrics(all_y_true, all_y_prob)
            
            print(f"  Summary AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
            
            results.append({
                'Experiment': full_name,
                'AUC_Mean': mean_auc,
                'AUC_Std': std_auc,
                'Brier': mean_brier,
                'Sensitivity': youden['sensitivity'],
                'Specificity': youden['specificity']
            })
            
            # Plot DCA (Aggregated over all repeats - might smooth out noise)
            fig, ax = plt.subplots(figsize=(8, 6))
            dca_utils.plot_dca(np.array(all_y_true), np.array(all_y_prob), full_name, ax=ax)
            plt.savefig(os.path.join(OUTPUT_DIR, f"dca_{full_name}.png"))
            plt.close()

    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, 'phase8_repeated_cv_results.csv'), index=False)

if __name__ == '__main__':
    run_experiment()
