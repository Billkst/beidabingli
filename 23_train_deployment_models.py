import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Paths
DATA_PATH = '/home/UserData/ljx/beidabingli/队列符合277.xlsx'
MODEL_DIR = '/home/UserData/ljx/beidabingli/models/final_deployment'
os.makedirs(MODEL_DIR, exist_ok=True)

# Feature sets
core_vars = ['age', 'gender', 'baseline GFR', 'baseline UTP', 'MAP', 'Alb', 'RASB', '尿酸']
mest_vars = ['M', 'E', 'S', 'T', 'C']
high_missing_vars = ['前驱感染', '肉眼血尿', 'IgA', 'C3', '血尿（RBC）', 'Hb']
all_features = list(set(core_vars + mest_vars + high_missing_vars))

def load_data():
    _df = pd.read_excel(DATA_PATH)
    DROP_COLS = ['Unnamed: 26', 'number', 'Biopsydate', '病理扫片']
    df = _df.drop(columns=[c for c in DROP_COLS if c in _df.columns])
    data = df.dropna(subset=['label1', 'label2']).copy()
    
    # Trajectory Label
    # 0: Resistant (0,0)
    # 1: Late Response (0,1)
    # 2: Relapse (1,0)
    # 3: Sustained (1,1)
    def get_traj(row):
        l1 = int(row['label1'])
        l2 = int(row['label2'])
        if l1 == 0 and l2 == 0: return 0
        if l1 == 0 and l2 == 1: return 1
        if l1 == 1 and l2 == 0: return 2
        if l1 == 1 and l2 == 1: return 3
        return -1
        
    data['traj'] = data.apply(get_traj, axis=1)
    return data

def add_features(X_in):
    X = X_in.copy()
    for col in high_missing_vars:
        if col in X.columns:
            X[f'{col}_missing'] = X[col].isna().astype(int)
    if 'IgA' in X.columns and 'C3' in X.columns:
        X['IgA_C3_ratio'] = X['IgA'] / X['C3'].replace(0, np.nan)
    if 'baseline UTP' in X.columns and 'S' in X.columns:
        X['UTP_x_S'] = X['baseline UTP'] * X['S']
    
    X = pd.get_dummies(X, columns=['gender'] if 'gender' in X.columns else [], drop_first=True)
    return X

def train_trajectory_model(data):
    print("Training Output 1: Trajectory Model (Innovation)...")
    X = data[all_features]
    y = data['traj']
    
    X_eng = add_features(X)
    
    # Pipeline: MICE -> XGB
    # Using sklearn Pipeline validation
    pipeline = Pipeline([
        ('imputer', IterativeImputer(max_iter=10, random_state=42)),
        ('clf', xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=4,
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=1,
            eval_metric='mlogloss'
        ))
    ])
    
    pipeline.fit(X_eng, y)
    
    path = os.path.join(MODEL_DIR, 'Trajectory_XGB_Pipeline.joblib')
    joblib.dump(pipeline, path)
    print(f"  Saved to {path}")

def train_voting_ensemble(data, label_col='label1'):
    print(f"Training Output 2: Voting Ensemble for {label_col} (Robust)...")
    X = data[all_features] # Use all features for simplicity
    y = data[label_col].astype(int)
    
    X_eng = add_features(X)
    
    # Base learners
    estimators = [
        ('xgb', xgb.XGBClassifier(eval_metric='logloss', random_state=42, use_label_encoder=False)),
        ('lr', LogisticRegression(random_state=42, max_iter=1000)),
        ('svm', SVC(probability=True, random_state=42)),
        ('rf', RandomForestClassifier(random_state=42))
    ]
    
    # Voting Pipeline
    pipeline = Pipeline([
        ('imputer', IterativeImputer(max_iter=10, random_state=42)),
        ('scaler', StandardScaler()), # Important for LR/SVM
        ('voting', VotingClassifier(estimators=estimators, voting='soft'))
    ])
    
    pipeline.fit(X_eng, y)
    
    path = os.path.join(MODEL_DIR, f'Voting_Ensemble_{label_col}.joblib')
    joblib.dump(pipeline, path)
    print(f"  Saved to {path}")

def train_svm_longterm(data):
    print("Training Output 3: SVM Pipeline for LongTerm (High Sensitivity)...")
    X = data[all_features]
    y = data['label2'].astype(int)
    
    X_eng = add_features(X)
    
    pipeline = Pipeline([
        ('imputer', IterativeImputer(max_iter=10, random_state=42)),
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, kernel='rbf', C=1.0, random_state=42))
    ])
    
    pipeline.fit(X_eng, y)
    
    path = os.path.join(MODEL_DIR, 'SVM_MICE_Pipeline_LongTerm.joblib')
    joblib.dump(pipeline, path)
    print(f"  Saved to {path}")

if __name__ == "__main__":
    data = load_data()
    
    # 1. Trajectory Model (Phase 9 SOTA)
    train_trajectory_model(data)
    
    # 2. Voting Ensemble Short Term (Phase 8/6 Recommended)
    train_voting_ensemble(data, label_col='label1')
    
    # 3. SVM Long Term (Phase 8 Recommended)
    train_svm_longterm(data)
