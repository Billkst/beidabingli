
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
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, make_scorer

# Constants
DATA_PATH = '/home/UserData/ljx/beidabingli/队列符合277.xlsx'

# Feature sets (Same as 23_train_deployment_models.py)
core_vars = ['age', 'gender', 'baseline GFR', 'baseline UTP', 'MAP', 'Alb', 'RASB', '尿酸']
mest_vars = ['M', 'E', 'S', 'T', 'C']
high_missing_vars = ['前驱感染', '肉眼血尿', 'IgA', 'C3', '血尿（RBC）', 'Hb']

full_base_features = list(set(core_vars + mest_vars + high_missing_vars))
clinical_base_features = list(set(core_vars + high_missing_vars))

def load_data():
    _df = pd.read_excel(DATA_PATH)
    DROP_COLS = ['Unnamed: 26', 'number', 'Biopsydate', '病理扫片']
    df = _df.drop(columns=[c for c in DROP_COLS if c in _df.columns])
    data = df.dropna(subset=['label1', 'label2']).copy()
    return data

def add_features(X_in):
    X = X_in.copy()
    # Missing indicators
    for col in high_missing_vars:
        if col in X.columns:
            X[f'{col}_missing'] = X[col].isna().astype(int)
    # Ratio
    if 'IgA' in X.columns and 'C3' in X.columns:
        X['IgA_C3_ratio'] = X['IgA'] / X['C3'].replace(0, np.nan)
    # Interaction (only if S exists)
    if 'baseline UTP' in X.columns and 'S' in X.columns:
        X['UTP_x_S'] = X['baseline UTP'] * X['S']
    
    X = pd.get_dummies(X, columns=['gender'] if 'gender' in X.columns else [], drop_first=True)
    return X

def get_short_term_model():
    # Voting Ensemble
    estimators = [
        ('xgb', xgb.XGBClassifier(eval_metric='logloss', random_state=42, use_label_encoder=False)),
        ('lr', LogisticRegression(random_state=42, max_iter=1000)),
        ('svm', SVC(probability=True, random_state=42)),
        ('rf', RandomForestClassifier(random_state=42))
    ]
    pipeline = Pipeline([
        ('imputer', IterativeImputer(max_iter=10, random_state=42)),
        ('scaler', StandardScaler()), 
        ('voting', VotingClassifier(estimators=estimators, voting='soft'))
    ])
    return pipeline

def get_long_term_model():
    # SVM Pipeline
    pipeline = Pipeline([
        ('imputer', IterativeImputer(max_iter=10, random_state=42)),
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, kernel='rbf', C=1.0, random_state=42))
    ])
    return pipeline

def evaluate(model, X, y, task_name):
    # Repeated 5-Fold CV (5 repeats)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', error_score='raise')
    print(f"[{task_name}] AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")
    return scores.mean(), scores.std()

if __name__ == "__main__":
    data = load_data()
    
    # ---------------------------
    # Task 1: Label1 (Short Term)
    # ---------------------------
    print("--- Short Term (Label 1) ---")
    y1 = data['label1'].astype(int)
    
    # Clinical Only
    X1_clin = add_features(data[clinical_base_features])
    auc1_clin, _ = evaluate(get_short_term_model(), X1_clin, y1, "Clinical Only")
    
    # Full (Clinical + MEST)
    X1_full = add_features(data[full_base_features])
    auc1_full, _ = evaluate(get_short_term_model(), X1_full, y1, "Full (Clin+MEST)")
    
    # ---------------------------
    # Task 2: Label2 (Long Term)
    # ---------------------------
    print("\n--- Long Term (Label 2) ---")
    y2 = data['label2'].astype(int)
    
    # Clinical Only
    X2_clin = add_features(data[clinical_base_features])
    auc2_clin, _ = evaluate(get_long_term_model(), X2_clin, y2, "Clinical Only")
    
    # Full (Clinical + MEST)
    X2_full = add_features(data[full_base_features])
    auc2_full, _ = evaluate(get_long_term_model(), X2_full, y2, "Full (Clin+MEST)")

    print("\nDone.")
