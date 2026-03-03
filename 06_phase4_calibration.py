import os
import ast
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb

DATA_PATH = '/home/UserData/ljx/beidabingli/队列符合277.xlsx'
PHASE3_PATH = '/home/UserData/ljx/beidabingli/results_phase3/phase3_max_optimization_results.csv'
OUTPUT_DIR = '/home/UserData/ljx/beidabingli/results_phase4'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
_df = pd.read_excel(DATA_PATH)
DROP_COLS = ['Unnamed: 26', 'number', 'Biopsydate', '病理扫片']
df = _df.drop(columns=[c for c in DROP_COLS if c in _df.columns])

core_vars = [
    'age', 'gender', 'baseline GFR', 'baseline UTP', 'MAP', 'Alb', 'RASB', '尿酸'
]

mest_vars = ['M', 'E', 'S', 'T', 'C']

high_missing_vars = ['前驱感染', '肉眼血尿', 'IgA', 'C3', '血尿（RBC）', 'Hb']

# Feature engineering

def add_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    for col in high_missing_vars:
        if col in X.columns:
            X[f'{col}_missing'] = X[col].isna().astype(int)

    if 'IgA' in X.columns and 'C3' in X.columns:
        X['IgA_C3_ratio'] = X['IgA'] / X['C3'].replace(0, np.nan)

    if 'baseline UTP' in X.columns and 'S' in X.columns:
        X['UTP_x_S'] = X['baseline UTP'] * X['S']

    if 'age' in X.columns:
        X['age_bin'] = pd.cut(X['age'], bins=[0, 30, 50, 120], labels=['<30', '30-50', '>50'])
    if 'baseline UTP' in X.columns:
        X['utp_bin'] = pd.cut(X['baseline UTP'], bins=[-np.inf, 1, 3, np.inf], labels=['<1', '1-3', '>3'])

    X = pd.get_dummies(X, columns=[c for c in ['age_bin', 'utp_bin'] if c in X.columns], dummy_na=True)

    return X


def evaluate(model, X, y, cv):
    aucs, briers = [], []
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]

        aucs.append(roc_auc_score(y_val, y_prob))
        briers.append(brier_score_loss(y_val, y_prob))

    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(briers))


# Load phase3 results and determine best model per experiment
phase3 = pd.read_csv(PHASE3_PATH)

experiments = [
    {'name': 'ShortTerm_Core', 'label': 'label1', 'features': core_vars},
    {'name': 'ShortTerm_Core+MEST', 'label': 'label1', 'features': core_vars + mest_vars},
    {'name': 'LongTerm_Core', 'label': 'label2', 'features': core_vars},
    {'name': 'LongTerm_Core+MEST', 'label': 'label2', 'features': core_vars + mest_vars},
]

for exp in experiments:
    exp['features'] = list(dict.fromkeys(exp['features'] + high_missing_vars))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for exp in experiments:
    name = exp['name']
    label = exp['label']
    feat_cols = exp['features']

    data = df.dropna(subset=[label]).copy()
    X = data[feat_cols]
    y = data[label]

    # feature pipeline
    fe = FunctionTransformer(lambda X: add_features(pd.DataFrame(X, columns=feat_cols)), validate=False)

    # locate best model from phase3
    row = phase3[phase3['Experiment'] == name].iloc[0]

    # determine best model by AUC
    aucs = {
        'LR': row['LR_AUC'],
        'SVM': row['SVM_AUC'],
        'XGB': row['XGB_AUC']
    }
    best_model_type = max(aucs, key=aucs.get)

    if best_model_type == 'LR':
        params = ast.literal_eval(row['LR_Params'])
        model = LogisticRegression(penalty='elasticnet', solver='saga', max_iter=8000, class_weight='balanced')
        pipe = Pipeline([
            ('fe', fe),
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        pipe.set_params(**params)

    elif best_model_type == 'SVM':
        params = ast.literal_eval(row['SVM_Params'])
        model = SVC(kernel='rbf', probability=True, class_weight='balanced')
        pipe = Pipeline([
            ('fe', fe),
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        pipe.set_params(**params)

    else:
        params = ast.literal_eval(row['XGB_Params'])
        model = xgb.XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1)
        pipe = Pipeline([
            ('fe', fe),
            ('imputer', KNNImputer(n_neighbors=5)),
            ('model', model)
        ])
        pipe.set_params(**params)

    # Uncalibrated performance
    base_auc, base_auc_std, base_brier = evaluate(pipe, X, y, cv)

    # Calibrated performance: sigmoid and isotonic
    calib_sigmoid = CalibratedClassifierCV(pipe, cv=cv, method='sigmoid')
    sig_auc, sig_auc_std, sig_brier = evaluate(calib_sigmoid, X, y, cv)

    calib_iso = CalibratedClassifierCV(pipe, cv=cv, method='isotonic')
    iso_auc, iso_auc_std, iso_brier = evaluate(calib_iso, X, y, cv)

    results.append({
        'Experiment': name,
        'Best_Model': best_model_type,
        'Base_AUC': base_auc,
        'Base_AUC_STD': base_auc_std,
        'Base_Brier': base_brier,
        'Sigmoid_AUC': sig_auc,
        'Sigmoid_AUC_STD': sig_auc_std,
        'Sigmoid_Brier': sig_brier,
        'Isotonic_AUC': iso_auc,
        'Isotonic_AUC_STD': iso_auc_std,
        'Isotonic_Brier': iso_brier
    })

    print(f"\n{name} (Best={best_model_type})")
    print(f"  Base:    AUC={base_auc:.4f} | Brier={base_brier:.4f}")
    print(f"  Sigmoid: AUC={sig_auc:.4f} | Brier={sig_brier:.4f}")
    print(f"  Isotonic:AUC={iso_auc:.4f} | Brier={iso_brier:.4f}")

results_df = pd.DataFrame(results)
results_path = os.path.join(OUTPUT_DIR, 'phase4_calibration_results.csv')
results_df.to_csv(results_path, index=False)
print(f"\nSaved results to {results_path}")
