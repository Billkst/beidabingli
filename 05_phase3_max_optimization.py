import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb

DATA_PATH = '/home/UserData/ljx/beidabingli/队列符合277.xlsx'
OUTPUT_DIR = '/home/UserData/ljx/beidabingli/results_phase3'
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

    # Missing indicators
    for col in high_missing_vars:
        if col in X.columns:
            X[f'{col}_missing'] = X[col].isna().astype(int)

    # Ratio IgA/C3
    if 'IgA' in X.columns and 'C3' in X.columns:
        X['IgA_C3_ratio'] = X['IgA'] / X['C3'].replace(0, np.nan)

    # Interaction: UTP * S
    if 'baseline UTP' in X.columns and 'S' in X.columns:
        X['UTP_x_S'] = X['baseline UTP'] * X['S']

    # Binning for Age and UTP (one-hot)
    if 'age' in X.columns:
        X['age_bin'] = pd.cut(X['age'], bins=[0, 30, 50, 120], labels=['<30', '30-50', '>50'])
    if 'baseline UTP' in X.columns:
        X['utp_bin'] = pd.cut(X['baseline UTP'], bins=[-np.inf, 1, 3, np.inf], labels=['<1', '1-3', '>3'])

    # One-hot encode bins
    X = pd.get_dummies(X, columns=[c for c in ['age_bin', 'utp_bin'] if c in X.columns], dummy_na=True)

    return X


def evaluate_model(model, X, y, cv):
    aucs, briers = [], []
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_val)[:, 1]
        else:
            scores = model.decision_function(X_val)
            y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

        aucs.append(roc_auc_score(y_val, y_prob))
        briers.append(brier_score_loss(y_val, y_prob))

    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(briers))

experiments = [
    {'name': 'ShortTerm_Core', 'label': 'label1', 'features': core_vars},
    {'name': 'ShortTerm_Core+MEST', 'label': 'label1', 'features': core_vars + mest_vars},
    {'name': 'LongTerm_Core', 'label': 'label2', 'features': core_vars},
    {'name': 'LongTerm_Core+MEST', 'label': 'label2', 'features': core_vars + mest_vars},
]

# Extend features with high-missing vars
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

    # Feature pipeline
    fe = FunctionTransformer(lambda X: add_features(pd.DataFrame(X, columns=feat_cols)), validate=False)

    # LR pipeline
    lr = LogisticRegression(penalty='elasticnet', solver='saga', max_iter=8000, class_weight='balanced')
    lr_pipe = Pipeline([
        ('fe', fe),
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler()),
        ('model', lr)
    ])

    lr_param_dist = {
        'model__C': [0.0005, 0.001, 0.005, 0.01, 0.1, 1, 5, 10, 50],
        'model__l1_ratio': [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    }

    lr_search = RandomizedSearchCV(
        lr_pipe,
        param_distributions=lr_param_dist,
        n_iter=30,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        random_state=42
    )
    lr_search.fit(X, y)
    best_lr = lr_search.best_estimator_
    lr_auc, lr_auc_std, lr_brier = evaluate_model(best_lr, X, y, cv)

    # SVM pipeline
    svm = SVC(kernel='rbf', probability=True, class_weight='balanced')
    svm_pipe = Pipeline([
        ('fe', fe),
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler()),
        ('model', svm)
    ])

    svm_param_dist = {
        'model__C': [0.01, 0.1, 1, 5, 10, 50, 100],
        'model__gamma': ['scale', 1, 0.1, 0.01, 0.005, 0.001]
    }

    svm_search = RandomizedSearchCV(
        svm_pipe,
        param_distributions=svm_param_dist,
        n_iter=25,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        random_state=42
    )
    svm_search.fit(X, y)
    best_svm = svm_search.best_estimator_
    svm_auc, svm_auc_std, svm_brier = evaluate_model(best_svm, X, y, cv)

    # XGBoost pipeline
    xgb_clf = xgb.XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    xgb_pipe = Pipeline([
        ('fe', fe),
        ('imputer', KNNImputer(n_neighbors=5)),
        ('model', xgb_clf)
    ])

    xgb_param_dist = {
        'model__n_estimators': [100, 200, 300, 500],
        'model__max_depth': [2, 3, 4, 5],
        'model__learning_rate': [0.01, 0.03, 0.05, 0.1],
        'model__subsample': [0.6, 0.7, 0.8, 0.9],
        'model__colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        'model__min_child_weight': [1, 3, 5, 7],
        'model__reg_alpha': [0, 0.05, 0.1, 0.5],
        'model__reg_lambda': [1, 3, 5, 10]
    }

    xgb_search = RandomizedSearchCV(
        xgb_pipe,
        param_distributions=xgb_param_dist,
        n_iter=40,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        random_state=42
    )
    xgb_search.fit(X, y)
    best_xgb = xgb_search.best_estimator_
    xgb_auc, xgb_auc_std, xgb_brier = evaluate_model(best_xgb, X, y, cv)

    results.append({
        'Experiment': name,
        'LR_AUC': lr_auc,
        'LR_AUC_STD': lr_auc_std,
        'LR_Brier': lr_brier,
        'SVM_AUC': svm_auc,
        'SVM_AUC_STD': svm_auc_std,
        'SVM_Brier': svm_brier,
        'XGB_AUC': xgb_auc,
        'XGB_AUC_STD': xgb_auc_std,
        'XGB_Brier': xgb_brier,
        'LR_Params': lr_search.best_params_,
        'SVM_Params': svm_search.best_params_,
        'XGB_Params': xgb_search.best_params_
    })

    print(f"\n{name}")
    print(f"  LR  AUC: {lr_auc:.4f} (+/- {lr_auc_std:.4f}) | Brier: {lr_brier:.4f} | Params: {lr_search.best_params_}")
    print(f"  SVM AUC: {svm_auc:.4f} (+/- {svm_auc_std:.4f}) | Brier: {svm_brier:.4f} | Params: {svm_search.best_params_}")
    print(f"  XGB AUC: {xgb_auc:.4f} (+/- {xgb_auc_std:.4f}) | Brier: {xgb_brier:.4f} | Params: {xgb_search.best_params_}")

results_df = pd.DataFrame(results)
results_path = os.path.join(OUTPUT_DIR, 'phase3_max_optimization_results.csv')
results_df.to_csv(results_path, index=False)
print(f"\nSaved results to {results_path}")
