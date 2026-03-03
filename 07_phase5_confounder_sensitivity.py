import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.svm import SVC
import xgboost as xgb

DATA_PATH = '/home/UserData/ljx/beidabingli/队列符合277.xlsx'
OUTPUT_DIR = '/home/UserData/ljx/beidabingli/results_phase5'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
_df = pd.read_excel(DATA_PATH)
DROP_COLS = ['Unnamed: 26', 'number', 'Biopsydate', '病理扫片']
df = _df.drop(columns=[c for c in DROP_COLS if c in _df.columns])

# Base feature options
core_base = ['age', 'gender', 'baseline UTP', 'Alb', 'RASB', '尿酸']

# Alternative renal function and BP sets
renal_gfr = ['baseline GFR']
renal_scr = ['baseline Scr']

bp_map = ['MAP']
bp_sbp_dbp = ['SBP', 'DBP']

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

    # Age and UTP bins
    if 'age' in X.columns:
        X['age_bin'] = pd.cut(X['age'], bins=[0, 30, 50, 120], labels=['<30', '30-50', '>50'])
    if 'baseline UTP' in X.columns:
        X['utp_bin'] = pd.cut(X['baseline UTP'], bins=[-np.inf, 1, 3, np.inf], labels=['<1', '1-3', '>3'])

    X = pd.get_dummies(X, columns=[c for c in ['age_bin', 'utp_bin'] if c in X.columns], dummy_na=True)

    return X


def evaluate_model(model, X, y, cv):
    aucs, briers = [], []
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]

        aucs.append(roc_auc_score(y_val, y_prob))
        briers.append(brier_score_loss(y_val, y_prob))

    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(briers))


# Experiments: switch GFR vs Scr, MAP vs SBP/DBP
feature_sets = []

for renal in [renal_gfr, renal_scr]:
    for bp in [bp_map, bp_sbp_dbp]:
        name = f"Renal={'GFR' if renal==renal_gfr else 'Scr'}_BP={'MAP' if bp==bp_map else 'SBP+DBP'}"
        feature_sets.append((name, core_base + renal + bp))

# Task definitions
experiments = [
    {'name': 'ShortTerm', 'label': 'label1'},
    {'name': 'LongTerm', 'label': 'label2'}
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

for exp in experiments:
    label = exp['label']
    for fs_name, base_feats in feature_sets:
        # Core vs Core+MEST
        for use_mest in [False, True]:
            feats = base_feats + (mest_vars if use_mest else [])
            feats = list(dict.fromkeys(feats + high_missing_vars))

            data = df.dropna(subset=[label]).copy()
            X = data[feats]
            y = data[label]

            fe = FunctionTransformer(lambda X: add_features(pd.DataFrame(X, columns=feats)), validate=False)

            # Use the best-performing family from Phase3: XGBoost (general best) and SVM (long-term good)
            # For fair compare, run both and keep best.
            # XGB
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

            # Light randomized tuning for XGB
            xgb_param_dist = {
                'model__n_estimators': [100, 200, 300],
                'model__max_depth': [2, 3, 4],
                'model__learning_rate': [0.03, 0.05, 0.1],
                'model__subsample': [0.7, 0.9],
                'model__colsample_bytree': [0.7, 0.9],
                'model__min_child_weight': [1, 5],
                'model__reg_alpha': [0, 0.1],
                'model__reg_lambda': [1, 5]
            }

            xgb_search = RandomizedSearchCV(
                xgb_pipe,
                param_distributions=xgb_param_dist,
                n_iter=15,
                scoring='roc_auc',
                cv=cv,
                n_jobs=-1,
                random_state=42
            )
            xgb_search.fit(X, y)
            best_xgb = xgb_search.best_estimator_
            xgb_auc, xgb_auc_std, xgb_brier = evaluate_model(best_xgb, X, y, cv)

            # SVM
            svm = SVC(kernel='rbf', probability=True, class_weight='balanced')
            svm_pipe = Pipeline([
                ('fe', fe),
                ('imputer', KNNImputer(n_neighbors=5)),
                ('scaler', StandardScaler()),
                ('model', svm)
            ])

            svm_param_dist = {
                'model__C': [0.1, 1, 5, 10, 50],
                'model__gamma': ['scale', 0.1, 0.01]
            }

            svm_search = RandomizedSearchCV(
                svm_pipe,
                param_distributions=svm_param_dist,
                n_iter=10,
                scoring='roc_auc',
                cv=cv,
                n_jobs=-1,
                random_state=42
            )
            svm_search.fit(X, y)
            best_svm = svm_search.best_estimator_
            svm_auc, svm_auc_std, svm_brier = evaluate_model(best_svm, X, y, cv)

            # Choose best
            if xgb_auc >= svm_auc:
                best_model = 'XGB'
                best_auc, best_auc_std, best_brier = xgb_auc, xgb_auc_std, xgb_brier
            else:
                best_model = 'SVM'
                best_auc, best_auc_std, best_brier = svm_auc, svm_auc_std, svm_brier

            results.append({
                'Task': exp['name'],
                'Feature_Set': fs_name,
                'Use_MEST': use_mest,
                'Best_Model': best_model,
                'Best_AUC': best_auc,
                'Best_AUC_STD': best_auc_std,
                'Best_Brier': best_brier,
                'XGB_AUC': xgb_auc,
                'SVM_AUC': svm_auc
            })

            print(f"\n{exp['name']} | {fs_name} | MEST={use_mest}")
            print(f"  XGB AUC={xgb_auc:.4f} | SVM AUC={svm_auc:.4f} | Best={best_model} AUC={best_auc:.4f}")

results_df = pd.DataFrame(results)
results_path = os.path.join(OUTPUT_DIR, 'phase5_confounder_sensitivity.csv')
results_df.to_csv(results_path, index=False)
print(f"\nSaved results to {results_path}")
