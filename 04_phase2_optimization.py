import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb

DATA_PATH = '/home/UserData/ljx/beidabingli/队列符合277.xlsx'
OUTPUT_DIR = '/home/UserData/ljx/beidabingli/results_phase2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load
_df = pd.read_excel(DATA_PATH)

# Drop non-feature columns
DROP_COLS = ['Unnamed: 26', 'number', 'Biopsydate', '病理扫片']
df = _df.drop(columns=[c for c in DROP_COLS if c in _df.columns])

# Base feature sets
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

    # Ratio IgA/C3 (avoid divide by zero)
    if 'IgA' in X.columns and 'C3' in X.columns:
        X['IgA_C3_ratio'] = X['IgA'] / X['C3'].replace(0, np.nan)

    # Interaction UTP * S
    if 'baseline UTP' in X.columns and 'S' in X.columns:
        X['UTP_x_S'] = X['baseline UTP'] * X['S']

    return X

# Evaluate function

def evaluate_model(name, model, X, y, cv):
    aucs, briers = [], []
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_val)[:, 1]
        else:
            # SVM with probability=True has predict_proba; fallback to decision_function
            scores = model.decision_function(X_val)
            y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

        aucs.append(roc_auc_score(y_val, y_prob))
        briers.append(brier_score_loss(y_val, y_prob))

    return np.mean(aucs), np.std(aucs), np.mean(briers)

# Prepare experiments
experiments = [
    {'name': 'ShortTerm_Core', 'label': 'label1', 'features': core_vars},
    {'name': 'ShortTerm_Core+MEST', 'label': 'label1', 'features': core_vars + mest_vars},
    {'name': 'LongTerm_Core', 'label': 'label2', 'features': core_vars},
    {'name': 'LongTerm_Core+MEST', 'label': 'label2', 'features': core_vars + mest_vars},
]

# Add extended features
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

    # Pipeline: Feature engineering -> KNN Impute -> Scale -> Model
    fe = FunctionTransformer(lambda X: add_features(pd.DataFrame(X, columns=feat_cols)), validate=False)

    base_steps = [
        ('fe', fe),
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler())
    ]

    # Logistic Regression Grid
    lr = LogisticRegression(penalty='elasticnet', solver='saga', max_iter=5000, class_weight='balanced')
    lr_pipe = Pipeline(base_steps + [('model', lr)])

    lr_param_grid = {
        'model__C': [0.001, 0.01, 0.1, 1, 5, 10, 50],
        'model__l1_ratio': [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    }

    lr_search = GridSearchCV(
        lr_pipe,
        param_grid=lr_param_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1
    )

    lr_search.fit(X, y)
    best_lr = lr_search.best_estimator_
    lr_auc, lr_auc_std, lr_brier = evaluate_model('LR', best_lr, X, y, cv)

    # SVM Grid
    svm = SVC(kernel='rbf', probability=True, class_weight='balanced')
    svm_pipe = Pipeline(base_steps + [('model', svm)])

    svm_param_grid = {
        'model__C': [0.01, 0.1, 1, 5, 10, 50],
        'model__gamma': ['scale', 1, 0.1, 0.01, 0.001]
    }

    svm_search = GridSearchCV(
        svm_pipe,
        param_grid=svm_param_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1
    )

    svm_search.fit(X, y)
    best_svm = svm_search.best_estimator_
    svm_auc, svm_auc_std, svm_brier = evaluate_model('SVM', best_svm, X, y, cv)

    # XGBoost Grid (compact but systematic)
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

    xgb_param_grid = {
        'model__n_estimators': [100, 300],
        'model__max_depth': [2, 3, 4],
        'model__learning_rate': [0.03, 0.05, 0.1],
        'model__subsample': [0.7, 0.9],
        'model__colsample_bytree': [0.7, 0.9],
        'model__min_child_weight': [1, 5],
        'model__reg_alpha': [0, 0.1],
        'model__reg_lambda': [1, 5]
    }

    xgb_search = GridSearchCV(
        xgb_pipe,
        param_grid=xgb_param_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1
    )

    xgb_search.fit(X, y)
    best_xgb = xgb_search.best_estimator_
    xgb_auc, xgb_auc_std, xgb_brier = evaluate_model('XGB', best_xgb, X, y, cv)

    results.append({
        'Experiment': name,
        'Best_LR_AUC': lr_auc,
        'Best_LR_AUC_STD': lr_auc_std,
        'Best_LR_Brier': lr_brier,
        'Best_SVM_AUC': svm_auc,
        'Best_SVM_AUC_STD': svm_auc_std,
        'Best_SVM_Brier': svm_brier,
        'Best_XGB_AUC': xgb_auc,
        'Best_XGB_AUC_STD': xgb_auc_std,
        'Best_XGB_Brier': xgb_brier,
        'LR_Params': lr_search.best_params_,
        'SVM_Params': svm_search.best_params_,
        'XGB_Params': xgb_search.best_params_
    })

    print(f"\n{name}")
    print(f"  LR AUC: {lr_auc:.4f} (+/- {lr_auc_std:.4f}) | Brier: {lr_brier:.4f} | Params: {lr_search.best_params_}")
    print(f"  SVM AUC: {svm_auc:.4f} (+/- {svm_auc_std:.4f}) | Brier: {svm_brier:.4f} | Params: {svm_search.best_params_}")
    print(f"  XGB AUC: {xgb_auc:.4f} (+/- {xgb_auc_std:.4f}) | Brier: {xgb_brier:.4f} | Params: {xgb_search.best_params_}")

# Save results
results_df = pd.DataFrame(results)
results_path = os.path.join(OUTPUT_DIR, 'phase2_optimization_results.csv')
results_df.to_csv(results_path, index=False)
print(f"\nSaved results to {results_path}")
