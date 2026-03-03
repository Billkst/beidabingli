import os
import ast
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
import xgboost as xgb

DATA_PATH = '/home/UserData/ljx/beidabingli/队列符合277.xlsx'
PHASE3_PATH = '/home/UserData/ljx/beidabingli/results_phase3/phase3_max_optimization_results.csv'
MODEL_DIR = '/home/UserData/ljx/beidabingli/models'

os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
_df = pd.read_excel(DATA_PATH)
DROP_COLS = ['Unnamed: 26', 'number', 'Biopsydate', '病理扫片']
df = _df.drop(columns=[c for c in DROP_COLS if c in _df.columns])

# Feature sets
core_vars = [
    'age', 'gender', 'baseline GFR', 'baseline UTP', 'MAP', 'Alb', 'RASB', '尿酸'
]

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

    if 'age' in X.columns:
        X['age_bin'] = pd.cut(X['age'], bins=[0, 30, 50, 120], labels=['<30', '30-50', '>50'])
    if 'baseline UTP' in X.columns:
        X['utp_bin'] = pd.cut(X['baseline UTP'], bins=[-np.inf, 1, 3, np.inf], labels=['<1', '1-3', '>3'])

    X = pd.get_dummies(X, columns=[c for c in ['age_bin', 'utp_bin'] if c in X.columns], dummy_na=True)

    return X


class FeatureEngineer:
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return add_features(pd.DataFrame(X, columns=self.columns))


# Load Phase 3 params
phase3 = pd.read_csv(PHASE3_PATH)

final_tasks = [
    {
        'name': 'ShortTerm_Core',
        'label': 'label1',
        'features': core_vars + high_missing_vars,
        'model_path': os.path.join(MODEL_DIR, 'short_term_xgb.joblib'),
        'meta_path': os.path.join(MODEL_DIR, 'short_term_xgb_meta.json')
    },
    {
        'name': 'LongTerm_Core+MEST',
        'label': 'label2',
        'features': core_vars + mest_vars + high_missing_vars,
        'model_path': os.path.join(MODEL_DIR, 'long_term_xgb.joblib'),
        'meta_path': os.path.join(MODEL_DIR, 'long_term_xgb_meta.json')
    }
]

for task in final_tasks:
    exp_row = phase3[phase3['Experiment'] == task['name']].iloc[0]
    xgb_params = ast.literal_eval(exp_row['XGB_Params'])

    data = df.dropna(subset=[task['label']]).copy()
    X = data[task['features']]
    y = data[task['label']]

    fe = FeatureEngineer(task['features'])

    model = xgb.XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline([
        ('fe', fe),
        ('imputer', KNNImputer(n_neighbors=5)),
        ('model', model)
    ])

    pipe.set_params(**xgb_params)
    pipe.fit(X, y)

    joblib.dump(pipe, task['model_path'])

    meta = {
        'experiment': task['name'],
        'label': task['label'],
        'features': task['features'],
        'xgb_params': xgb_params
    }
    with open(task['meta_path'], 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved model: {task['model_path']}")
    print(f"Saved meta: {task['meta_path']}")
