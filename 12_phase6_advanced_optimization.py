import os
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
import xgboost as xgb
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')

DATA_PATH = '/home/UserData/ljx/beidabingli/队列符合277.xlsx'
OUTPUT_DIR = '/home/UserData/ljx/beidabingli/results_phase6_ensemble'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# 1. Data Loading & Preprocessing
# ------------------------------------------------------------------------------
def load_and_preprocess_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data not found at {DATA_PATH}")
        
    _df = pd.read_excel(DATA_PATH)
    
    # Drop non-feature columns
    DROP_COLS = ['Unnamed: 26', 'number', 'Biopsydate', '病理扫片']
    df = _df.drop(columns=[c for c in DROP_COLS if c in _df.columns])
    
    return df

core_vars = [
    'age', 'gender', 'baseline GFR', 'baseline UTP', 'MAP', 'Alb', 'RASB', '尿酸'
]
mest_vars = ['M', 'E', 'S', 'T', 'C']
high_missing_vars = ['前驱感染', '肉眼血尿', 'IgA', 'C3', '血尿（RBC）', 'Hb']

def add_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    # Missing indicators
    for col in high_missing_vars:
        if col in X.columns:
            X[f'{col}_missing'] = X[col].isna().astype(int)
            
    # Ratio IgA/C3
    if 'IgA' in X.columns and 'C3' in X.columns:
        X['IgA_C3_ratio'] = X['IgA'] / X['C3'].replace(0, np.nan)
        
    # Interaction UTP * S
    if 'baseline UTP' in X.columns and 'S' in X.columns:
        X['UTP_x_S'] = X['baseline UTP'] * X['S']
        
    return X

# ------------------------------------------------------------------------------
# 2. Best Hyperparameters (From Phase 3)
# ------------------------------------------------------------------------------
# We hardcode the best params found in Phase 3 to initialize our base learners.

# Short Term (Label 1) Best Params
short_term_params = {
    'lr': {'C': 0.0005, 'l1_ratio': 0.0, 'penalty': 'elasticnet', 'solver': 'saga', 'max_iter': 5000},
    'svm': {'C': 1, 'gamma': 0.001, 'probability': True, 'kernel': 'rbf'},
    'xgb': {
        'n_estimators': 500, 'learning_rate': 0.01, 'max_depth': 2, 
        'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 3,
        'reg_alpha': 0.05, 'reg_lambda': 1, 'use_label_encoder': False, 'eval_metric': 'logloss'
    }
}

# Long Term (Label 2) Best Params
long_term_params = {
    'lr': {'C': 0.001, 'l1_ratio': 0.0, 'penalty': 'elasticnet', 'solver': 'saga', 'max_iter': 5000},
    'svm': {'C': 1, 'gamma': 0.01, 'probability': True, 'kernel': 'rbf'},
    'xgb': {
        'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 4, 
        'subsample': 0.6, 'colsample_bytree': 0.7, 'min_child_weight': 7,
        'reg_alpha': 0.05, 'reg_lambda': 5, 'use_label_encoder': False, 'eval_metric': 'logloss'
    }
}

def get_base_models(label_type='short'):
    params = short_term_params if label_type == 'short' else long_term_params
    
    lr = LogisticRegression(**params['lr'], random_state=42)
    svm = SVC(**params['svm'], random_state=42)
    xgb_clf = xgb.XGBClassifier(**params['xgb'], random_state=42, n_jobs=1)
    
    # Random Forest as an additional diversity source (standard robust params)
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=4, random_state=42)
    
    return [
        ('lr', lr),
        ('svm', svm), 
        ('xgb', xgb_clf),
        ('rf', rf)
    ]

# ------------------------------------------------------------------------------
# 3. Model Pipeline Wrapper
# ------------------------------------------------------------------------------
def get_pipeline(model):
    return Pipeline([
        ('features', FunctionTransformer(add_features)),
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler()),
        ('model', model)
    ])

# Helper because Voting/Stacking expect estimators, not pipelines generally, 
# but we need preprocessing inside. 
# Actually, proper way is: Pipeline(Preprocessing -> Ensemble(Models))
# But Ensemble models might need different preprocessing (e.g. Tree doesn't need scaling, SVM does).
# So we create a pipeline per estimator for the ensemble if we want them to be independent,
# OR we just preprocess globally. Global preprocessing (Impute + Scale) is fine for all these models.

def get_ensemble_models(label_type='short'):
    base_estimators = get_base_models(label_type)
    
    # 1. Voting Classifier (Soft)
    voting_clf = VotingClassifier(estimators=base_estimators, voting='soft')
    
    # 2. Stacking Classifier
    # Final estimator: LogisticRegression is standard and robust
    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(C=1.0, penalty='l2', solver='lbfgs'),
        cv=5
    )
    
    return {
        'Voting': voting_clf,
        'Stacking': stacking_clf
    }

# ------------------------------------------------------------------------------
# 4. Main Execution Loop
# ------------------------------------------------------------------------------
from sklearn.preprocessing import FunctionTransformer

def main():
    df = load_and_preprocess_data()
    
    experiments = [
        # Short Term: Using XGBoost Core best features (from Phase 5 conclusion)
        # Actually Phase 5 said "Short Term: GFR + MAP (XGB)". 
        # But let's stick to the full "Core" set because Ensembles benefit from diversity.
        {'name': 'ShortTerm_Ensemble', 'label': 'label1', 'features': core_vars, 'type': 'short'},
        
        # Long Term: Using Core + MEST (from Phase 2/3 conclusion)
        {'name': 'LongTerm_Ensemble', 'label': 'label2', 'features': core_vars + mest_vars, 'type': 'long'}
    ]
    
    results = []
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print(f"{'Experiment':<25} | {'Model':<10} | {'AUC (Mean)':<10} | {'AUC (Std)':<10} | {'Brier':<10}")
    print("-" * 80)
    
    for exp in experiments:
        name = exp['name']
        label = exp['label']
        feat_cols = exp['features'] + high_missing_vars # Ensure raw cols for feature eng exist
        label_type = exp['type']
        
        # Prepare Data
        data = df.dropna(subset=[label]).copy()
        X = data[feat_cols]
        y = data[label]
        
        # Get Ensembles
        ensembles = get_ensemble_models(label_type)
        
        # Also run the single best XGB for comparison
        best_xgb_params = short_term_params['xgb'] if label_type == 'short' else long_term_params['xgb']
        single_xgb = xgb.XGBClassifier(**best_xgb_params, random_state=42, n_jobs=1)
        ensembles['Single_XGB'] = single_xgb
        
        for model_name, model in ensembles.items():
            # Build full pipeline
            # Note: We must put preprocessing OUTSIDE the ensemble if we wrap the whole thing,
            # OR wrap each estimator.
            # Wrapping the whole ensemble in a pipeline is easier and correct here 
            # since all models (LR, SVM, XGB, RF) can handle Scaled data (Trees strictly don't need it but don't hurt).
            
            full_pipeline = Pipeline([
                ('features', FunctionTransformer(add_features)),
                ('imputer', KNNImputer(n_neighbors=5)),
                ('scaler', StandardScaler()),
                ('classifier', model)
            ])
            
            aucs = []
            briers = []
            
            # Manual CV loop to get metrics
            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                full_pipeline.fit(X_train, y_train)
                
                # Predict
                if hasattr(full_pipeline, 'predict_proba'):
                    y_prob = full_pipeline.predict_proba(X_val)[:, 1]
                else:
                    y_prob = full_pipeline.decision_function(X_val) # Shouldn't happen for these
                    
                aucs.append(roc_auc_score(y_val, y_prob))
                briers.append(brier_score_loss(y_val, y_prob))
            
            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)
            mean_brier = np.mean(briers)
            
            results.append({
                'Experiment': name,
                'Model': model_name,
                'AUC': mean_auc,
                'AUC_Std': std_auc,
                'Brier': mean_brier
            })
            
            print(f"{name:<25} | {model_name:<10} | {mean_auc:.4f}     | {std_auc:.4f}     | {mean_brier:.4f}")
            
            # Save the best ensemble model (retrained on full data)
            # We retrain on full data to save the final artifact
            full_pipeline.fit(X, y)
            model_filename = f"{OUTPUT_DIR}/{name}_{model_name}.joblib"
            joblib.dump(full_pipeline, model_filename)

    # Save Results CSV
    res_df = pd.DataFrame(results)
    res_df.to_csv(f"{OUTPUT_DIR}/phase6_ensemble_results.csv", index=False)
    print("\nDeep optimization complete. Results saved.")

if __name__ == "__main__":
    main()
