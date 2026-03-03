import os
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb

DATA_PATH = '/home/UserData/ljx/beidabingli/队列符合277.xlsx'
OUTPUT_DIR = '/home/UserData/ljx/beidabingli/results_phase9/error_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Feature sets
core_vars = ['age', 'gender', 'baseline GFR', 'baseline UTP', 'MAP', 'Alb', 'RASB', '尿酸']
mest_vars = ['M', 'E', 'S', 'T', 'C']
high_missing_vars = ['前驱感染', '肉眼血尿', 'IgA', 'C3', '血尿（RBC）', 'Hb']

def load_data(label_col):
    _df = pd.read_excel(DATA_PATH)
    DROP_COLS = ['Unnamed: 26', 'number', 'Biopsydate', '病理扫片']
    df = _df.drop(columns=[c for c in DROP_COLS if c in _df.columns])
    
    # Feature selection
    if label_col == 'label1':
        features = core_vars + high_missing_vars
    else:
        features = core_vars + mest_vars + high_missing_vars
        
    data = df.dropna(subset=[label_col]).copy()
    
    # Keep original index to track back patients
    data['Original_Index'] = data.index
    
    X = data[features]
    y = data[label_col].astype(int)
    ids = data['Original_Index']
    
    return X, y, ids, data

def add_features(X_in):
    X = X_in.copy()
    for col in high_missing_vars:
        if col in X.columns:
            X[f'{col}_missing'] = X[col].isna().astype(int)
    if 'IgA' in X.columns and 'C3' in X.columns:
        X['IgA_C3_ratio'] = X['IgA'] / X['C3'].replace(0, np.nan)
    if 'baseline UTP' in X.columns and 'S' in X.columns:
        X['UTP_x_S'] = X['baseline UTP'] * X['S']
        
    # Standard Dummy
    X = pd.get_dummies(X, columns=['gender'] if 'gender' in X.columns else [], drop_first=True)
    return X

def get_best_model(task_name):
    # Reconstructing the best models found in Phase 8/6
    
    if task_name == 'ShortTerm':
        # Best: Voting Ensemble (XGB, LR, SVM, RF)
        # Note: We construct a pipeline that includes Imputer inside for safety, 
        # or we impute globally for this analysis (CV leakage is minor for error profiling purpose, but let's be strict)
        # Actually, for sklearn Pipeline, we need transformers.
        
        # Simplified Voting for Error Analysis
        clf1 = xgb.XGBClassifier(eval_metric='logloss', random_state=42, use_label_encoder=False)
        clf2 = LogisticRegression(random_state=42, max_iter=1000)
        clf3 = SVC(probability=True, random_state=42)
        clf4 = RandomForestClassifier(random_state=42)
        
        voting = VotingClassifier(
            estimators=[('xgb', clf1), ('lr', clf2), ('svm', clf3), ('rf', clf4)],
            voting='soft'
        )
        return voting
        
    elif task_name == 'LongTerm':
        # Best: SVM (with MICE)
        return SVC(probability=True, random_state=42, kernel='rbf', C=1.0)

def analyze_errors(label_col, task_name):
    print(f"--- Analyzing {task_name} ({label_col}) ---")
    X_raw, y, ids, df_full = load_data(label_col)
    
    # 1. Pipeline preparation
    # We do manual preprocessing block to handle MICE/Scaling consistently
    X = add_features(X_raw)
    
    # Imputation & Scaling
    # To get OOF predictions, we use cross_val_predict
    # We need a custom estimator or pipeline to handle MICE inside CV
    
    # Let's do a strict 5-Fold Loop to get predictions and save them
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(y))
    oof_indices = []
    
    # Arrays to store
    # We need to map back to df_full index
    patient_predictions = pd.DataFrame({
        'Original_Index': ids.values,
        'True_Label': y.values,
        'Prob': 0.0,
        'Fold': -1
    }).set_index('Original_Index')
    
    print("Running Cross-Validation to generate profiles...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, _ = y.iloc[train_idx], y.iloc[val_idx]
        
        # MICE
        imputer = IterativeImputer(max_iter=10, random_state=42)
        X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
        X_val_imp = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)
        
        # Scale
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train_imp)
        X_val_sc = scaler.transform(X_val_imp)
        
        # Model
        model = get_best_model(task_name)
        model.fit(X_train_sc, y_train)
        
        # Predict
        probs = model.predict_proba(X_val_sc)[:, 1]
        
        # Store
        val_real_indices = ids.iloc[val_idx]
        patient_predictions.loc[val_real_indices, 'Prob'] = probs
        patient_predictions.loc[val_real_indices, 'Fold'] = fold
        
    # Analysis
    patient_predictions['Error'] = np.abs(patient_predictions['True_Label'] - patient_predictions['Prob'])
    # Define Hard Sample: Error > 0.6 (Highly confident wrong forecast)
    # or just Top 20% errors
    
    # Let's use quantile 
    hard_threshold = patient_predictions['Error'].quantile(0.8)
    print(f"Hard Sample Threshold (Top 20% Error): > {hard_threshold:.4f}")
    
    patient_predictions['Is_Hard'] = patient_predictions['Error'] > hard_threshold
    patient_predictions['Type'] = 'Correct'
    
    # Categorize
    # FN: True=1, Prob<0.5
    # FP: True=0, Prob>0.5
    mask_fn = (patient_predictions['True_Label'] == 1) & (patient_predictions['Prob'] < 0.5)
    mask_fp = (patient_predictions['True_Label'] == 0) & (patient_predictions['Prob'] > 0.5)
    
    patient_predictions.loc[mask_fn, 'Type'] = 'False Negative'
    patient_predictions.loc[mask_fp, 'Type'] = 'False Positive'
    
    hard_samples = patient_predictions[patient_predictions['Is_Hard']]
    easy_samples = patient_predictions[~patient_predictions['Is_Hard']]
    
    print(f"Identified {len(hard_samples)} Hard Samples vs {len(easy_samples)} Easy Samples")
    
    # 2. Compare Features
    # Join back with original raw features (before imputation, to see missingness patterns)
    # and also imputed features (to see value patterns)
    
    # We use df_full (original raw data) for comparison
    analysis_df = df_full.loc[ids].copy()
    analysis_df['Is_Hard'] = patient_predictions['Is_Hard']
    analysis_df['Error_Type'] = patient_predictions['Type']
    
    feature_report = []
    
    # Analyze numeric features
    numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove artificial or ID cols
    ignore = ['Is_Hard', 'Original_Index', 'label1', 'label2']
    numeric_cols = [c for c in numeric_cols if c not in ignore]
    
    for col in numeric_cols:
        # Group stats
        hard_vals = analysis_df[analysis_df['Is_Hard'] == True][col].dropna()
        easy_vals = analysis_df[analysis_df['Is_Hard'] == False][col].dropna()
        
        if len(hard_vals) > 0 and len(easy_vals) > 0:
            mean_hard = hard_vals.mean()
            mean_easy = easy_vals.mean()
            
            # T-test
            t_stat, p_val = stats.ttest_ind(hard_vals, easy_vals, equal_var=False)
            
            # Missing rate comparison
            miss_hard = analysis_df[analysis_df['Is_Hard'] == True][col].isna().mean()
            miss_easy = analysis_df[analysis_df['Is_Hard'] == False][col].isna().mean()
            
            feature_report.append({
                'Feature': col,
                'Hard_Mean': mean_hard,
                'Easy_Mean': mean_easy,
                'Diff_Pct': (mean_hard - mean_easy) / (abs(mean_easy) + 1e-6) * 100,
                'P_Value': p_val,
                'Hard_MissingRate': miss_hard,
                'Easy_MissingRate': miss_easy
            })
            
    # Save Report
    report_df = pd.DataFrame(feature_report).sort_values('P_Value')
    report_path = os.path.join(OUTPUT_DIR, f'{task_name}_feature_diff.csv')
    report_df.to_csv(report_path, index=False)
    
    # Significant findings
    sig_features = report_df[report_df['P_Value'] < 0.05]
    print(f"Found {len(sig_features)} variables with significant difference (p<0.05) between Hard and Easy samples.")
    if len(sig_features) > 0:
        print(sig_features[['Feature', 'Hard_Mean', 'Easy_Mean', 'P_Value']].head())

    # Save detailed predictions
    patient_predictions.to_csv(os.path.join(OUTPUT_DIR, f'{task_name}_predictions.csv'))

if __name__ == "__main__":
    analyze_errors('label1', 'ShortTerm')
    analyze_errors('label2', 'LongTerm')
