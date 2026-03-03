import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

DATA_PATH = '/home/UserData/ljx/beidabingli/队列符合277.xlsx'
RESULT_DIR = '/home/UserData/ljx/beidabingli/results_phase9'
os.makedirs(RESULT_DIR, exist_ok=True)

# Feature definitions
core_vars = ['age', 'gender', 'baseline GFR', 'baseline UTP', 'MAP', 'Alb', 'RASB', '尿酸']
mest_vars = ['M', 'E', 'S', 'T', 'C']
high_missing_vars = ['前驱感染', '肉眼血尿', 'IgA', 'C3', '血尿（RBC）', 'Hb']
all_features = list(set(core_vars + mest_vars + high_missing_vars))

def load_data():
    _df = pd.read_excel(DATA_PATH)
    DROP_COLS = ['Unnamed: 26', 'number', 'Biopsydate', '病理扫片']
    df = _df.drop(columns=[c for c in DROP_COLS if c in _df.columns])
    
    # We need both labels to exist
    data = df.dropna(subset=['label1', 'label2']).copy()
    
    # Create Trajectory Label
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
    
    X = data[all_features]
    y = data['traj']
    
    return X, y, data

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

def run_trajectory_analysis():
    print("--- Starting Task 9.3: Clinical Trajectory Analysis ---")
    
    X_raw, y, df = load_data()
    print(f"Trajectory Distribution:\n{y.value_counts().sort_index()}")
    # 0: Resistant, 1: Late, 2: Relapse, 3: Sustained
    
    X_eng = add_features(X_raw)
    
    # Impute globally (for simplicity in this exploratory phase, though strict CV is better)
    # Using IterativeImputer
    imputer = IterativeImputer(max_iter=10, random_state=42)
    X_imp = pd.DataFrame(imputer.fit_transform(X_eng), columns=X_eng.columns)
    
    # Model: Multiclass XGBoost
    # We want probability of each class
    clf = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=4,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    
    acc_scores = []
    auc_l1 = []
    auc_l2 = []
    
    for i, (train_idx, val_idx) in enumerate(rskf.split(X_imp, y)):
        X_train, X_val = X_imp.iloc[train_idx], X_imp.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        y_prob = clf.predict_proba(X_val) # shape (N, 4)
        
        # Accuracy
        acc = accuracy_score(y_val, y_pred)
        acc_scores.append(acc)
        
        # Recover Binary Labels
        # L1=1 if Traj is 2 or 3
        # L2=1 if Traj is 1 or 3
        
        y_val_l1 = y_val.isin([2, 3]).astype(int)
        y_val_l2 = y_val.isin([1, 3]).astype(int)
        
        prob_l1 = y_prob[:, 2] + y_prob[:, 3]
        prob_l2 = y_prob[:, 1] + y_prob[:, 3]
        
        try:
            auc1 = roc_auc_score(y_val_l1, prob_l1)
            auc2 = roc_auc_score(y_val_l2, prob_l2)
        except:
            auc1 = 0.5
            auc2 = 0.5
            
        auc_l1.append(auc1)
        auc_l2.append(auc2)
        
    print(f"\nResults (Trajectory Model):")
    print(f"  Overall Accuracy: {np.mean(acc_scores):.4f}")
    print(f"  Recovered ShortTerm AUC: {np.mean(auc_l1):.4f} +/- {np.std(auc_l1):.4f}")
    print(f"  Recovered LongTerm AUC:  {np.mean(auc_l2):.4f} +/- {np.std(auc_l2):.4f}")
    
    # Save Feature Importance to see what drives Trajectory
    # Retrain on full
    clf.fit(X_imp, y)
    imp = pd.DataFrame({
        'Feature': X_imp.columns,
        'Importance': clf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    imp_path = os.path.join(RESULT_DIR, 'phase9_task3_trajectory_importance.csv')
    imp.to_csv(imp_path, index=False)
    print(f"Saved feature importance to {imp_path}")

if __name__ == "__main__":
    run_trajectory_analysis()
