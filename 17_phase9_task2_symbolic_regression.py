import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import joblib

try:
    from gplearn.genetic import SymbolicTransformer
    GPLEARN_AVAILABLE = True
except ImportError:
    GPLEARN_AVAILABLE = False
    print("gplearn is not installed. Please install it using `pip install gplearn`.")

# Monkey Patch for gplearn with recent scikit-learn
if GPLEARN_AVAILABLE:
    from sklearn.utils.validation import check_X_y, check_array
    from gplearn.genetic import BaseSymbolic

    # Check if _validate_data is missing (sklearn > 1.6)
    if not hasattr(BaseSymbolic, "_validate_data"):
        def _validate_data(self, X, y=None, reset=True, validate_separately=False, **check_params):
            if y is None:
                out = check_array(X, **check_params)
            else:
                out = check_X_y(X, y, **check_params)
            
            # Manually set n_features_in_ if reset is True
            if reset:
                if isinstance(out, tuple):
                    X_checked = out[0]
                else:
                    X_checked = out
                
                if hasattr(X_checked, "shape"):
                     self.n_features_in_ = X_checked.shape[1]
            return out
        
        # Patch both BaseSymbolic and SymbolicTransformer just in case
        BaseSymbolic._validate_data = _validate_data
        SymbolicTransformer._validate_data = _validate_data

DATA_PATH = '/home/UserData/ljx/beidabingli/队列符合277.xlsx'
RESULT_DIR = '/home/UserData/ljx/beidabingli/results_phase9'
os.makedirs(RESULT_DIR, exist_ok=True)

# Feature definitions
core_vars = ['age', 'gender', 'baseline GFR', 'baseline UTP', 'MAP', 'Alb', 'RASB', '尿酸']
mest_vars = ['M', 'E', 'S', 'T', 'C']
high_missing_vars = ['前驱感染', '肉眼血尿', 'IgA', 'C3', '血尿（RBC）', 'Hb']

def load_and_preprocess(label_col):
    _df = pd.read_excel(DATA_PATH)
    DROP_COLS = ['Unnamed: 26', 'number', 'Biopsydate', '病理扫片']
    df = _df.drop(columns=[c for c in DROP_COLS if c in _df.columns])
    
    if label_col == 'label1':
        feature_cols = core_vars + high_missing_vars
    else:
        feature_cols = core_vars + mest_vars + high_missing_vars
    
    data = df.dropna(subset=[label_col]).copy()
    X = data[feature_cols]
    y = data[label_col].astype(int)
    
    return X, y

def process_features_for_gp(X_raw):
    # Dummies for categorical
    X = pd.get_dummies(X_raw, columns=['gender'] if 'gender' in X_raw.columns else [], drop_first=True)
    # Handle others if any. MEST are usually 0/1 or ordinal.
    
    # Impute
    imputer = IterativeImputer(max_iter=10, random_state=42)
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    return X_imputed

def run_symbolic_experiment():
    if not GPLEARN_AVAILABLE:
        return

    results = []
    
    for label_col, task_name in [('label1', 'ShortTerm'), ('label2', 'LongTerm')]:
        print(f"Starting Symbolic Regression for {task_name}...")
        X_raw, y = load_and_preprocess(label_col)
        X = process_features_for_gp(X_raw)
        feature_names = X.columns.tolist()
        
        # We need to scale data for Logistic Regression convergence, 
        # but GP learns shape, scale matters less but helps with numeric stability (log of large numbers)
        # It's safer to scale.
        # However, for interpretation of formulas, scaling makes variables 'x0', 'x1' refer to scaled vals.
        # We will map them back later if possible, but for performance checking, scaling is fine.
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Setup GP
        # Generating 10 new features
        gp = SymbolicTransformer(
            generations=20, 
            population_size=1000,
            hall_of_fame=100, 
            n_components=10,
            function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv'],
            parsimony_coefficient=0.001,
            max_samples=0.9, 
            verbose=0,
            random_state=42,
            n_jobs=1
        )
        
        # 5-Fold CV
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        aucs_base = []
        aucs_gp = []
        
        # Also let's run a full fit to see the formulas later
        print("  Discovering formulas on full dataset (for inspection)...")
        gp.fit(X_scaled, y)
        print("  Top generated formulas:")
        for i, program in enumerate(gp._best_programs):
            if i >= 5: break
            print(f"    Feature {i+1}: {program}")
            
        print("  Validating improvement via CV...")
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Baseline: Logistic Regression on original features
            lr = LogisticRegression(max_iter=1000)
            lr.fit(X_train, y_train)
            base_pred = lr.predict_proba(X_val)[:, 1]
            base_auc = roc_auc_score(y_val, base_pred)
            aucs_base.append(base_auc)
            
            # GP Enhanced: Fit GP on train (to avoid leakage), transform, then LR
            # Re-instantiate GP to avoid using full-fit knowledge
            gp_fold = SymbolicTransformer(
                generations=10,  # Reduced for speed in CV
                population_size=1000,
                n_components=10,
                function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv'],
                parsimony_coefficient=0.001,
                max_samples=0.9, verbose=0, random_state=42, n_jobs=1
            )
            gp_fold.fit(X_train, y_train)
            
            X_train_new = gp_fold.transform(X_train)
            X_val_new = gp_fold.transform(X_val)
            
            # Combine
            X_train_comb = np.hstack([X_train, X_train_new])
            X_val_comb = np.hstack([X_val, X_val_new])
            
            lr_gp = LogisticRegression(max_iter=1000)
            lr_gp.fit(X_train_comb, y_train)
            gp_pred = lr_gp.predict_proba(X_val_comb)[:, 1]
            gp_auc = roc_auc_score(y_val, gp_pred)
            aucs_gp.append(gp_auc)
            
            print(f"    Fold {fold}: Base AUC={base_auc:.4f}, GP AUC={gp_auc:.4f}")
            
        mean_base = np.mean(aucs_base)
        mean_gp = np.mean(aucs_gp)
        print(f"  {task_name} Summary: Base={mean_base:.4f} -> GP={mean_gp:.4f}")
        
        results.append({
            'Task': task_name,
            'Algorithm': 'SymbolicRegression+LR',
            'Base_AUC': mean_base,
            'GP_AUC': mean_gp,
            'Improvement': mean_gp - mean_base
        })
        
    # Save
    pd.DataFrame(results).to_csv(os.path.join(RESULT_DIR, 'phase9_task2_symbolic_results.csv'), index=False)

if __name__ == "__main__":
    run_symbolic_experiment()
