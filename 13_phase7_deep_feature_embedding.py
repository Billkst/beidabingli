import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import random
import copy

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

DATA_PATH = '/home/UserData/ljx/beidabingli/队列符合277.xlsx'
OUTPUT_DIR = '/home/UserData/ljx/beidabingli/results_phase7_deep_learning'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# 1. Data Prep
# ------------------------------------------------------------------------------
def load_data():
    _df = pd.read_excel(DATA_PATH)
    DROP_COLS = ['Unnamed: 26', 'number', 'Biopsydate', '病理扫片']
    df = _df.drop(columns=[c for c in DROP_COLS if c in _df.columns])
    return df

core_vars = ['age', 'gender', 'baseline GFR', 'baseline UTP', 'MAP', 'Alb', 'RASB', '尿酸']
mest_vars = ['M', 'E', 'S', 'T', 'C']
high_missing_vars = ['前驱感染', '肉眼血尿', 'IgA', 'C3', '血尿（RBC）', 'Hb']

# Simple cleaning for NN
def preprocess_for_nn(df, label_col):
    # Only use basic features + high missing ones (raw values)
    feature_cols = core_vars + mest_vars + high_missing_vars
    # Use all rows for self-supervised training, even if label is missing
    X_raw = df[feature_cols].copy()
    
    # Impute
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X_raw)
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    return X_scaled, df[label_col].values

# ------------------------------------------------------------------------------
# 2. Denoising AutoEncoder (DAE) Model
# ------------------------------------------------------------------------------
class DAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=64):
        super(DAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU() # Latent representation
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

def swap_noise(x, swap_prob=0.15):
    # Applies swap noise to tensor x: for each cell, with prob p, swap with random value from same column
    n_samples, n_features = x.shape
    x_noisy = x.clone()
    for i in range(n_features):
        mask = torch.rand(n_samples) < swap_prob
        # Randomly sample from the column itself
        random_indices = torch.randperm(n_samples)
        x_noisy[mask, i] = x[random_indices][mask, i]
    return x_noisy

def train_dae(X, epochs=50, batch_size=32, lr=0.001):
    input_dim = X.shape[1]
    model = DAE(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    X_tensor = torch.FloatTensor(X)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        for batch in loader:
            batch_x = batch[0]
            # Add noise
            noisy_x = swap_noise(batch_x)
            
            optimizer.zero_grad()
            reconstructed, _ = model(noisy_x)
            loss = criterion(reconstructed, batch_x) # Try to reconstruct clean input
            loss.backward()
            optimizer.step()
            
    return model

def get_latent_features(model, X):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        _, latent = model(X_tensor)
    return latent.numpy()

# ------------------------------------------------------------------------------
# 3. Hybrid Modeling Loop
# ------------------------------------------------------------------------------
def run_phase7():
    df = load_data()
    
    for label in ['label1', 'label2']:
        print(f"\nProcessing {label}...")
        
        # 1. Preprocssing
        X_all, y_all = preprocess_for_nn(df, label)
        
        # 2. Train DAE on ALL data (Unsupervised) - Self-Supervised Learning
        print("  Training Denoising AutoEncoder (Self-Supervised)...")
        dae_model = train_dae(X_all, epochs=100) # Quick training
        
        # 3. Extract Features
        X_latent = get_latent_features(dae_model, X_all)
        print(f"  Extracted Latent Features shape: {X_latent.shape}")
        
        # 4. Concatenate: Original + Deep Features
        # X_final = np.hstack([X_all, X_latent]) 
        # Actually, let's try purely Deep Features vs Hybrid
        
        # Filter rows where target exists
        valid_mask = ~np.isnan(y_all)
        X_train_full = X_all[valid_mask]
        X_latent_full = X_latent[valid_mask]
        y_train_full = y_all[valid_mask]
        
        X_hybrid = np.hstack([X_train_full, X_latent_full])
        
        # 5. CV Evaluation with XGBoost
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # XGB Params (General Robust Params)
        params = {
            'n_estimators': 300, 'learning_rate': 0.03, 'max_depth': 3,
            'subsample': 0.8, 'colsample_bytree': 0.7, 'use_label_encoder': False, 'eval_metric': 'logloss'
        }
        
        # Experiment A: Hybrid (Original + DAE)
        aucs_hybrid = []
        for train_idx, val_idx in cv.split(X_hybrid, y_train_full):
            model = xgb.XGBClassifier(**params, n_jobs=1)
            model.fit(X_hybrid[train_idx], y_train_full[train_idx])
            probs = model.predict_proba(X_hybrid[val_idx])[:, 1]
            aucs_hybrid.append(roc_auc_score(y_train_full[val_idx], probs))
            
        print(f"  [Hybrid Features] XGB AUC: {np.mean(aucs_hybrid):.4f} +/- {np.std(aucs_hybrid):.4f}")
        
        # Experiment B: Baseline (Just Original)
        aucs_base = []
        for train_idx, val_idx in cv.split(X_train_full, y_train_full):
            model = xgb.XGBClassifier(**params, n_jobs=1)
            model.fit(X_train_full[train_idx], y_train_full[train_idx])
            probs = model.predict_proba(X_train_full[val_idx])[:, 1]
            aucs_base.append(roc_auc_score(y_train_full[val_idx], probs))
            
        print(f"  [Baseline Features] XGB AUC: {np.mean(aucs_base):.4f} +/- {np.std(aucs_base):.4f}")

if __name__ == "__main__":
    run_phase7()
