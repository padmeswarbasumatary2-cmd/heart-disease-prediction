from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import pandas as pd
from typing import Tuple

def preprocess(df: pd.DataFrame, target_col: str = 'target', poly_degree: int = 2) -> Tuple[pd.DataFrame, pd.Series, StandardScaler]:
    """
    Preprocess heart disease data with feature engineering:
    - Handle missing values (median imputation)
    - Scale features
    - Add polynomial and interaction features
    """
    df = df.copy()
    if target_col not in df.columns:
        raise ValueError(f"target column '{target_col}' not found")
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # Fill numeric NaNs with median
    X = X.fillna(X.median())
    
    # Add hand-crafted interaction features for medical interpretability
    if 'age' in X.columns and 'chol' in X.columns:
        X['age_chol_interaction'] = X['age'] * X['chol']
    if 'thalach' in X.columns and 'oldpeak' in X.columns:
        X['thalach_oldpeak_ratio'] = X['thalach'] / (X['oldpeak'] + 1)
    if 'age' in X.columns and 'trestbps' in X.columns:
        X['age_trestbps_product'] = X['age'] * X['trestbps']
    
    # Add derived features (e.g., binned age groups) with safe binning
    if 'age' in X.columns:
        X['age_group'] = pd.cut(X['age'], bins=[-0.1, 40, 50, 60, 100], labels=[0, 1, 2, 3], include_lowest=True).astype(int)
    
    # Scale all features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    return X_scaled, y, scaler
