"""
Data Preprocessor - Cleaning, encoding, scaling, and splitting.
"""

import pickle
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


class DataPreprocessor:
    """
    Preprocessing pipeline for credit scoring data.
    """
    
    def __init__(
        self,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        target: str = "default",
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42
    ):
        """test_size + val_size = 0.30 => train 70% per dissertation §3.3.3."""
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.target = target
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.label_encoders = {}
        self.feature_columns_ = []
    
    def _infer_features(self, df: pd.DataFrame) -> None:
        """Infer feature columns from dataframe."""
        exclude = {self.target, "customer_id", "class", "id", "ID", "default.payment.next.month"}
        all_cols = [c for c in df.columns if c not in exclude]
        
        if self.numerical_features is None:
            self.numerical_features = [
                c for c in all_cols
                if df[c].dtype in ["int64", "float64", "int32", "float32"]
            ]
        if self.categorical_features is None:
            self.categorical_features = [
                c for c in all_cols
                if c not in self.numerical_features
            ]
        
        self.feature_columns_ = self.numerical_features + self.categorical_features
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Fit preprocessor and transform data."""
        self._infer_features(df)
        
        X = df[self.feature_columns_].copy()
        y = df[self.target].values if self.target in df.columns else None
        
        # Encode categoricals
        for col in self.categorical_features:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str).fillna("missing"))
                self.label_encoders[col] = le
        
        # Fill missing and scale numerical
        X_num = X[self.numerical_features].values
        X_num = self.imputer.fit_transform(X_num)
        X_num = self.scaler.fit_transform(X_num)
        
        X[self.numerical_features] = X_num
        
        # Numerical + encoded categorical columns feed all models
        X_final = self._feature_matrix(X)
        
        return X_final, y, df
    
    def _feature_matrix(self, X: pd.DataFrame) -> np.ndarray:
        """Stack scaled numerical features and encoded categoricals."""
        parts = [X[self.numerical_features].values.astype(np.float32)]
        if self.categorical_features:
            cat = X[self.categorical_features].astype(np.float32).values
            parts.append(cat)
        return np.hstack(parts).astype(np.float32)
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data."""
        X = df[self.feature_columns_].copy()
        
        for col in self.categorical_features:
            if col in X.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                X[col] = X[col].astype(str).fillna("missing")
                unknown = set(X[col]) - set(le.classes_)
                for u in unknown:
                    X.loc[X[col] == u, col] = "missing"
                if "missing" not in le.classes_:
                    X.loc[~X[col].isin(le.classes_), col] = le.classes_[0]
                X[col] = le.transform(X[col])
        
        X_num = X[self.numerical_features].values
        X_num = self.imputer.transform(X_num)
        X_num = self.scaler.transform(X_num)
        X[self.numerical_features] = X_num
        
        return self._feature_matrix(X)
    
    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        stratify: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Train/val/test split: 70/15/15 per dissertation §3.3.3."""
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.test_size,
            stratify=y if stratify else None,
            random_state=self.random_state
        )
        val_ratio = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio,
            stratify=y_temp if stratify else None,
            random_state=self.random_state
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def split_indices(
        self,
        n: int,
        y: np.ndarray,
        stratify: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return train, val, test indices for 70/15/15 split."""
        idx = np.arange(n)
        temp_idx, test_idx = train_test_split(
            idx, test_size=self.test_size, stratify=y if stratify else None,
            random_state=self.random_state
        )
        val_ratio = self.val_size / (1 - self.test_size)
        train_idx, val_idx = train_test_split(
            temp_idx, test_size=val_ratio,
            stratify=y[temp_idx] if stratify else None,
            random_state=self.random_state
        )
        return train_idx, val_idx, test_idx
    
    def save(self, path: str) -> None:
        """Save preprocessor artifacts."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "scaler": self.scaler,
                "imputer": self.imputer,
                "label_encoders": self.label_encoders,
                "numerical_features": self.numerical_features,
                "categorical_features": self.categorical_features,
                "feature_columns_": self.feature_columns_,
                "target": self.target,
                "test_size": self.test_size,
                "val_size": getattr(self, "val_size", 0.15),
                "random_state": self.random_state,
            }, f)
    
    @classmethod
    def load(cls, path: str) -> "DataPreprocessor":
        """Load preprocessor."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        p = cls()
        for k, v in data.items():
            setattr(p, k, v)
        return p
