"""
Baseline Models - Logistic Regression, Random Forest, XGBoost.
"""

import time
import joblib
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class BaselineModel(ABC):
    """Abstract baseline model interface."""
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> dict:
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pass
    
    def save_model(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
    
    def load_model(self, path: str) -> None:
        self.model = joblib.load(path)
    
    def get_feature_importance(self) -> np.ndarray:
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        if hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_).flatten()
        return np.array([])


class LogisticRegressionModel(BaselineModel):
    def __init__(self, **kwargs):
        self.model = LogisticRegression(class_weight='balanced', max_iter=2000, **kwargs)
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> dict:
        t0 = time.time()
        self.model.fit(X, y)
        elapsed = time.time() - t0
        pred = self.model.predict(X)
        return {"train_time": elapsed, "accuracy": accuracy_score(y, pred)}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class RandomForestModel(BaselineModel):
    def __init__(self, n_estimators=100, max_depth=10, **kwargs):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            class_weight='balanced', random_state=42, **kwargs
        )
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> dict:
        t0 = time.time()
        self.model.fit(X, y)
        return {"train_time": time.time() - t0}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class GradientBoostingModel(BaselineModel):
    def __init__(self, n_estimators=100, max_depth=6, **kwargs):
        self.model = XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            eval_metric='logloss', random_state=42, **kwargs
        )
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> dict:
        t0 = time.time()
        self.model.fit(X, y)
        return {"train_time": time.time() - t0}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]
