"""
LoRA Training Pipeline - Training loop with early stopping.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score


class LoRATrainer:
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 2e-5,
        device: str = "cpu",
        patience: int = 5,
        class_weights: tuple = None,
    ):
        """class_weights: (w0, w1) inversely proportional to class freq per dissertation §3.6."""
        self.model = model.to(device)
        self.device = device
        if hasattr(model, "trainable_parameter_groups"):
            groups = model.trainable_parameter_groups()
            self.optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
        else:
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        if class_weights is not None:
            w = torch.FloatTensor(class_weights)
            self.criterion = nn.CrossEntropyLoss(weight=w.to(device))
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.patience = patience
        self.best_auc = 0
        self.best_state = None
        self.history = {"train_loss": [], "val_auc": [], "val_f1": []}
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0
        for Xb, yb in tqdm(train_loader, desc="Train", leave=False):
            Xb, yb = Xb.to(self.device), yb.to(self.device).long()
            self.optimizer.zero_grad()
            logits = self.model(Xb)
            loss = self.criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> tuple:
        self.model.eval()
        all_preds, all_probs, all_labels = [], [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(self.device)
                logits = self.model(Xb)
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = (probs > 0.5).long()
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(yb.numpy())
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        return auc, f1
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 15,
        batch_size: int = 32
    ) -> dict:
        train_ds = TensorDataset(
            torch.FloatTensor(X_train), torch.LongTensor(y_train)
        )
        val_ds = TensorDataset(
            torch.FloatTensor(X_val), torch.LongTensor(y_val)
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)
        
        no_improve = 0
        for epoch in range(epochs):
            loss = self.train_epoch(train_loader)
            auc, f1 = self.validate(val_loader)
            self.history["train_loss"].append(loss)
            self.history["val_auc"].append(auc)
            self.history["val_f1"].append(f1)
            
            print(f"Epoch {epoch+1}/{epochs} - loss: {loss:.4f} val_auc: {auc:.4f} val_f1: {f1:.4f}")
            
            if auc > self.best_auc:
                self.best_auc = auc
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if self.best_state:
            self.model.load_state_dict(self.best_state)
        
        return {"best_auc": self.best_auc, "history": self.history}
