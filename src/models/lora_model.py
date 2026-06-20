"""
LoRA-Enhanced Credit Scorer using Hugging Face PEFT.

Uses DistilBERT as base with LoRA adaptation. Tabular features are projected
to transformer embedding space. Flexible for any number of features.
Reference: Hu et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models"
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType


class LoRACreditScorer(nn.Module):
    """
    LoRA-adapted Transformer for credit scoring.
    Projects tabular features to BERT embedding dim, fine-tunes with LoRA.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 768,
        base_model_name: str = "distilbert-base-uncased",
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim

        self.feature_projection = nn.Sequential(
            nn.Linear(num_features, min(hidden_dim * 2, num_features * 4)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(min(hidden_dim * 2, num_features * 4), hidden_dim),
        )

        config = AutoConfig.from_pretrained(base_model_name, num_labels=2)
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name, config=config
        )

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_lin", "v_lin"],
            bias="none",
            task_type=TaskType.SEQ_CLS,
        )
        self.model = get_peft_model(self.base_model, lora_config)
        self.base_model_name = base_model_name
        self.classifier_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        emb = self.feature_projection(x)
        inputs_embeds = emb.unsqueeze(1)
        attention_mask = torch.ones(batch_size, 1, dtype=torch.long, device=x.device)
        # Route through PEFT-wrapped encoder so LoRA adapters participate in inference.
        base = self.model.get_base_model()
        encoder = getattr(base, "distilbert", None) or base.model.distilbert
        hidden = encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state[:, 0, :]
        # Residual tabular signal — avoids DistilBERT collapsing diverse inputs to one logit.
        hidden = hidden + emb
        return self.classifier_head(hidden)

    def trainable_parameter_groups(self) -> list:
        """Higher LR for tabular layers; lower LR for LoRA adapters."""
        tabular = list(self.feature_projection.parameters()) + list(self.classifier_head.parameters())
        tabular_ids = {id(p) for p in tabular}
        lora = [p for p in self.model.parameters() if p.requires_grad and id(p) not in tabular_ids]
        return [
            {"params": tabular, "lr": 1e-3, "name": "tabular"},
            {"params": lora, "lr": 2e-4, "name": "lora"},
        ]

    def predict_proba_numpy(self, X: np.ndarray, device: str = "cpu") -> np.ndarray:
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(X).to(device)
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            return probs.cpu().numpy()

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "trainable_pct": 100 * trainable / total if total > 0 else 0,
        }

    def save_pretrained(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "num_features": self.num_features,
                "hidden_dim": self.hidden_dim,
            },
            path,
        )

    @classmethod
    def load_pretrained(cls, path: str, device: str = "cpu") -> "LoRACreditScorer":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(num_features=ckpt["num_features"], hidden_dim=ckpt["hidden_dim"])
        state = ckpt["model_state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            head_missing = [k for k in missing if k.startswith(("feature_projection", "classifier_head"))]
            if head_missing:
                raise RuntimeError(
                    f"Incomplete checkpoint at {path}; missing layers: {head_missing[:3]}. "
                    "Retrain with scripts/train.py."
                )
        model.eval()
        return model
