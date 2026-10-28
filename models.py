"""Random Forest and two-layer PyTorch LSTM model implementations."""

from __future__ import annotations

from dataclasses import dataclass
from random import seed as python_seed

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - gives CLI users an actionable error
    raise ImportError("PyTorch is required. Install dependencies with: pip install -r requirements.txt") from exc


def set_reproducible(seed: int = 42) -> None:
    python_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_random_forest(seed: int = 42) -> CalibratedClassifierCV:
    """Balanced forest with out-of-fold sigmoid calibration."""
    forest = RandomForestClassifier(
        n_estimators=450,
        max_depth=10,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=1,
        random_state=seed,
    )
    return CalibratedClassifierCV(forest, method="sigmoid", cv=3, n_jobs=1)


class TheftLSTM(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 48, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden_size, 24), nn.ReLU(), nn.Dropout(dropout), nn.Linear(24, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(inputs)
        return self.head(self.dropout(output[:, -1, :])).squeeze(1)


@dataclass
class LSTMTrainingResult:
    model: TheftLSTM
    history: list[dict[str, float]]
    device: str


def train_lstm(
    train_sequences: np.ndarray,
    train_labels: np.ndarray,
    validation_sequences: np.ndarray,
    validation_labels: np.ndarray,
    *,
    epochs: int = 24,
    batch_size: int = 32,
    learning_rate: float = 2e-3,
    seed: int = 42,
) -> LSTMTrainingResult:
    set_reproducible(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TheftLSTM().to(device)
    x_train = torch.as_tensor(train_sequences, dtype=torch.float32)
    y_train = torch.as_tensor(train_labels, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    positive = max(float(train_labels.sum()), 1.0)
    pos_weight = torch.tensor([(len(train_labels) - positive) / positive], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    validation_x = torch.as_tensor(validation_sequences, dtype=torch.float32, device=device)
    validation_y = torch.as_tensor(validation_labels, dtype=torch.float32, device=device)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    patience_left = 7
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        losses: list[float] = []
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            validation_loss = float(criterion(model(validation_x), validation_y).cpu())
        scheduler.step(validation_loss)
        history.append(
            {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_loss": validation_loss}
        )
        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = 7
        else:
            patience_left -= 1
            if patience_left == 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return LSTMTrainingResult(model=model, history=history, device=device)


def predict_lstm_logits(model: TheftLSTM, sequences: np.ndarray, device: str | None = None) -> np.ndarray:
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(resolved_device).eval()
    inputs = torch.as_tensor(sequences, dtype=torch.float32, device=resolved_device)
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in torch.split(inputs, 128):
            outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs)

