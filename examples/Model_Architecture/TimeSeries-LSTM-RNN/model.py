"""
LSTM Time Series Forecaster
============================
Architecture:
    3-layer stacked LSTM (input_size=1, hidden_size=128) followed by a fully
    connected projection head that maps the final hidden state to a multi-step
    prediction horizon.

    Input:  (batch, 96 timesteps, 1 feature)
    Output: (batch, 24 timesteps) -- predicted future values

Real-world use cases:
    - Energy demand forecasting (electricity load, solar generation)
    - Financial time series prediction (stock prices, volatility)
    - Weather forecasting (temperature, precipitation)
    - Sensor / IoT anomaly detection and predictive maintenance

This demo generates a synthetic noisy sine wave, trains the model to predict
24 future steps from 96 past steps, and saves a prediction-vs-ground-truth
plot to 'prediction_result.png'.
"""

import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class LSTMForecaster(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 3,
        prediction_horizon: int = 24,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, prediction_horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)          # (batch, seq_len, hidden)
        last_hidden = lstm_out[:, -1, :]    # (batch, hidden)
        return self.fc(last_hidden)          # (batch, prediction_horizon)


# ---------------------------------------------------------------------------
# Dataset -- synthetic sine wave with noise
# ---------------------------------------------------------------------------
class SineWaveDataset(Dataset):
    def __init__(self, num_samples: int, lookback: int, horizon: int, seed: int = 42):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon

        rng = torch.Generator().manual_seed(seed)
        total_len = lookback + horizon
        t = torch.linspace(0, 8 * math.pi, total_len).unsqueeze(0).expand(num_samples, -1)

        # Random phase and frequency perturbations per sample
        phase = torch.rand(num_samples, 1, generator=rng) * 2 * math.pi
        freq = 1.0 + 0.3 * torch.randn(num_samples, 1, generator=rng)
        noise = 0.05 * torch.randn(num_samples, total_len, generator=rng)

        self.data = torch.sin(freq * t + phase) + noise  # (num_samples, total_len)

    def __len__(self) -> int:
        return self.data.size(0)

    def __getitem__(self, idx: int):
        seq = self.data[idx]
        x = seq[: self.lookback].unsqueeze(-1)  # (lookback, 1)
        y = seq[self.lookback:]                   # (horizon,)
        return x, y


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        total_loss += criterion(pred, y).item() * x.size(0)
    return total_loss / len(loader.dataset)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    LOOKBACK = 96
    HORIZON = 24
    EPOCHS = 30
    BATCH_SIZE = 64
    LR = 1e-3

    device = select_device()
    print(f"Using device: {device}")

    train_ds = SineWaveDataset(num_samples=2000, lookback=LOOKBACK, horizon=HORIZON, seed=42)
    val_ds = SineWaveDataset(num_samples=400, lookback=LOOKBACK, horizon=HORIZON, seed=99)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = LSTMForecaster(prediction_horizon=HORIZON).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("-" * 50)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  train_mse={train_loss:.6f}  val_mse={val_loss:.6f}")

    # --- Plot a sample prediction vs ground truth ---
    model.eval()
    sample_x, sample_y = val_ds[0]
    with torch.no_grad():
        pred = model(sample_x.unsqueeze(0).to(device)).cpu().squeeze()

    fig, ax = plt.subplots(figsize=(12, 4))
    full_gt = torch.cat([sample_x.squeeze(), sample_y])
    time_all = range(len(full_gt))
    ax.plot(time_all, full_gt.numpy(), label="Ground Truth", color="steelblue")
    time_pred = range(LOOKBACK, LOOKBACK + HORIZON)
    ax.plot(time_pred, pred.numpy(), label="Prediction", color="tomato", linestyle="--")
    ax.axvline(x=LOOKBACK, color="gray", linestyle=":", alpha=0.6, label="Forecast start")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Value")
    ax.set_title("LSTM Time Series Forecast -- Prediction vs Ground Truth")
    ax.legend()
    fig.tight_layout()
    fig.savefig("prediction_result.png", dpi=150)
    print("\nPlot saved to prediction_result.png")
