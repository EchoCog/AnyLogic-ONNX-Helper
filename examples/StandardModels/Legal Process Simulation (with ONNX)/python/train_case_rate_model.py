"""
Train a simple linear regression model for predicting court case filing rates.

The model takes a sliding window of 6 recent filing rates and predicts the next
filing rate.  Synthetic time-series data is generated with seasonal patterns,
weekly cycles, and holiday effects.

Output: ../case_rate_model.onnx  (input [1,6] -> output [1,1])
"""

import os
import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

WINDOW = 6
NUM_DAYS = 730  # ~2 years of daily rates
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)


def generate_filing_rates(num_days: int) -> np.ndarray:
    """Generate realistic daily case filing rates."""
    t = np.arange(num_days, dtype=np.float32)

    # Base rate (cases per day)
    base = 12.0

    # Yearly seasonal pattern (peak in spring/fall, dip in summer/winter)
    seasonal = 3.0 * np.sin(2 * np.pi * t / 365.0 - np.pi / 4)

    # Weekly cycle (lower on weekends)
    weekly = -4.0 * np.maximum(0, np.sin(2 * np.pi * (t % 7) / 7.0 - 1.0))

    # Holiday dips (simplified – every ~90 days a 3-day dip)
    holiday = np.zeros(num_days, dtype=np.float32)
    for start in range(0, num_days, 90):
        holiday[start : start + 3] = -5.0

    # Random noise
    noise = np.random.normal(0, 1.2, num_days).astype(np.float32)

    rates = base + seasonal + weekly + holiday + noise
    rates = np.clip(rates, 0.5, None)  # filing rate can't go below 0.5
    return rates.astype(np.float32)


def make_windows(rates: np.ndarray, window: int):
    """Create sliding-window samples (X) and targets (y)."""
    X, y = [], []
    for i in range(len(rates) - window):
        X.append(rates[i : i + window])
        y.append(rates[i + window])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------


class FilingRatePredictor(nn.Module):
    """Simple two-layer network for rate prediction."""

    def __init__(self, input_size: int = WINDOW):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train():
    rates = generate_filing_rates(NUM_DAYS)
    X, y = make_windows(rates, WINDOW)

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    model = FilingRatePredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    epochs = 300
    for epoch in range(1, epochs + 1):
        pred = model(X_t)
        loss = loss_fn(pred, y_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}/{epochs}  loss={loss.item():.4f}")

    # ------------------------------------------------------------------
    # Export to ONNX
    # ------------------------------------------------------------------
    model.eval()
    dummy = torch.randn(1, WINDOW)
    out_path = os.path.join(os.path.dirname(__file__), "..", "case_rate_model.onnx")
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=13,
    )
    print(f"Exported ONNX model to {out_path}")

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------
    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)

    sess = ort.InferenceSession(out_path)
    sample = X[:1]
    ort_out = sess.run(None, {"input": sample})[0]
    torch_out = model(torch.from_numpy(sample)).detach().numpy()
    print(f"Validation  ONNX={ort_out[0][0]:.4f}  PyTorch={torch_out[0][0]:.4f}")


if __name__ == "__main__":
    train()
