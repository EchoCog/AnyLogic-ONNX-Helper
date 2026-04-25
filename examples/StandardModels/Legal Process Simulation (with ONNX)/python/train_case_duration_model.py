"""
Train a feedforward neural network for predicting case duration (in days).

Features (8 total):
  0  case_type          – 0=CIVIL, 1=CRIMINAL, 2=FAMILY, 3=CORPORATE
  1  complexity         – 1-10
  2  evidence_count     – number of evidence items
  3  num_parties        – number of parties involved
  4  prior_continuances – number of prior continuances
  5  judge_experience   – years of experience
  6  jurisdiction_code  – 0-4 (district)
  7  is_jury_trial      – 0 or 1

Target: case duration in days

Output: ../case_duration_model.onnx  (input [1,8] -> output [1,1])
"""

import os
import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

NUM_SAMPLES = 5000
NUM_FEATURES = 8
SEED = 123

np.random.seed(SEED)
torch.manual_seed(SEED)

# Base durations by case type (days)
BASE_DURATION = {0: 120.0, 1: 180.0, 2: 90.0, 3: 240.0}


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


def generate_case_data(n: int):
    """Generate synthetic case feature/duration data."""
    case_type = np.random.randint(0, 4, n).astype(np.float32)
    complexity = np.random.randint(1, 11, n).astype(np.float32)
    evidence_count = np.random.poisson(15, n).astype(np.float32)
    num_parties = np.random.randint(2, 7, n).astype(np.float32)
    prior_continuances = np.random.poisson(1.5, n).astype(np.float32)
    judge_experience = np.random.uniform(1, 30, n).astype(np.float32)
    jurisdiction_code = np.random.randint(0, 5, n).astype(np.float32)
    is_jury_trial = np.random.binomial(1, 0.3, n).astype(np.float32)

    X = np.column_stack(
        [
            case_type,
            complexity,
            evidence_count,
            num_parties,
            prior_continuances,
            judge_experience,
            jurisdiction_code,
            is_jury_trial,
        ]
    )

    # Duration = base + complexity effect + evidence effect + continuances
    #          + jury effect - judge experience effect + noise
    duration = np.array([BASE_DURATION[int(ct)] for ct in case_type], dtype=np.float32)
    duration += complexity * 12.0                       # higher complexity → longer
    duration += evidence_count * 2.5                    # more evidence → longer
    duration += num_parties * 8.0                       # more parties → longer
    duration += prior_continuances * 15.0               # continuances add delay
    duration -= judge_experience * 1.5                  # experienced judges faster
    duration += is_jury_trial * 30.0                    # jury trials take longer
    duration += np.random.normal(0, 15, n).astype(np.float32)  # noise
    duration = np.clip(duration, 14, None)              # minimum 2 weeks

    return X.astype(np.float32), duration.reshape(-1, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------


class CaseDurationPredictor(nn.Module):
    """Simple feedforward network for case duration prediction."""

    def __init__(self, input_size: int = NUM_FEATURES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train():
    X, y = generate_case_data(NUM_SAMPLES)

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    model = CaseDurationPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    epochs = 500
    for epoch in range(1, epochs + 1):
        pred = model(X_t)
        loss = loss_fn(pred, y_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs}  loss={loss.item():.4f}")

    # ------------------------------------------------------------------
    # Export to ONNX
    # ------------------------------------------------------------------
    model.eval()
    dummy = torch.randn(1, NUM_FEATURES)
    out_path = os.path.join(os.path.dirname(__file__), "..", "case_duration_model.onnx")
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

    # Print a few sample predictions for reference
    print("\nSample predictions:")
    print(f"{'Type':>6} {'Cmplx':>5} {'Evid':>5} {'Party':>5} {'Cont':>5} "
          f"{'JudgExp':>7} {'Juris':>5} {'Jury':>4}  {'Predicted':>9}")
    for i in range(5):
        row = X[i]
        p = ort_out[0][0] if i == 0 else sess.run(None, {"input": X[i : i + 1]})[0][0][0]
        ct_names = ["CIVIL", "CRIM", "FAMILY", "CORP"]
        print(
            f"{ct_names[int(row[0])]:>6} {row[1]:5.0f} {row[2]:5.0f} {row[3]:5.0f} "
            f"{row[4]:5.0f} {row[5]:7.1f} {row[6]:5.0f} {row[7]:4.0f}  {p:9.1f} days"
        )


if __name__ == "__main__":
    train()
