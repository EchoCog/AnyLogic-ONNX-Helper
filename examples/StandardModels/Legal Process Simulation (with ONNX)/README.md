# Legal Process Simulation (with ONNX)

A court/legal proceedings simulation driven by two ONNX neural networks, built using the AnyLogic ONNX Helper library.

## Overview

This model simulates the lifecycle of legal cases flowing through a court system. Cases are generated, filed, reviewed by clerks, go through discovery, get scheduled, attend court hearings, and reach resolution (settlement, verdict, or dismissal).

Two AI models drive key simulation parameters:

| Model | File | Input | Output |
|-------|------|-------|--------|
| **Case Filing Rate Predictor** | `case_rate_model.onnx` | Sliding window of 6 recent daily filing rates | Predicted next filing rate |
| **Case Duration Predictor** | `case_duration_model.onnx` | 8 case attributes (type, complexity, evidence count, parties, continuances, judge experience, jurisdiction, jury trial) | Predicted duration in days |

## Architecture — Case Flow

```
caseIntake ──► filingQueue ──► initialReview ──► discoveryQueue ──► discovery ──► schedulingQueue ──► courtHearing ──► resolution
 (Source)       (Queue)         (Delay)           (Queue)           (Delay)        (Queue)            (Delay)          (Sink)
    │                           capacity:                           duration:                         duration:
    │                           totalClerks                         complexity *                      ONNX predicted
    │                                                               uniform(5,10)                    (or stochastic
    │                                                               days                             fallback)
    │
    └── rate driven by ONNX prediction (updated every 24h)
```

## Agent Types

### Main
Top-level simulation agent containing the process flow, ONNX helpers, and statistics.

### Case
Entity agent representing a legal case with:
- `caseType` — 0=CIVIL, 1=CRIMINAL, 2=FAMILY, 3=CORPORATE
- `complexity` — 1–10 scale
- `evidenceCount` — number of evidence items
- `numberOfParties` — parties involved
- `priorContinuances` — prior continuances
- `judgeExperience` — years of experience (1–30)
- `jurisdictionCode` — 0–4 (district)
- `isJuryTrial` — boolean
- `predictedDuration` — set when entering court hearing
- `predictedOutcome` — outcome tracking
- `flowchartEntryDay` — timestamp for total time tracking

## ONNX Models

### Case Rate Model (`case_rate_model.onnx`)
- **Architecture**: 2-layer feedforward network (6 → 16 → 1)
- **Input**: `float[6]` — sliding window of last 6 daily filing rates
- **Output**: `float[1]` — predicted next filing rate
- **Usage**: Called every 24 simulated hours via `updateFilingRate` event; updates `caseIntake.set_rate()`

### Case Duration Model (`case_duration_model.onnx`)
- **Architecture**: 3-layer feedforward network (8 → 32 → 16 → 1)
- **Input**: `float[8]` — case features (type, complexity, evidence, parties, continuances, judge experience, jurisdiction, jury)
- **Output**: `float[1]` — predicted duration in days
- **Usage**: Called per-case when entering `courtHearing` delay block

## Running the Simulation in AnyLogic

1. Install the [ONNX Helper Library](https://github.com/the-anylogic-company/AnyLogic-ONNX-Helper/releases) in AnyLogic
2. Open `Legal Process Simulation (with ONNX).alp` in AnyLogic
3. Run the `Simulation` experiment
4. Adjust parameters:
   - **totalCourtrooms** (default 5) — capacity of the court hearing phase
   - **totalClerks** (default 3) — capacity of the initial review phase
   - **useAIPredictions** (default true) — toggle between AI-driven and stochastic rates/durations

## Retraining the Models

### Prerequisites
```bash
cd python
pip install -r requirements.txt
```

### Train the filing rate model
```bash
python train_case_rate_model.py
```
Generates `../case_rate_model.onnx` — input `[1,6]`, output `[1,1]`.

### Train the case duration model
```bash
python train_case_duration_model.py
```
Generates `../case_duration_model.onnx` — input `[1,8]`, output `[1,1]`.

Both scripts use synthetic data with realistic distributions:
- Criminal cases average ~180 days, civil ~120, family ~90, corporate ~240
- Higher complexity, more evidence, more parties, and jury trials increase duration
- More experienced judges reduce duration
- Filing rates follow seasonal patterns with weekly cycles and holiday effects

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `totalCourtrooms` | int | 5 | Concurrent court hearings capacity |
| `totalClerks` | int | 3 | Concurrent initial review capacity |
| `useAIPredictions` | boolean | true | Use ONNX models vs. stochastic defaults |

## Statistics & Visualization

- **filingRateDS** — tracks daily filing rate over time
- **pendingCasesDS** — tracks total pending cases across all queues and delays
- **resolutionTimeHD** — histogram of total case resolution times (days)
