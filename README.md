# test-HumanAI

**Logic Layer Implementation for ISSR AI4MH Crisis Detection**  
**Google Summer of Code 2026 – Contributor Selection Task**  
---

## 📋 Project Overview

This repository contains a **modular Python implementation** of the exact logic layer requested in the GSoC 2026 ISSR contributor selection task.

It implements the full **Crisis Signal Design** (Composite Crisis Index + Confidence Estimate) and **Governance & Risk Controls** (bot amplification, media-driven spikes, rural underrepresentation, escalation thresholds, audit logging) — exactly as specified in the test document.

**Note:** This is the **basic working version** I submitted with the updated PDF. I can expand it into a production-ready module today if required.

---

## 🗂️ Repository Structure
test-HumanAI/
├── Crisis_detection_Framework/          # Core Crisis Scoring Framework
│   ├── ci.py                           # Composite Crisis Index (S + V + G)
│   └── ce.py                           # Confidence Estimate + Noise Penalty
├── Gov_and_Risk_Controles/             # Governance & Risk Controls
│   ├── MD_spikes.py                    # Media-driven spike detection (GNews API)
│   ├── temporal_fingerprinting.py      # Coordinated activity / bot patterns
│   ├── bayesian_smoothning.py          # Bayesian smoothing for sparse/rural data
│   ├── E_and_H.py                      # Escalation thresholds + Human-in-the-loop
│   └── text_classification.py          # Additional risk classification helper
├── requirements.txt
├── .gitignore
└── README.md
text---

## ✅ Mapping to GSoC Test Requirements

### 1. Crisis Signal Design (Core Component)
- **Sentiment Intensity (S)**, **Volume Spike Detection (V)**, **Geographic Clustering (G)** → `ci.py`
- **Minimum sample size threshold**, **Smoothing**, **Confidence Estimate (CE)** → `ce.py`
- Full **Composite Crisis Index (CI)** calculation with weighted MCDA
- Noise penalty for bots/media (exactly as described in the task screenshot)

### 2. Governance & Risk Controls
- Bot amplification & coordinated activity → `temporal_fingerprinting.py`
- Media-driven spikes → `MD_spikes.py`
- Rural underrepresentation / sparse data → `bayesian_smoothning.py`
- Escalation thresholds & Human-in-the-loop → `E_and_H.py`
- Full **audit logging** structure implemented

### 3. Governance Reflection
Included in the submitted PDF (primary risk + single most important safeguard).

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/nilesh03pandey/test-HumanAI.git
cd test-HumanAI
2. Install dependencies
Bashpip install -r requirements.txt
3. Run the demo
Bashpython Crisis_detection_Framework/ci.py
python Crisis_detection_Framework/ce.py
(Each file contains its own example at the bottom.)

🛠️ Tech Stack

Python 3.10+
SentenceTransformers (embeddings)
GNews (real-time media spike detection)
NumPy / Scikit-learn
RAKE keyword extraction

See requirements.txt for full list.
