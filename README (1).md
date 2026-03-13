# Radiotherapy Treatment Planning — Convex Optimization

Convex optimization approach to IMRT treatment planning for lung cancer,
comparing plans with and without spectral (nuclear norm) regularization.

## Team
- **Chaturya Ganne** — Convex Optimization + Spectral Regularization
- Teammate 2 — Inverse Optimization  
- Teammate 3 — Genetic Algorithm (NSGA-II)

## Project Structure

```
radiotherapy-optimisation/
├── convex_optimization/
│   └── convex_optimisation.ipynb   ← main notebook (run this)
├── results/                        ← plots and CSV saved here after running
├── requirements.txt
└── README.md
```

## Setup & Run

```bash
pip install -r requirements.txt
# Then open convex_optimization/convex_optimisation.ipynb in Colab or Jupyter
```

## Data

Downloads automatically from HuggingFace on first run (~15 min).  
Patient: `Lung_Patient_3` from [PortPy Dataset](https://huggingface.co/datasets/PortPy-Project/PortPy_Dataset).  
**Do NOT commit `.h5` files — they are too large for GitHub.**

## Problem Formulation

Minimize over beamlet intensities x ≥ 0:

    w1 * F1(x)  +  w2 * F2(x)  +  λ * nuclear_norm(X)

where:
- **F1** = mean PTV underdose  (tumor must receive ≥ 60 Gy)
- **F2** = mean OAR overdose   (esophagus ≤ 34 Gy, cord ≤ 45 Gy)
- **nuclear_norm(X)** = spectral regularization for smooth deliverable plans
- Hard constraint: mean lung dose ≤ 20 Gy

Solved via direct SCS cone programming (fully sparse, no memory issues).

## Key Outputs
- DVH comparison across 3 plans
- Singular value spectra showing effect of spectral regularization  
- Lambda sweep Pareto curve (tumor coverage vs OAR sparing)
- Metrics table (D95, Esoph max, Cord max, Lung mean)
