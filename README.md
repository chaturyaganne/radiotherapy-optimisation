# Convex Optimisation for IMRT with Spectral Regularisation  
### Lung_Patient_3 — PortPy Dataset

---

## Overview

This project implements **Intensity‑Modulated Radiation Therapy (IMRT)** optimisation using **convex programming**.  
The goal is to compute beamlet intensities that:

- Deliver **60 Gy** to the tumour (PTV)  
- Protect nearby organs (OARs)  
- Produce **simple, structured, low‑rank fluence patterns**  
- Avoid the RAM crash caused by true **nuclear norm** optimisation  

To achieve this, we introduce a **combined spectral regulariser**:



\[
R(X) = \lambda \|X\|_F + \lambda \sum_i \|X_{i,:}\|_2
\]



This acts as a **practical surrogate** for nuclear norm regularisation.

---

##  Objectives

1. Maintain tumour coverage (D95 ≥ 60 Gy)  
2. Reduce hotspots and improve homogeneity  
3. Keep OAR doses within clinical limits  
4. Reduce nuclear norm and effective rank of the fluence matrix  
5. Produce simpler, more deliverable beam patterns  

---

##  Notebook Structure

### **1. Installation & Imports**
Installs PortPy, CVXPY, solvers, and scientific libraries.

### **2. Clinical Dose Limits**
Defines protocol constraints:

| Structure | Limit |
|----------|--------|
| PTV | ≥ 60 Gy |
| Esophagus | ≤ 45 Gy |
| Spinal Cord | ≤ 30 Gy |
| Lung | Mean ≤ 20 Gy |

These appear in the objective and constraints.

### **3. Dataset Download**
Downloads **Lung_Patient_3** from HuggingFace, including:

- CT scan  
- Structure masks  
- Dose voxel map  
- Beam influence matrices  
- Planner-selected beam angles  

### **4. Dose Influence Matrix (A)**

The IMRT dose model:



\[
d = A x
\]



Where:

- **x** = beamlet intensities  
- **A** = sparse dose influence matrix  
- **d** = dose delivered to each voxel  

Each beam’s matrix is loaded and horizontally stacked.

### **5. Structure Masks**
Extracts voxel indices for:

- PTV  
- Esophagus  
- Cord  
- Lung  

Builds structure-specific matrices:

- `A_ptv`, `A_esoph`, `A_cord`, `A_lung`

### **6. Planner Beam Selection**
Restricts A to the **7 beams** chosen by the clinical planner.

Variables:

- `planner_bpb` — beamlets per beam  
- `nb` — number of beams  
- `n_beamlets` — total beamlets  

### **7. Convex Optimisation Solver**

#### **Decision variable**


\[
x \in \mathbb{R}^{4420},\quad x \ge 0
\]



#### **Objective terms**
- **F1** — PTV underdose  
- **F2** — OAR overdose  

#### **Constraint**


\[
\text{mean}(A_{\text{lung}} x) \le 20
\]



#### **Combined Spectral Regulariser**


\[
R(X) = \lambda \|X\|_F + \lambda \sum_i \|X_{i,:}\|_2
\]



- Frobenius → shrinks all singular values  
- Group sparsity → removes entire beams  
- Together → approximates nuclear norm without SDP memory explosion  

### **8. Metrics Computation**

For each plan, computes:

#### **PTV Metrics**
- D95, D05  
- Dmean  
- Homogeneity Index (HI)  
- Coverage Index (CI proxy)  

#### **OAR Metrics**
- Dmax esophagus  
- Dmax cord  
- Lung mean dose  
- V20 lung  

#### **Plan Structure Metrics**
- Sparsity (%)  
- Total MU  
- Nuclear norm  
- Effective rank  

---

#  Plot Explanations

## **1. Dose Volume Histogram (DVH)**

Shows dose distribution for:

- PTV  
- Esophagus  
- Cord  
- Lung  

Interpretation:

- Right shift → higher dose  
- Left shift → lower dose  
- Steeper curve → more homogeneous  
- Vertical lines show clinical limits  

Spectral regularisation:

- Removes hotspots (D05 drops dramatically)  
- Maintains D95 ≈ 60 Gy  

---

## **2. Beamlet Intensity Maps**

Each row = one plan  
Each column = one beam  

Shows:

- Fluence smoothness  
- Beam sparsity  
- Beamlet activation patterns  

Spectral regularisation:

- Increases sparsity  
- Reduces total MU  
- Produces smoother, more structured beams  

---

## **3. Lambda Sweep Plot**

Three subplots:

### **(a) F1 — PTV underdose**
- Slight increase with λ  
- Still clinically acceptable  

### **(b) F2 — OAR overdose**
- Slight increase  
- Tradeoff for smoother fluence  

### **(c) Nuclear norm**
- Drops from **6547 → ~1035**  
- Confirms spectral compression  

This is the key evidence that your combined regulariser works.

---

## **4. Comparison Bar Charts**

Two groups:

### **Dose Quality**
- F1  
- F2  
- HI  
- Lung violation  

### **Plan Structure**
- Nuclear norm  
- Effective rank  
- Sparsity  

Shows:

- Homogeneity improves  
- Nuclear norm collapses  
- Sparsity increases  
- MU decreases  

---

#  Key Variables

| Variable | Meaning |
|---------|---------|
| `A` | Full dose influence matrix |
| `A_ptv`, `A_esoph`, `A_cord`, `A_lung` | Structure-specific matrices |
| `x` | Beamlet intensities |
| `X_mat` | Reshaped beam matrix |
| `lam` | Regularisation weight |
| `reg_type` | `"fro"`, `"group"`, `"both"` |
| `F1` | PTV underdose |
| `F2` | OAR overdose |
| `nuc_norm` | Nuclear norm of X |
| `eff_rank` | Effective rank |
| `total_MU` | Total monitor units |
| `sparsity_%` | Fraction of beamlets ≈ 0 |

---

#  Summary

This notebook demonstrates that:

- True nuclear norm optimisation is infeasible due to memory  
- Combined Frobenius + Group regularisation is an effective **spectral surrogate**  
- Nuclear norm drops by **>80%**  
- Hotspots collapse (D05 from 110 Gy → 66 Gy)  
- Homogeneity improves dramatically (HI from 0.83 → 0.10)  
- Total MU drops by ~50%  
- Beam patterns become simpler and more structured  

This provides strong justification for spectral regularisation in IMRT.

---
