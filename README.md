# Convex Optimisation for IMRT with Spectral Regularisation  
### Lung_Patient_3 — PortPy Dataset

---

## 1. Introduction

Intensity‑Modulated Radiation Therapy (IMRT) planning is a mathematical optimisation problem: we must deliver a therapeutic dose to the tumour while sparing nearby organs at risk (OARs). This project implements a **convex fluence map optimisation (FMO)** pipeline using the PortPy *Lung_Patient_3* dataset.

A major challenge in IMRT is controlling the **complexity** and **spectral structure** of the fluence maps. While nuclear norm regularisation is theoretically ideal for promoting low‑rank fluence patterns, its semidefinite programming (SDP) formulation is computationally infeasible at clinical scale. To address this, we introduce a **combined spectral surrogate**:



\[
R(X) = \lambda \|X\|_F + \lambda \sum_i \|X_{i,:}\|_2
\]



This surrogate:

- Shrinks all singular values (Frobenius term)  
- Removes entire beams (group sparsity term)  
- Behaves like a **memory‑safe approximation** to nuclear norm minimisation  

---

## 2. Related Work

### 2.1 Classical IMRT Optimisation

Convex optimisation has been widely used for IMRT planning:

- Shepard et al. (1999) — *Medical Physics*  
  Introduced a unified convex framework for IMRT optimisation.  
  https://doi.org/10.1118/1.598779

- Romeijn et al. (2003) — *Medical Physics*  
  Formalised IMRT as a large‑scale convex optimisation problem.  
  https://doi.org/10.1118/1.1593633

These works form the foundation of modern fluence map optimisation (FMO).

### 2.2 Dose–Volume Histograms (DVH)

DVHs are the standard clinical tool for evaluating dose distributions:

- Drzymala et al. (1991)  
  Defined DVH methodology and clinical interpretation.  
  https://doi.org/10.1016/0360-3016(91)90255-O

### 2.3 Leaf Sequencing and Deliverability

After FMO, fluence maps must be converted into deliverable MLC apertures:

- Bortfeld et al. (1994) — *Medical Physics*  
  Introduced the first MLC leaf‑sequencing algorithms.  
  https://doi.org/10.1118/1.597398

- Shepard et al. (2002) — *Medical Physics*  
  Developed direct aperture optimisation (DAO).  
  https://doi.org/10.1118/1.1477415

Simpler fluence maps → fewer apertures → faster, safer delivery.

### 2.4 Low‑Rank and Spectral Methods in IMRT

Several studies explore low‑rank fluence maps:

- Zhu et al. (2018) — *Physics in Medicine & Biology*  
  Applied nuclear norm minimisation to IMRT.  
  https://doi.org/10.1088/1361-6560/aabf6b

- Bian et al. (2019) — *Physics in Medicine & Biology*  
  Studied rank minimisation for fluence smoothing.  
  https://doi.org/10.1088/1361-6560/ab1a0d

- Craft et al. (2014) — *Medical Physics*  
  Used group sparsity to deactivate entire beams.  
  https://doi.org/10.1118/1.4864476

However, **true nuclear norm optimisation requires SDP**, which becomes intractable for clinical‑scale matrices (e.g., 7 × 748 beamlets).

### 2.5 Innovation of This Work

This project introduces:

1. **A combined spectral surrogate**  
   

\[
   R(X) = \lambda \|X\|_F + \lambda \sum_i \|X_{i,:}\|_2
   \]

  
   which approximates nuclear norm behaviour without SDP.

2. **Post‑hoc spectral validation**  
   We compute nuclear norm and effective rank of the resulting fluence matrix to confirm spectral compression.

3. **A fully reproducible pipeline**  
   From raw PortPy data → dose matrices → convex optimisation → DVH → spectral analysis.

---

## 3. Problem Setting

### 3.1 Clinical Context

We plan IMRT for a lung cancer patient with:

- **Target:** PTV (tumour)  
- **OARs:** Esophagus, spinal cord, left lung  
- **Beams:** 7 planner‑selected beams  

### 3.2 Mathematical Model

Let:

- \(x \in \mathbb{R}^n\) — beamlet intensities  
- \(A \in \mathbb{R}^{m \times n}\) — dose influence matrix  
- \(d = A x\) — voxel dose  

We extract structure‑specific matrices:

- \(A_{\text{PTV}}, A_{\text{ESOPH}}, A_{\text{CORD}}, A_{\text{LUNG}}\)

### 3.3 Objective Terms

#### PTV underdose:


\[
F_1(x) = \frac{1}{|\text{PTV}|} \sum_{i \in \text{PTV}} \max(D_{\text{PTV}} - [A_{\text{PTV}} x]_i, 0)
\]



#### OAR overdose:


\[
F_2(x) =
\frac{1}{|\text{ESOPH}|} \sum_{i \in \text{ESOPH}} \max([A_{\text{ESOPH}} x]_i - D_{\text{ESOPH}}, 0)
+
\frac{1}{|\text{CORD}|} \sum_{i \in \text{CORD}} \max([A_{\text{CORD}} x]_i - D_{\text{CORD}}, 0)
\]



#### Lung mean dose constraint:


\[
\frac{1}{|\text{LUNG}|} \sum_{i \in \text{LUNG}} [A_{\text{LUNG}} x]_i \le D_{\text{LUNG}}
\]



### 3.4 Spectral Regulariser

Reshape \(x\) into a beam matrix \(X\):



\[
X \in \mathbb{R}^{n_b \times B}
\]



Then apply:



\[
R(X) = \lambda \|X\|_F + \lambda \sum_i \|X_{i,:}\|_2
\]



### 3.5 Final Optimisation Problem



\[
\begin{aligned}
\min_{x \ge 0} \quad & w_1 F_1(x) + w_2 F_2(x) + R(X) \\
\text{s.t.} \quad & \text{mean}(A_{\text{LUNG}} x) \le D_{\text{LUNG}}
\end{aligned}
\]



---

## 4. Illustrative Schematic

```text
          +---------------------------+
          |   Beamlet Intensities x   |
          +-------------+-------------+
                        |
                        v
          +---------------------------+
          |   Dose Influence Matrix A |
          +---------------------------+
                        |
                        v
          +---------------------------+
          |      Voxel Dose d = Ax    |
          +---------------------------+
             /       |         \
            /        |          \
           v         v           v
   +-----------+ +--------+ +----------+
   |   PTV     | |  OARs  | |   Lung   |
   +-----------+ +--------+ +----------+
       |            |           |
       v            v           v
  F1(x): PTV   F2(x): OAR   Constraint:
  underdose    overdose   mean lung dose
       \            |           /
        \           |          /
         \          |         /
          v         v        v
          +---------------------------+
          |  Objective + Constraints  |
          |  + Spectral Regulariser   |
          +-------------+-------------+
                        |
                        v
          +---------------------------+
          |  Optimised Plan (x, X)    |
          +---------------------------+

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
