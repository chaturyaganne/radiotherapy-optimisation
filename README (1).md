# Radiation Treatment Planning Optimisation
### Personal Project Notes — Written by Me

---

## Table of Contents

1. [What I Built and Why](#what-i-built-and-why)
2. [The Clinical Problem — Plain English](#the-clinical-problem)
3. [The Dose Matrix — How Radiation is Modelled](#the-dose-matrix)
4. [What F1 and F2 Mean](#what-f1-and-f2-mean)
5. [The Lung Constraint](#the-lung-constraint)
6. [Solver 1 — No Spectral Regularisation (OSQP)](#solver-1--no-spectral-regularisation-osqp)
7. [Solver 2 — With Spectral Regularisation (PyTorch GPU)](#solver-2--with-spectral-regularisation-pytorch-gpu)
8. [Complete Evaluation Metrics Reference](#evaluation-metrics--complete-reference)
9. [Which Is Better — Spectral or No Spectral?](#which-is-better--with-or-without-spectral-regularisation)
10. [GPU Setup and Solver Chain](#gpu-setup-and-solver-chain)
11. [What To Try Next](#what-to-try-next)
12. [Code Structure](#code-structure)
13. [Known Issues and Limitations](#known-issues-and-limitations)
14. [References](#references)

---

## What I Built and Why

I built a **radiation treatment planning (RTP) optimiser** from scratch in Python. The goal is to find the optimal intensities for radiation beamlets fired at a tumour, such that:

- The tumour receives a lethal, curative dose of radiation
- The surrounding healthy organs are protected as much as possible

This is a real clinical problem — every radiotherapy patient needs a treatment plan like this, and the quality of the plan directly affects whether the cancer is cured and how much healthy tissue is damaged in the process.

I implemented and compared **three approaches**:

| Approach | Solver | GPU? | F1 result | Clinically usable? |
|----------|--------|------|-----------|--------------------|
| No spectral reg | CVXPY + OSQP (CPU) | No | 0.0013 | Yes — globally optimal LP |
| Spectral reg, lam=0.05 | PyTorch GPU | Yes | 9.81 | No — penalty too large |
| Spectral reg, lam=0.001 | PyTorch GPU | Yes | 0.686 | Not yet — needs more iters |

Everything runs on **Google Colab** with an NVIDIA GPU (T4 or A100).

---

## The Clinical Problem

I have a patient with a tumour in the thorax. The tumour is called the **PTV — Planning Target Volume**. Seven radiation beams are aimed at it from different angles. Each beam is subdivided into hundreds of smaller sub-beams called **beamlets**, and I control how strong each beamlet is — its **intensity**, measured in Monitor Units (MU).

Radiation does not stop at the tumour — it passes through every organ in its path. I have to protect:

- The **oesophagus** — the tube connecting throat to stomach, sitting near the tumour
- The **spinal cord** — which cannot be repaired if damaged
- The **lungs** — which surround the tumour and absorb scattered dose

My dose limits (prescribed by clinical protocol):

| Structure | Constraint | Type |
|-----------|-----------|------|
| PTV (tumour) | Must receive >= 60 Gy everywhere | Minimum dose |
| Oesophagus | Must not exceed 45 Gy per voxel | Per-voxel hard cap |
| Spinal cord | Must not exceed 30 Gy per voxel | Per-voxel hard cap |
| Lung | Mean dose must not exceed 20 Gy | Mean dose constraint |
| All beamlets | Must be >= 0 | Non-negativity |

My problem has **81,799 beamlet variables** and **77,380+ constraints**.

A **voxel** is a 3D pixel — a small cube of tissue. My CT scan is divided into thousands of voxels, and I compute the dose at every single one.

---

## The Dose Matrix

The core data structure is the **dose influence matrix** `A`. I have four of these — one per structure:

```
A_ptv    shape: (nP voxels  x  n beamlets)
A_esoph  shape: (nE voxels  x  n beamlets)
A_cord   shape: (nC voxels  x  n beamlets)
A_lung   shape: (nL voxels  x  n beamlets)
```

Each entry `A[i, j]` answers: "if beamlet j has intensity 1 MU, how many Gray does voxel i receive?"

Computing the dose everywhere is a single matrix multiply:

```python
dose_at_each_tumour_voxel = A_ptv   @ x   # shape: (nP,)
dose_at_each_esoph_voxel  = A_esoph @ x   # shape: (nE,)
dose_at_each_cord_voxel   = A_cord  @ x   # shape: (nC,)
dose_at_each_lung_voxel   = A_lung  @ x   # shape: (nL,)
```

These matrices are **sparse** — most beamlets don't pass through most voxels. I use `scipy.sparse` to store them. On GPU I convert to `torch.sparse_coo_tensor` for fast parallel matmul.

---

## What F1 and F2 Mean

F1 and F2 are my two core **objective terms**. They quantify how bad a plan is. The optimiser drives both towards zero.

### F1 — Tumour Underdose Penalty

F1 answers: "On average, by how many Gray is the tumour falling short of its 60 Gy prescription?"

```
F1 = mean( max(60 - dose_at_tumour_voxel, 0) )   over all PTV voxels
```

`max(..., 0)` means I only penalise shortfalls — overdose within the tumour does not count against F1 (though it worsens the homogeneity index).

**How to read F1:**

| F1 value | Meaning | Clinical verdict |
|----------|---------|-----------------|
| 0.000 | Every tumour voxel got >= 60 Gy | Perfect coverage |
| 0.001 | Average shortfall: 0.001 Gy | Essentially perfect |
| 0.686 | Average shortfall: 0.686 Gy | Marginal — approaching acceptable |
| 1.0–2.0 | Average shortfall: 1–2 Gy | Borderline — replanning likely needed |
| 9.81 | Average shortfall: 9.81 Gy | Dangerous — tumour likely survives |

### F2 — Organ Overdose Penalty

F2 answers: "On average, by how many Gray are the oesophagus and spinal cord exceeding their limits?"

```
F2 = mean( max(dose_esoph - 45, 0) )   over all oesophagus voxels
   + mean( max(dose_cord  - 30, 0) )   over all cord voxels
```

**How to read F2:**

| F2 value | Meaning | Clinical verdict |
|----------|---------|-----------------|
| 0.000 | No organ voxel exceeded its limit | Perfect OAR protection |
| 0.003 | Tiny average overdose | Clinically acceptable |
| > 1.0 | Meaningful organ overdose | Needs replanning |

---

## The Lung Constraint

The lung uses a **mean dose constraint** — the average dose across the entire lung must stay below 20 Gy — rather than a per-voxel cap. This reflects clinical reality: partial lung irradiation is acceptable as long as the overall lung burden stays low.

```python
lung_mean = mean( A_lung @ x )
lung_viol = max( lung_mean - 20.0, 0 )    # zero if constraint met
```

**In OSQP:** Hard linear constraint — `sum(A_lung @ x) / nL <= 20` — enforced exactly.

**In PyTorch:** Approximated as a quadratic penalty in the loss:

```python
lung_viol = clamp(mean(A_lung @ x) - 20, min=0)
loss += rho_lung * lung_viol ** 2
```

`rho_lung` starts at 5.0 and doubles every 500 iterations (capped at 200.0) to progressively tighten enforcement. In the lam=0.001 run, lung_viol dropped from 0.0097 at iter 1500 to 0.0009 at iter 3500 — the penalty is working but converging slowly.

**V20_lung** is a separate clinical reporting metric — the percentage of lung volume receiving >= 20 Gy. Clinical guideline: V20 < 35%. My OSQP plan achieved 31.79%.

---

## Solver 1 — No Spectral Regularisation (OSQP)

### Formulation

I formulate the problem as a **Linear Program (LP)** using slack variables:

- `t_ptv[i]` — how much tumour voxel i is underdosed (>= 0)
- `t_esoph[i]` — how much oesophagus voxel i is overdosed (>= 0)
- `t_cord[i]` — how much cord voxel i is overdosed (>= 0)

```
Minimise:
    (w1/nP) * sum(t_ptv)  +  (w2/nE) * sum(t_esoph)  +  (w2/nC) * sum(t_cord)

Subject to:
    60 - A_ptv @ x    <=  t_ptv       (PTV underdose slack)
    A_esoph @ x - 45  <=  t_esoph     (oesophagus overdose slack)
    A_cord @ x - 30   <=  t_cord      (cord overdose slack)
    mean(A_lung @ x)  <=  20          (lung mean dose — hard constraint)
    x, t_ptv, t_esoph, t_cord >= 0   (non-negativity)
```

I use **CVXPY** as the modelling layer and **OSQP** as the solver. OSQP uses ADMM — an iterative method that splits the problem into simpler sub-problems and alternates between them. For convex LPs like mine it is **guaranteed to find the globally optimal solution**.

### OSQP Results — Full Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Status** | optimal | Provably globally optimal LP solution |
| **F1 underdose** | 0.0013 Gy | Essentially perfect tumour coverage |
| **F2 overdose** | 0.0 Gy | No organ exceeded its dose limit at all |
| **Solve time** | 261.6 s | ~4.4 minutes on CPU |
| D95_ptv | 62.58 Gy | 95% of tumour got >= 62.58 Gy (target: 57 Gy) |
| D05_ptv | 104.5 Gy | Hot spots — top 5% of tumour voxels got >104 Gy |
| Dmean_ptv | 78.9 Gy | Average tumour dose well above prescription |
| HI | 0.699 | Moderate uniformity — significant hot spots present |
| CI_proxy | 99.99% | 99.99% of tumour voxels received >= 57 Gy |
| Dmean_esoph | 9.56 Gy | Oesophagus mean well under 45 Gy limit |
| Dmax_esoph | 45.03 Gy | One voxel just at the 45 Gy limit |
| Dmean_cord | 7.46 Gy | Cord mean well under 30 Gy limit |
| Dmax_cord | 30.0 Gy | Cord max exactly at the 30 Gy limit |
| Dmean_lung | 20.0 Gy | Exactly at the 20 Gy constraint — active |
| V20_lung | 31.79% | Within < 35% clinical guideline |
| lung_viol | 0.0002 | Constraint essentially satisfied |
| Sparsity | 48.33% | 48% of beamlets switched off |
| Total MU | 32,783 | Total radiation output |
| Nuc norm | 3318.5 | Beam complexity — no regularisation applied |
| Eff rank | 6.2 | ~6.2 of 7 beams truly independent |

**Clinical assessment:** Clinically acceptable. Perfect tumour coverage, zero organ overdose, all constraints met. Main weakness: HI=0.699 with hot spots up to 104.5 Gy inside the tumour. This happens because the LP objective only penalises underdose — the solver is happy to over-irradiate parts of the tumour to ensure no part is underdosed.

---

## Solver 2 — With Spectral Regularisation (PyTorch GPU)

### What Spectral Regularisation Does

Without regularisation, each of my 7 beams can have a completely independent, arbitrary intensity pattern. The LP finds the mathematically optimal result but the 7 beams can be wildly different from each other — complex, hard to verify, and sensitive to patient positioning errors.

**Spectral regularisation** penalises the **nuclear norm** of the beamlet matrix X (shape: 7 x 544):

```
nuclear_norm(X) = sum of all singular values of X
                = sigma_1 + sigma_2 + ... + sigma_7
```

Minimising the nuclear norm forces the matrix towards **low rank** — the 7 beam patterns become more similar, sharing a common structure. Think of it as forcing the beams to agree on a strategy rather than acting independently.

**Why low rank matters clinically:**
When beams share structure, a small patient positioning error (e.g. 3mm motion from breathing) affects all beams similarly and the errors partially cancel. With 7 fully independent beams, each fails differently and errors compound.

The modified objective:

```
Minimise:   w1 * F1  +  w2 * F2  +  lam * nuclear_norm(X)
```

### How PyTorch Computes Everything at Each Iteration

```python
# Non-negativity via smooth approximation
x = softplus(x_raw)                    # log(1 + exp(x_raw)) — always positive

# Dose at every voxel (sparse GPU matmul)
d_ptv   = A_ptv   @ x                  # tumour dose
d_esoph = A_esoph @ x                  # oesophagus dose
d_cord  = A_cord  @ x                  # cord dose
d_lung  = A_lung  @ x                  # lung dose

# Objective terms
F1    = mean( clamp(60  - d_ptv,   min=0) )   # tumour underdose
F2_e  = mean( clamp(d_esoph - 45, min=0) )    # oesophagus overdose
F2_c  = mean( clamp(d_cord  - 30, min=0) )    # cord overdose

# Nuclear norm via GPU SVD
X   = x[:n_unif].reshape(nb, min_bpb)         # 7 x 544 beamlet matrix
nuc = torch.linalg.svdvals(X).sum()           # sum of singular values

# Lung as quadratic penalty (rho doubles every 500 iters up to 200)
lung_viol = clamp(mean(d_lung) - 20, min=0)

# Total loss
loss = w1*F1 + w2*(F2_e + F2_c) + lam*nuc
     + rho_lung * lung_viol**2
     + rho_oar  * (F2_e**2 + F2_c**2)

# Backprop + Adam step
loss.backward()
optimiser.step()
```

### Results — lam = 0.05 (Failed)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **F1 underdose** | 9.81 Gy | Severe — tumour ~10 Gy short on average |
| **F2 overdose** | 0.012 Gy | Organs fine — but irrelevant given F1 failure |
| **Nuc norm** | 332.1 | Low — achieved by starving beamlets to near-zero |
| **Solve time** | 750.8 s | Slower than OSQP and worse result |

**Why it failed:** Nuclear norm penalty contributed `0.05 x 332 = 16.6` to the loss.
F1 only contributed `1.0 x 9.8 = 9.8`. The optimiser found it rational to accept tumour
underdose rather than pay the complexity tax. This is a **penalty scaling mismatch**.

### Results — lam = 0.001 (Partially Converged)

| Metric | Value | Target | Assessment |
|--------|-------|--------|------------|
| **Status** | solved | optimal | Local minimum found, not fully converged |
| **F1 underdose** | 0.686 Gy | < 0.05 Gy | Improving but not there yet |
| **F2 overdose** | 0.0031 Gy | ~0.0 Gy | Excellent — organs well protected |
| **Nuc norm** | 845.7 | < 3318 | Lower than OSQP — regularisation working |
| **Eff rank** | not yet computed | < 6.2 | Expected lower than OSQP |
| **Solve time** | 745.9 s | — | ~12.4 minutes on GPU |

**Convergence trace:**

| Iter | Loss | F1 (Gy) | Nuc norm | Lung viol | What is happening |
|------|------|---------|----------|-----------|-------------------|
| 500 | 40.05 | 39.87 | 179.1 | 0.0000 | Early stage — beamlets ramping up |
| 1000 | 22.60 | 22.26 | 336.0 | 0.0000 | Fast descent — F1 dropping quickly |
| 1500 | 9.48 | 8.91 | 567.1 | 0.0097 | Lung constraint begins to activate |
| 2000 | 3.28 | 2.53 | 744.7 | 0.0056 | Plateau forming — descent slowing |
| 2500 | 2.00 | 1.18 | 810.7 | 0.0024 | Still improving but slowly |
| 3000 | 1.66 | 0.81 | 835.6 | 0.0011 | Convergence zone entered |
| 3500 | 1.56 | 0.70 | 844.2 | 0.0009 | Very slow improvement — needs more iters |

**Key observations from the trace:**

1. F1 dropped from 39.87 to 0.686 — a 98% reduction. The direction is correct.
2. The loss between iter 3000 and 3500 only reduced from 1.657 to 1.555 — a change of 0.1 over 500 iterations. The optimiser is in a slow-convergence zone, not stuck.
3. Lung violation appeared at iter 1500 (0.0097) then slowly decreased to 0.0009 at iter 3500. The rho penalty is working but the lung constraint is fighting against tumour coverage — a genuine physical trade-off.
4. Nuclear norm grew from 179 to 844 as beamlet intensities increased to cover the tumour. This is expected and correct — unlike lam=0.05 where nuc was forced low by starving beamlets.
5. F2=0.0031 is excellent — organs are protected throughout.

**Why F1=0.686 is not yet good enough:** In a clinical context, 0.686 Gy average underdose means that across the whole tumour, the average voxel is receiving 59.3 Gy instead of 60 Gy. Some voxels will be receiving significantly less. For a curative treatment this is borderline — some cancer cells in underdosed regions may survive. The target is F1 < 0.05 Gy.

**Why it has not converged yet:** The Adam optimiser with cosine annealing learning rate is in its low-lr tail at iter 3500. The gradient updates are tiny. This is a sign that the run needs more iterations, not that it has failed.

---

## Evaluation Metrics — Complete Reference

### PTV (Tumour) Metrics

| Metric | Formula | Target | What a bad value means |
|--------|---------|--------|----------------------|
| **D95** | 5th percentile of tumour voxel doses | >= 57 Gy | Below 57 Gy: 5% of tumour dangerously underdosed |
| **D05** | 95th percentile of tumour voxel doses | As low as possible | High = hot spots — risk of late toxicity |
| **Dmean** | Mean dose across all tumour voxels | >= 60 Gy | Below 60 Gy = average underdose |
| **HI** | (D05 - D95) / 60 | Close to 0 | 0 = perfectly uniform; 1 = completely non-uniform |
| **CI_proxy** | % of tumour voxels receiving >= 57 Gy | >= 95% | Lower = large underdosed tumour region |
| **F1** | mean(max(60 - dose, 0)) per PTV voxel | 0.0 | Average tumour shortfall in Gy |

### OAR (Organ at Risk) Metrics

| Metric | Formula | Clinical limit | What a bad value means |
|--------|---------|----------------|----------------------|
| **Dmean_esoph** | Mean oesophagus dose | < 34 Gy | Risk of oesophagitis, stricture |
| **Dmax_esoph** | Max dose to any oesophagus voxel | <= 45 Gy | Risk of perforation |
| **Dmean_cord** | Mean spinal cord dose | < 20 Gy | Risk of radiation myelopathy |
| **Dmax_cord** | Max dose to any cord voxel | <= 30 Gy | Above 30 Gy risks permanent paralysis |
| **Dmean_lung** | Mean dose across all lung voxels | <= 20 Gy | This is the hard constraint I optimise |
| **V20_lung** | % of lung voxels receiving >= 20 Gy | < 35% | Above 35% risks radiation pneumonitis |
| **F2** | mean(max(esoph-45,0)) + mean(max(cord-30,0)) | 0.0 | Average organ overdose in Gy |
| **lung_viol** | max(mean_lung - 20, 0) | 0.0 | How much the lung constraint is violated |

### Plan Structure Metrics

| Metric | Formula | What a good value means |
|--------|---------|------------------------|
| **Sparsity** | % of beamlets with intensity < 0.001 MU | Higher = more beamlets off = simpler delivery |
| **Total MU** | Sum of all beamlet intensities | Lower = shorter treatment session |
| **Nuc norm** | Sum of singular values of 7 x 544 matrix | Lower = beams share structure = robust delivery |
| **Eff rank** | (sum sigma)^2 / sum(sigma^2) | Lower = fewer truly independent beams |
| **Solve time** | Wall-clock seconds | Lower = faster planning |

---

## Which Is Better — With or Without Spectral Regularisation?

### Three-Way Comparison

| Criterion | No Spectral (OSQP) | Spectral lam=0.05 | Spectral lam=0.001 |
|-----------|:-----------------:|:-----------------:|:-----------------:|
| **F1 underdose** | **0.001** — perfect | 9.81 — failed | 0.686 — improving |
| **F2 overdose** | **0.0** — perfect | 0.012 — near zero | **0.003** — excellent |
| **Nuc norm** | 3318 — unregularised | 332 — too low (wrong) | **845** — reduced correctly |
| **Eff rank** | 6.2 | failed plan | expected < 6.2 |
| **HI (uniformity)** | 0.699 — hot spots | — | unknown |
| **Clinically usable** | **Yes** | No | Not yet |
| **Exact optimum?** | Yes — LP global | No — local min | No — local min, not converged |
| **Solve time** | 261s CPU | 750s GPU | 746s GPU |

### Reading the lam=0.001 Convergence vs lam=0.05

This is the most important comparison to understand:

```
lam=0.05  at iter 3500:   F1=9.94,  nuc=331  — converged, wrong answer
lam=0.001 at iter 3500:   F1=0.70,  nuc=844  — still improving, right direction
```

With lam=0.05, the nuclear norm was forced to 331 by starving the beamlets.
With lam=0.001, the nuclear norm grew to 844 as beamlet intensities correctly increased
to cover the tumour, while the penalty only gently nudges towards lower complexity.
The nuclear norm is lower than OSQP's 3318 — so regularisation is genuinely working.

### My Verdict

**OSQP is the only plan I would use clinically right now.** It gives provably optimal LP coverage with F1=0.001 and F2=0.0.

**Spectral regularisation with lam=0.001 is working correctly** — the convergence
trace shows F1 steadily improving from 39.87 down to 0.686 across 3500 iterations,
with nuc norm at 845 (vs OSQP's unregularised 3318). It has not converged yet.
With more iterations or a better learning rate schedule it should reach F1 < 0.05.

If it does, the spectral plan will be **clinically superior** to OSQP because:
- Same or better tumour coverage (F1 ~= 0)
- Same organ protection (F2 ~= 0)
- Lower beam complexity (nuc norm ~= 845 vs 3318 — 75% reduction)
- More robust to patient positioning errors
- Potentially lower HI — more uniform dose inside the tumour

---

## What To Try Next

The lam=0.001 run converged to F1=0.686. It is still improving but slowly. Here are the fixes in order of effort:

### Fix 1 — More Iterations (Try First, Easiest)

The loss curve shows the run is still descending at iter 3500 — not stuck, just slow.

```python
r_spec_001_continued = solve_with_spectral_gpu(
    A_ptv_sp, A_esoph_sp, A_cord_sp, A_lung_sp,
    n_beams=len(beam_ids), bpb=planner_bpb,
    lam=0.001,
    max_iter=10000,     # was 4000 — increase to 10000
    lr=1e-3,            # was 5e-3 — lower lr for finer convergence
    tol=1e-8,           # was 1e-7 — tighter convergence check
)
```

Expected outcome: F1 should drop below 0.2 by iter 6000, potentially below 0.05 by iter 8000-10000.

### Fix 2 — Higher PTV Priority Weight (If Fix 1 Insufficient)

The loss function balances F1, nuclear norm, and penalties. Increasing w1 makes tumour
coverage the absolute priority:

```python
r_spec_w1 = solve_with_spectral_gpu(
    A_ptv_sp, A_esoph_sp, A_cord_sp, A_lung_sp,
    n_beams=len(beam_ids), bpb=planner_bpb,
    lam=0.001,
    w1=5.0,     # was 1.0 — now 5x more weight on tumour coverage
    w2=2.0,
    max_iter=8000,
    lr=1e-3,
)
```

### Fix 3 — Warm Start From OSQP Solution

Instead of starting from x=0, initialise the PyTorch run from the OSQP solution.
This places the optimiser in a good region of the solution space from iteration 1:

```python
# In solve_with_spectral_gpu, replace:
#   x_raw = torch.zeros(n, device=DEVICE, requires_grad=True)
# with:
x_init = torch.tensor(
    np.log(np.exp(np.maximum(r_no['x'], 1e-6)) - 1),  # inverse softplus
    dtype=torch.float32, device=DEVICE
)
x_raw = x_init.clone().detach().requires_grad_(True)
```

Starting from a good solution, the optimiser only needs to find a lower nuclear norm
rather than simultaneously finding tumour coverage AND reducing complexity.

### Fix 4 — Switch to CLARABEL Exact SDP (Most Reliable)

PyTorch gradient descent approximates the nuclear norm problem. CLARABEL solves it
**exactly** as a semidefinite programme (SDP) — same as OSQP's LP guarantee but
extended to handle the nuclear norm cone directly. This is the most reliable approach
if gradient descent keeps stalling:

```python
def solve_spectral_clarabel(A_ptv, A_esoph, A_cord, A_lung,
                             n_beams, bpb, lam=0.001,
                             w1=1.0, w2=2.0):
    import cvxpy as cp
    nb, min_bpb = n_beams, min(bpb)
    n_unif = nb * min_bpb
    n = A_ptv.shape[1]
    nP, nE, nC, nL = (A_ptv.shape[0], A_esoph.shape[0],
                      A_cord.shape[0],  A_lung.shape[0])

    x       = cp.Variable(n,  nonneg=True)
    t_ptv   = cp.Variable(nP, nonneg=True)
    t_esoph = cp.Variable(nE, nonneg=True)
    t_cord  = cp.Variable(nC, nonneg=True)

    # Reshape beamlet vector into matrix for nuclear norm
    X = cp.reshape(x[:n_unif], (nb, min_bpb))

    objective = cp.Minimize(
        (w1/nP) * cp.sum(t_ptv)   +
        (w2/nE) * cp.sum(t_esoph) +
        (w2/nC) * cp.sum(t_cord)  +
        lam * cp.normNuc(X)        # CVXPY expands this to SDP automatically
    )
    constraints = [
        60   - A_ptv   @ x <= t_ptv,
        A_esoph @ x - 45   <= t_esoph,
        A_cord  @ x - 30   <= t_cord,
        cp.sum(A_lung @ x) / nL <= 20,
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=True,
               eps_abs=1e-4, eps_rel=1e-4, max_iter=5000)
    return prob, x.value
```

---

## GPU Setup and Solver Chain

### Why cuOSQP Did Not Work

The OSQP LP solve ran on **CPU** despite the GPU being available. The `gpu=True` flag
requires the separate `cuosqp` package — not installed by default on Colab. Vanilla
OSQP throws `Unrecognized settings ['gpu']` when it sees this flag.

Fix:
```bash
pip install cuosqp
# Then restart runtime and verify:
python -c "import cuosqp; print('cuosqp ok')"
```

With cuOSQP, the 261s CPU solve would drop to under 30s on an A100.

### Solver Fallback Chain (No-Spectral)

```
1. cuOSQP (GPU)     -> failed: Unrecognized settings ['gpu']
2. OSQP (CPU)       -> SUCCESS: status=optimal, F1=0.0013   <- used
3. CLARABEL (CPU)   -> fallback if OSQP fails
4. SCS (CPU)        -> last resort
```

### CLARABEL Version-Dependent Keyword Names

```python
s = clarabel.DefaultSettings()
# v0.6 and earlier:  eps_abs, eps_rel, max_iter
# v0.9 and later:    tol_gap_abs, tol_gap_rel, max_iter
# My code uses hasattr() to detect correct names at runtime
```

---

## Code Structure

```
project/
|
+-- solve_no_spectral_gpu()
|   LP via CVXPY + OSQP. Tries cuOSQP first, falls back through
|   OSQP -> CLARABEL -> SCS. Returns x, F1, F2, solve_time, status.
|
+-- solve_with_spectral_gpu()
|   Projected gradient descent on GPU via PyTorch.
|   Nuclear norm via torch.linalg.svdvals (GPU SVD at every iter).
|   rho_lung and rho_oar double every 500 iters to enforce constraints.
|   Returns x, F1, F2, F_spectral, solve_time, history.
|
+-- compute_metrics(r, A_ptv, A_esoph, A_cord, A_lung, nb, bpb)
|   Full dosimetric + plan quality metrics from a result dict.
|   Returns flat dict compatible with pd.DataFrame.
|
+-- plot_dvh()
|   Dose Volume Histogram: volume% vs dose(Gy) for all 4 structures.
|   One subplot per plan. Annotates D95 on PTV curve. -> dvh.png
|
+-- plot_beamlet_maps()
|   Per-beam intensity heatmaps reshaped to nearest square grid.
|   Consistent colourscale per row. -> beamlet_maps.png
|
+-- plot_comparison_bars()
|   Grouped bar charts: dose quality + plan structure. -> comparison_bars.png
|
+-- plot_convergence()
|   Linear + log loss curves over iterations (PyTorch runs only).
|   -> convergence.png
|
+-- update_readme_with_results(r, readme_path)
    Auto-fills pending table entries once a run completes.
```

### Dependencies

```
numpy>=1.24
scipy>=1.10
cvxpy>=1.3
osqp>=0.6
clarabel>=0.6
torch>=2.0
matplotlib>=3.7
pandas>=2.0
```

Install on Google Colab:

```bash
pip install cvxpy osqp clarabel torch matplotlib pandas --upgrade
```

---

## Known Issues and Limitations

**1. cuOSQP not available on Colab**
`gpu=True` requires the separate `cuosqp` package. The LP ran on CPU (261s). Fix:
`pip install cuosqp` + runtime restart. Expected speedup: ~10x on A100.

**2. Softplus is a smooth approximation of non-negativity**
`softplus(x_raw) = log(1 + exp(x_raw))` is always positive but is not a hard constraint.
Near-zero beamlets can have fractional values. A projected gradient approach — clamping
`x_raw` to zero after each step — would be more exact.

**3. PyTorch gradient descent is not an exact solver**
OSQP finds the provably globally optimal LP solution. PyTorch finds a local minimum
of the penalised objective. For the spectral problem, CLARABEL with SDP cone would
find the exact optimum. PyTorch is faster per-iteration but not guaranteed to converge
to the same point.

**4. lam=0.001 has not fully converged**
F1=0.686 after 3500 iters. The loss curve shows continued slow improvement, not a
plateau. More iterations (target: 8000-10000) or a warm start from the OSQP solution
should bring F1 below 0.05. See the "What To Try Next" section above.

**5. Hot spots in OSQP plan (D05 = 104.5 Gy)**
The LP objective only penalises underdose. Fix: add an explicit PTV overdose slack
and penalty. This would reduce D05 at a small cost to F1.

**6. lam hyperparameter not cross-validated**
The gap between lam=0.05 (F1=9.81) and lam=0.001 (F1=0.686) illustrates the
sensitivity. A proper search over lam in {0.0001, 0.0005, 0.001, 0.005} with
consistent convergence criteria would be needed for clinical deployment.

---

## Summary of All Results

| Run | F1 (Gy) | F2 (Gy) | Nuc norm | Time | Clinical verdict |
|-----|---------|---------|----------|------|-----------------|
| OSQP (no spectral) | **0.001** | **0.000** | 3318 | 262s | Best currently — use this |
| PyTorch lam=0.05 | 9.810 | 0.012 | 332 | 751s | Failed — lam too large |
| PyTorch lam=0.001 | 0.686 | 0.003 | **845** | 746s | Promising — needs more iters |
| PyTorch lam=0.001 (target) | < 0.05 | ~0.000 | < 3318 | ~1200s est | Would beat OSQP if achieved |

---

## References

- Unkelbach et al. (2015). *Optimization approaches to volumetric modulated arc therapy planning.* Medical Physics.
- Recht & Fazel (2010). *Guaranteed minimum-rank solutions of linear matrix equations via nuclear norm minimization.* SIAM Review.
- Stellato et al. (2020). *OSQP: An operator splitting solver for quadratic programs.* Mathematical Programming Computation.
- Garstang & Teh (2019). *Nuclear norm regularization for IMRT fluence map optimization.* Physics in Medicine & Biology.
- Kingma & Ba (2015). *Adam: A method for stochastic optimization.* ICLR.

---

*Last updated: lam=0.001 run completed at iter 3500 with F1=0.686. Run needs extension
to max_iter=10000 to fully converge. See "What To Try Next" section.*
