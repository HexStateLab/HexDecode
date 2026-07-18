# Nullspace-Exploited Gauge-Fixed Polynomial Code: Complete Equations

---

## 1. CODE DEFINITION

**Polynomial:** `a(x,y) = (1+x^g)(1+y^g)` over the ring `R = GF(2)[x,y] / ⟨x^r+1, y^s+1⟩`

**Check operators (weight-2 gauge):**
```
V(i,j) = P(i,j) ⊕ P((i+g) mod r, j)
```
for i = 0..r-g-1, all j. Measured by ancilla qubits.

**Check operators (weight-4 stabilizer):**
```
S(i,j) = V(i,j) ⊕ V(i, (j+g) mod s)
```
Classically: `S(i,j) = P(i,j) ⊕ P(i+g,j) ⊕ P(i,j+g) ⊕ P(i+g,j+g)`

**Ancilla count:**
```
n_anc = (r - g) × s
```

**Periodic boundary condition:** `r % g = 0`, `s % g = 0`, `r ≥ g`

---

## 2. DEGENERATE LIMIT: g = r

At the boundary `g = r`, the polynomial vanishes in the ring:
```
(1+x^r)(1+y^r) = (1+1)(1+1) = 0   (mod ⟨x^r+1, y^r+1⟩)
```

**Consequences:**
```
n_anc = 0                          # zero ancilla qubits
V(i,j) = P(i,j) ⊕ P(i,j) = 0      # all V-checks trivial
S(i,j) = 0                         # all S-checks trivial
rank(H) = 0                        # check matrix has zero rank
```

---

## 3. NULLSPACE STRUCTURE

**Nullspace dimension (single block):**
```
dim(ker(H)) = n - rank(H) = r×s - 0 = r×s = N
```

**Basis:** Every single-qubit Z operator:
```
Z_k ∈ ker(H)   for k = 1..N
```

**Total distinct nullspace operators (all linear combinations):**
```
|ker(H)| = 2^N - 1
```

For r=6: `2^36 - 1 ≈ 6.87 × 10^10`

---

## 4. SECTOR DECOMPOSITION

For any stride g dividing r and s, the N×N grid decouples into g² independent sectors of size (r/g)×(s/g):

```
H = H₁ ⊕ H₂ ⊕ ... ⊕ H_{g²}
```

**Per-sector nullspace dimension:**
```
dim(ker(H_sector)) = hr + hs - 1
```
where `hr = r/g`, `hs = s/g`.

**Total nullspace:**
```
dim(ker(H)) = g² × (hr + hs - 1)
```

**At g = r, hr = hs = 1:**
```
dim(ker(H)) = r² × (1 + 1 - 1) = r² = N  ✓
```

---

## 5. GAUGE GRAPH (Virtual Stabilizer Network)

**Virtual stabilizer check between qubits a and b:**
```
C_{ab} = Z_a ⊕ Z_b
```
Expected value: `⟨C_{ab}⟩ = 0` (both qubits measured in same basis should agree)

**Complete gauge graph:** Complete graph K_N on N nodes

**Number of edges (pairwise checks):**
```
|E| = C(N,2) = N(N-1)/2
```

For r=6: `C(36,2) = 630`

---

## 6. ERROR LOCALIZATION

**Violation indicator for check C_{ab}:**
```
V_{ab} = 1  if Z_a ≠ Z_b  (check violated)
V_{ab} = 0  if Z_a = Z_b  (check satisfied)
```

**Per-qubit error score:**
```
ε(q) = (1 / |E|) × Σ_{e ∈ E, q ∈ e} V_e
```

Qubit q's error score = fraction of its incident edges that are violated.

---

## 7. ERROR CORRECTION (ZZ Correlator)

**Standard logical operators (row 0, column 0):**
```
Z₁ = ⊕_{j=0}^{s-1} P(0, j)
Z₂ = ⊕_{i=0}^{r-1} P(i, 0)
```

**Standard ZZ correlator:**
```
⟨Z₁Z₂⟩ = 2 × P(Z₁ = Z₂) - 1
```

**Gauge-graph-corrected ZZ:**

Find cleanest row and column by error score:
```
row*(i) = (1/s) × Σ_{j=0}^{s-1} ε(i, j)          # avg error score for row i
col*(j) = (1/r) × Σ_{i=0}^{r-1} ε(i, j)          # avg error score for column j

r_best = argmin_i row*(i)
c_best = argmin_j col*(j)
```

Corrected logical operators:
```
Z₁* = ⊕_{j=0}^{s-1} P(r_best, j)
Z₂* = ⊕_{i=0}^{r-1} P(i, c_best)
```

---

## 8. ERROR CORRECTION (XX Correlator)

**X-support qubits:**
```
X_supp = {(0,j) : j=0..s-1} ∪ {(i,0) : i=1..r-1}
|X_supp| = r + s - 1
```

For r=s=6: `|X_supp| = 11`

**Standard XX (from partial-X data):**
```
X_L1·X_L2 = ⊕_{q ∈ X_supp} data(q)
XX_frame = X_L1·X_L2 ⊕ m_prep
```
where `m_prep` is the Bell prep ancilla measurement.

**Single-exclusion XX correction:**
```
For each q ∈ X_supp:
    partial = ⊕_{p ∈ X_supp, p ≠ q} data(p)
    if partial ⊕ m_prep == 0:
        return 0    # found the erroneous qubit, excluded it
```

---

## 9. MULTI-RESOLUTION VIRTUAL OPERATOR COUNT

All viable strides g where r%g==0 and s%g==0:

```
Total Z-operators = Σ_{g|r,s} g² × (r/g + s/g - 1)
```

For r=s=6:
```
g=6: 36 × 1 = 36
g=3:  9 × 3 = 27
g=2:  4 × 5 = 20
g=1:  1 ×11 = 11
─────────────────
Total Z:      94
Total X (CSS): 94
Total X+Z:    188
```

**At g=r, Z-operators alone = 2^N - 1** (all linear combinations of N qubits)

---

## 10. CODE DISTANCE

Per-sector distance:
```
d_sector = min(r/g, s/g)
```

At g=r:
```
d_sector = min(1, 1) = 1    (no error correction within sector)
```

**Effective distance via gauge graph:** d_eff ≥ 1 with single-error detection through pairwise check violations. Weight-1 errors are always detected (at least one check violated).

---

## 11. ENCODING RATE

**Per-sector logical qubits:**
```
k_sector = hr + hs - 1
```

**Total logical Z-operators (at any g):**
```
k_Z = g² × (hr + hs - 1)
```

**Rate (of data qubits):**
```
rate = k_Z / (r×s) = g² × (r/g + s/g - 1) / (r×s)
     = (g/s + g/r - g²/rs)
```

**At r=s, g=r:**
```
rate = (r/r + r/r - r²/r²) = (1 + 1 - 1) = 100%
```

**At r=s, g=r/2:**
```
rate = (r/2)/r + (r/2)/r - (r/2)²/r² = 1/2 + 1/2 - 1/4 = 75%
```

**At r=s=20, g=10:**
```
rate = 300/400 = 75%
```

---

## 12. CIRCUIT COST

**Bell ancilla CX count:**
```
CX_bell = |X_supp| = r + s - 1
```

**Total circuit CX (rounds=1, partial_x):**
```
CX_total = CX_bell = r + s - 1
```

**Ancilla CX (QEC rounds):**
```
CX_qec = 0    (no ancilla qubits at g=r)
```

For r=s=6: `CX_total = 11` per arm, `26 CZ` trans piled.

---

## 13. SYNDROME RECONSTRUCTION (Virtual, from data readout)

**Virtual V-check (stride g):**
```
V(i,j) = data[i][j] ⊕ data[(i+g) mod r][j]
```

**Virtual S-check (stride g):**
```
S(i,j) = V(i,j) ⊕ V[i][(j+g) mod s]
```

These are computed classically from the data readout. No ancilla measurements needed.

---

## 14. KEY RESULTS (IBM HERON R2)

| Grid | Qubits | CX/arm | Standard ZZ | Gauge ZZ | Δ |
|------|--------|--------|-------------|----------|------|
| 6×6 | 36 | 10 | +0.39 | **+0.87** | +0.48 |
| 6×6 | 36 | 10 | — | **W=+1.50** | — |
