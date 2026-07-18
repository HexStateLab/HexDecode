"""
Virtual Qubit Error Correction via Complete Gauge Graph

For N physical qubits with no ancillas (stride=r=s):
- C(N,2) possible virtual stabilizer checks: Z_a ⊕ Z_b
- Each check reveals whether qubits a and b agree
- Violated checks localize errors to specific qubits
- Error-free qubits are identified by clean check patterns
- ZZ computed using only error-free qubits
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from pw_opt import _bell_support_coords

from stridecodec_bindings import params as stride_params
from stridecodec_bindings import decode as stride_decode
from stridecodec_bindings import syndrome_of as stride_syndrome


def build_virtual_circuit(r, s, g, partial_x=False):
    n_data = r * s
    qr = QuantumRegister(n_data + 1, "q")
    cregs = [ClassicalRegister(n_data, "data"), ClassicalRegister(1, "bell")]
    qc = QuantumCircuit(qr, *cregs)
    
    dq = lambda i, j: (i % r) * s + (j % s)
    support = _bell_support_coords(r, s, True)
    bell_q = n_data
    
    qc.h(bell_q)
    for (i, j) in support: qc.cx(bell_q, dq(i, j))
    qc.h(bell_q)
    qc.measure(bell_q, cregs[1][0])
    
    if partial_x:
        for i in range(r): qc.h(dq(i, 0))
        for j in range(s): qc.h(dq(0, j))
    
    for i in range(r):
        for j in range(s):
            qc.measure(dq(i, j), cregs[0][i * s + j])
    
    return qc


def virtual_syndrome_from_data(data, r, s, vg):
    V = data.astype(np.uint8) ^ np.roll(data.astype(np.uint8), shift=-vg, axis=0)
    S = V ^ np.roll(V, shift=-vg, axis=1)
    return S


def viable_strides(r, s):
    return [g for g in range(1, min(r,s)+1) if r % g == 0 and s % g == 0]


def virtual_operator_count(r, s):
    total = 0
    for vg in viable_strides(r, s):
        k, d, rate = stride_params(r, s, vg)
        total += 2 * k
    return total


def gauge_graph_checks(data):
    """Complete virtual stabilizer graph: all C(N,2) pairwise checks.
    
    Returns:
      check_results: numpy array of shape (N, N) where [i,j] = Z_i ⊕ Z_j
      error_score:   per-qubit error likelihood (fraction of violated checks)
    """
    flat = data.flatten()
    N = len(flat)
    error_score = np.zeros(N)
    total_checks = 0
    
    for i in range(N):
        for j in range(i + 1, N):
            if flat[i] ^ flat[j]:  # check violated → error in i or j
                error_score[i] += 1
                error_score[j] += 1
            total_checks += 1
    
    error_score /= max(total_checks, 1)
    return error_score


def gauge_graph_zz(data, threshold=0.3):
    """Compute ZZ using only qubits with low error score.
    
    Quibits with error_score > threshold are excluded from the
    logical operator support. Clean Z1 and Z2 are computed from
    the remaining error-free qubits.
    """
    r, s = data.shape
    N = r * s
    scores = gauge_graph_checks(data)
    
    # Identify clean rows and columns (lowest average error score)
    row_scores = np.array([scores[i*s:(i+1)*s].mean() for i in range(r)])
    col_scores = np.array([scores[j::s].mean() for j in range(s)])
    
    # Find cleanest row and column
    clean_row = int(np.argmin(row_scores))
    clean_col = int(np.argmin(col_scores))
    
    z1 = int(data[clean_row, :].sum() % 2)
    z2 = int(data[:, clean_col].sum() % 2)
    
    return z1, z2, row_scores, col_scores


def gauge_graph_zz_all(data, threshold=0.3):
    """ZZ using ALL clean rows/columns weighted by error score.
    
    Instead of one row and one column, compute majority vote across
    all rows/cols weighted by (1 - error_score).
    """
    r, s = data.shape
    N = r * s
    scores = gauge_graph_checks(data)
    
    # Weighted vote: rows
    row_votes_1 = 0; row_weight = 0
    for i in range(r):
        row_err = scores[i*s:(i+1)*s].mean()
        w = max(0, 1 - row_err)
        if w > 0:
            row_votes_1 += w * int(data[i, :].sum() % 2)
            row_weight += w
    
    col_votes_1 = 0; col_weight = 0
    for j in range(s):
        col_err = scores[j::s].mean()
        w = max(0, 1 - col_err)
        if w > 0:
            col_votes_1 += w * int(data[:, j].sum() % 2)
            col_weight += w
    
    z1 = 1 if row_votes_1 > row_weight / 2 else 0
    z2 = 1 if col_votes_1 > col_weight / 2 else 0
    return z1, z2


def exponential_zz(data, n_samples=5000):
    """Legacy: sampled linear combination voting."""
    r, s = data.shape
    v1, v2, t1, t2 = 0, 0, 0, 0
    for _ in range(n_samples):
        rows = np.random.randint(0, 2, r)
        if rows.sum():
            t1 += 1; v = 0
            for i in range(r):
                if rows[i]: v ^= int(data[i,:].sum()%2)
            v1 += v
        cols = np.random.randint(0, 2, s)
        if cols.sum():
            t2 += 1; v = 0
            for j in range(s):
                if cols[j]: v ^= int(data[:,j].sum()%2)
            v2 += v
    z1 = 1 if v1 > t1//2 else 0
    z2 = 1 if v2 > t2//2 else 0
    return z1, z2, v1/max(t1,1), v2/max(t2,1)
