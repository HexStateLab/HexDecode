#!/usr/bin/env python3
"""
hexdecode_virtual.py — Virtual QEC via Nullspace Partitioning

For the (1+x^g)(1+y^g) code at stride g:
  - Nullspace dimension: g² × (r/g + s/g - 1)
  - Virtual QEC: designate some nullspace vectors as stabilizers,
    others as logical operators. No ancilla qubits needed.

For stride-6 on 6×6: 36 nullspace vectors.
  Partition into: 9 virtual stabilizers + 27 logical operators
  → Effectively stride-3 code, d=2, 75% rate
  → 0 ancilla qubits, 0 ancilla CX gates

Circuit: Bell ancilla only (20 CX). No QEC ancilla rounds.
Virtual syndrome reconstructed from data readout via stride-3
check equation. Correction from stride-3 decoder.
"""

import sys, os, time, argparse, getpass
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'Heron-R2'))
sys.path.insert(0, os.path.join(_here, '..'))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, 'hexdecode'))

from stridecodec_bindings import decode as stride_decode
from stridecodec_bindings import syndrome_of as stride_syndrome
from stridecodec_bindings import is_stabilizer as stride_is_stab
from stridecodec_bindings import params as stride_params

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
except ImportError:
    pass

from pw_opt import _bell_support_coords
from offline_sim import setup


def build_virtual_circuit(r, s, g, partial_x=False):
    """Build Bell test circuit for virtual QEC code.
    
    No QEC ancilla rounds. Only Bell prep ancilla.
    XX correlator from partial_x data + frame correction.
    """
    n_data = r * s
    n_extra = 1  # Bell ancilla only
    
    qr = QuantumRegister(n_data + n_extra, "q")
    cregs = [ClassicalRegister(n_data, "data"), ClassicalRegister(1, "bell")]
    qc = QuantumCircuit(qr, *cregs)
    
    dq = lambda i, j: (i % r) * s + (j % s)
    support = _bell_support_coords(r, s, True)
    bell_q = n_data
    
    # Bell prep (creates Bell state, measures initial frame)
    qc.h(bell_q)
    for (i, j) in support: qc.cx(bell_q, dq(i, j))
    qc.h(bell_q)
    qc.measure(bell_q, cregs[1][0])
    
    # Partial-X: H on X-support qubits (for XX arm)
    if partial_x:
        for i in range(r): qc.h(dq(i, 0))      # col 0
        for j in range(s): qc.h(dq(0, j))      # row 0
    
    # Data readout
    for i in range(r):
        for j in range(s):
            qc.measure(dq(i, j), cregs[0][i * s + j])
    
    return qc


def virtual_syndrome_from_data(data, r, s, virtual_g):
    """Reconstruct virtual stride-g syndrome from data readout.
    
    Uses the free_final_round computation:
      V[i][j] = data[i][j] ⊕ data[(i+g)%r][j]
      S[i][j] = V[i][j] ⊕ V[i][(j+g)%s]
    """
    V = data.astype(np.uint8) ^ np.roll(data.astype(np.uint8), 
                                          shift=-virtual_g, axis=0)
    S = V ^ np.roll(V, shift=-virtual_g, axis=1)
    return S


def virtual_decode(data, r, s, physical_g, virtual_g):
    """Decode using virtual QEC.
    
    physical_g: the code's built-in stride (e.g. 6)
    virtual_g:  the effective stride from nullspace partitioning (e.g. 3)
    
    The virtual syndrome is reconstructed from data readout.
    The decoder uses the virtual stride.
    """
    syn = virtual_syndrome_from_data(data, r, s, virtual_g)
    corr = stride_decode(r, s, virtual_g, syn)
    return corr


def demo():
    """Test virtual QEC on 6×6 stride=6 → virtual stride=3."""
    r, s, g = 6, 6, 6
    vg = 3  # virtual stride from nullspace partitioning
    
    k6, d6, r6 = stride_params(r, s, g)
    k3, d3, r3 = stride_params(r, s, vg)
    
    print(f"{'='*60}")
    print(f"  Virtual QEC: stride-{g} → stride-{vg}")
    print(f"  Physical: stride-{g} ({k6} nullspace, d≈{d6}, {r6:.0f}% rate)")
    print(f"  Virtual:  stride-{vg} ({k3} logicals, d≈{d3}, {r3:.0f}% rate)")
    print(f"  {g*g} sectors → partitioned into {vg*vg} virtual sectors")
    print(f"{'='*60}")
    
    # Build circuits
    qc_zz = build_virtual_circuit(r, s, g, partial_x=False)
    qc_xx = build_virtual_circuit(r, s, g, partial_x=True)
    cx_z = qc_zz.count_ops().get('cx', 0)
    cx_x = qc_xx.count_ops().get('cx', 0)
    print(f"\nCircuits: ZZ={qc_zz.num_qubits}q {cx_z}CX  XX={qc_xx.num_qubits}q {cx_x}CX")
    
    # Test noiselessly
    _, sampler = setup(seed=42)
    pub_z = sampler.run([qc_zz], shots=500).result()[0]
    pub_x = sampler.run([qc_xx], shots=500).result()[0]
    
    db_z = pub_z.data.data.to_bool_array(order='little').astype(np.uint8).reshape(-1, r, s)
    bp_z = pub_z.data.bell.to_bool_array(order='little')[:, 0].astype(np.uint8)
    
    db_x = pub_x.data.data.to_bool_array(order='little').astype(np.uint8).reshape(-1, r, s)
    bp_x = pub_x.data.bell.to_bool_array(order='little')[:, 0].astype(np.uint8)
    
    # Virtual decode on ZZ arm
    corr = np.zeros_like(db_z)
    t0 = time.perf_counter()
    for shot in range(500):
        corr[shot] = virtual_decode(db_z[shot], r, s, g, vg)
    dt = time.perf_counter() - t0
    
    fixed = db_z ^ corr
    z1 = fixed[:, 0, :].sum(1) % 2
    z2 = fixed[:, :, 0].sum(1) % 2
    zz = 2.0 * (z1 == z2).mean() - 1.0
    
    # XX from partial_x data + frame correction
    xx_par = np.zeros(500, dtype=np.uint8)
    for j in range(s): xx_par ^= db_x[:, 0, j]
    for i in range(r): xx_par ^= db_x[:, i, 0]
    xx = 2.0 * ((xx_par ^ bp_x) == 0).mean() - 1.0
    W = zz + xx
    
    print(f"  ZZ_corr={zz:+.4f}  XX_frame={xx:+.4f}  W={W:+.4f}")
    print(f"  Decode: {dt*1e3:.1f}ms  Corr weight: {corr.sum(axis=(1,2)).mean():.1f}")
    
    # Test with noise
    for noise in [0.001, 0.005, 0.01]:
        _, sampler_n = setup(seed=42, two_qubit_rate=noise)
        pub_zn = sampler_n.run([qc_zz], shots=300).result()[0]
        pub_xn = sampler_n.run([qc_xx], shots=300).result()[0]
        db_zn = pub_zn.data.data.to_bool_array(order='little').astype(np.uint8).reshape(-1, r, s)
        bp_xn = pub_xn.data.bell.to_bool_array(order='little')[:, 0].astype(np.uint8)
        db_xn = pub_xn.data.data.to_bool_array(order='little').astype(np.uint8).reshape(-1, r, s)
        
        corr_n = np.zeros_like(db_zn)
        for shot in range(300):
            corr_n[shot] = virtual_decode(db_zn[shot], r, s, g, vg)
        fixed_n = db_zn ^ corr_n
        
        z1n = fixed_n[:, 0, :].sum(1) % 2
        z2n = fixed_n[:, :, 0].sum(1) % 2
        zz_n = 2.0 * (z1n == z2n).mean() - 1.0
        
        xx_par_n = np.zeros(300, dtype=np.uint8)
        for j in range(s): xx_par_n ^= db_xn[:, 0, j]
        for i in range(r): xx_par_n ^= db_xn[:, i, 0]
        xx_n = 2.0 * ((xx_par_n ^ bp_xn) == 0).mean() - 1.0
        W_n = zz_n + xx_n
        
        n_errs = int(corr_n.sum())
        print(f"  noise={noise}: ZZ={zz_n:+.4f} XX={xx_n:+.4f} W={W_n:+.4f}  "
              f"corrections={n_errs} flips")
    
    # Test error injection
    print(f"\n  Error injection test:")
    err_test = db_z[0].copy()
    err_test[2, 3] ^= 1
    corr_test = virtual_decode(err_test, r, s, g, vg)
    fixed_test = err_test ^ corr_test
    ok = stride_is_stab(r, s, vg, fixed_test)
    print(f"    Injected bit-flip at (2,3): corrected={'✓' if ok else '✗'}  "
          f"corr_weight={int(corr_test.sum())}")


if __name__ == '__main__':
    demo()
