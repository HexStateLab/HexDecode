#!/usr/bin/env python3
"""
demo_hexdecode.py — End-to-end HexDecode stride-1 demo.

Compares HexDecode (stride-1) against the legacy stride-2 pipeline
on identical grid dimensions, showing:
  - Circuit build (stride-1 vs stride-2)
  - Decode speed and accuracy
  - Bell witness if requested
  - Heavy-hex zero-SWAP verification
"""
import sys, os, random, numpy as np, time, argparse

# Path: import from parent directories (Heron-R2 for circuit, parent for offline_sim)
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'Heron-R2'))
sys.path.insert(0, os.path.join(_here, '..'))
sys.path.insert(0, _here)

from hexdecode.bindings import (solve_plane, preprocess_syndrome,
                                  is_stabilizer, syndrome_of)
from hexdecode.decoder import (prep, solve, S_of, check_logical,
                                tesseract_decode_ffinal)
from hexdecode.bell import bell_witness
from hexdecode.heavyhex import make_heavyhex_coupling


def bench_memory(r=6, s=8, stride=1, rounds=1, shots=300, noise=0.005):
    """Memory test: |0⟩_L + QEC rounds → decode → LER."""
    periodic = True

    print(f"\n{'='*60}")
    print(f"  HexDecode memory test  {r}×{s}  stride={stride}  "
          f"rounds={rounds}  noise={noise}")

    qc, _, _, _, n_anc = build_circuit(
        r, s, rounds, logical_state='00',
        stabilizer_basis='Z', no_reset=True, full_stabilizer=False,
        periodic=periodic, compact=True, stride=stride)

    n_data = r * s
    cx = qc.count_ops().get('cx', 0)
    print(f"  circuit: {qc.num_qubits}q ({n_data}d+{n_anc}a)  CX={cx}  depth={qc.depth()}")

    # Heavy-hex coupling check
    coupl = make_heavyhex_coupling(r, s, stride)
    print(f"  heavy-hex edges: {len(coupl)}  "
          f"(native={'✓' if stride==1 else '✗ (skip-g not native)'})")

    _, sampler = setup(seed=42, two_qubit_rate=noise)
    pub = sampler.run([qc], shots=shots).result()[0]

    data_bits = pub.data.data.to_bool_array(order='little')
    data_raw = data_bits[:, :n_data].astype(np.uint8).reshape(shots, r, s)

    syn = all_syndromes_opt(pub, rounds, r, s, n_anc, no_reset=True,
                            free_final_round=True, data_raw=data_raw,
                            full_stabilizer=False, periodic=periodic,
                            stride=stride)

    corr = np.zeros((shots, r, s), dtype=np.uint8)
    t0 = time.perf_counter()

    for shot in range(shots):
        syn_last = syn[shot, rounds - 1].copy()
        prep(syn_last, r, s)
        corr[shot] = solve(syn_last, r, s)

    dt = time.perf_counter() - t0

    data_fixed = data_raw ^ corr
    raw_flips = int(data_raw.sum())
    fixed_flips = int(data_fixed.sum())

    ler = sum(not is_stabilizer(r, s, data_fixed[i]) for i in range(shots))

    d_eff = min(r, s) if stride == 1 else min(r // 2, s // 2)
    print(f"  d_eff≈{d_eff}  raw={raw_flips}  fixed={fixed_flips}  "
          f"LER={ler}/{shots} ({ler/shots*100:.1f}%)  "
          f"decode={dt*1e3:.1f}ms")

    return ler / shots * 100


def bench_bell(r=6, s=8, stride=1, rounds=1, shots=200, noise=0.001):
    """Bell test: prepare |Φ⁺⟩_L, QEC rounds, measure ZZ and XX."""
    periodic = True

    print(f"\n{'='*60}")
    print(f"  HexDecode Bell test  {r}×{s}  stride={stride}  "
          f"rounds={rounds}  noise={noise}")

    ff = (rounds == 1 and stride == 2)  # free_final only for single-round stride-2
    qc, dm, lq0, lq1, n_anc = build_circuit(
        r, s, rounds, logical_state='00', bell=True, bell_measure=True,
        free_final_round=ff, stabilizer_basis='Z',
        no_reset=True, full_stabilizer=False,
        periodic=periodic, compact=True, stride=stride)

    n_data = r * s
    ops = qc.count_ops()
    print(f"  circuit: {qc.num_qubits}q ({n_data}d+{n_anc}a)  "
          f"CX={ops.get('cx',0)}  depth={qc.depth()}")

    _, sampler = setup(seed=42, two_qubit_rate=noise)
    pub = sampler.run([qc], shots=shots).result()[0]

    data_bits = pub.data.data.to_bool_array(order='little')
    data_raw = data_bits[:, :n_data].astype(np.uint8).reshape(shots, r, s)
    bell_prep = pub.data.bell.to_bool_array(order='little')[:, 0].astype(np.uint8)
    bell_meas = pub.data.bell_m.to_bool_array(order='little')[:, 0].astype(np.uint8)

    syn = all_syndromes_opt(pub, rounds, r, s, n_anc, no_reset=True,
                            free_final_round=ff, data_raw=data_raw if ff else None,
                            full_stabilizer=False, periodic=periodic,
                            stride=stride)

    corr = np.zeros((shots, r, s), dtype=np.uint8)
    t0 = time.perf_counter()

    for shot in range(shots):
        syn_last = syn[shot, rounds - 1].copy()
        prep(syn_last, r, s)
        corr[shot] = solve(syn_last, r, s)

    dt = time.perf_counter() - t0
    data_fixed = data_raw ^ corr

    # ZZ
    lq0_a = np.array(lq0); lq1_a = np.array(lq1)
    z1 = data_fixed[:, lq0_a//s, lq0_a%s].sum(axis=1) % 2
    z2 = data_fixed[:, lq1_a//s, lq1_a%s].sum(axis=1) % 2
    zz_bits = (z1 ^ z2).astype(np.uint8)

    # XX
    xx_bits = (bell_meas ^ bell_prep).astype(np.uint8)

    W, W_lo, W_hi, zz, xx, _, _ = bell_witness(zz_bits, xx_bits)

    print(f"  ZZ={zz:+.4f}  XX={xx:+.4f}  W={W:+.4f}  95%CI=[{W_lo:+.4f},{W_hi:+.4f}]")
    print(f"  decode={dt*1e3:.1f}ms  "
          f"bell_prep 0/1={int(sum(bell_prep==0))}/{int(sum(bell_prep==1))}  "
          f"preserved={int(sum(bell_meas==bell_prep))}/{shots}")

    if W_lo > 1.0:
        print(f"  ✓ ENTANGLED")
    return W


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='HexDecode — stride-1 toric code demo')
    ap.add_argument('-r', type=int, default=6)
    ap.add_argument('-s', type=int, default=8)
    ap.add_argument('--rounds', type=int, default=1)
    ap.add_argument('--shots', type=int, default=300)
    ap.add_argument('--noise', type=float, default=0.005)
    ap.add_argument('--bell', action='store_true', help='Run Bell test')
    ap.add_argument('--compare', action='store_true', help='Compare stride-1 vs stride-2')
    args = ap.parse_args()

    if args.compare:
        print("=" * 60)
        print("  HexDecode: stride-1 vs stride-2 comparison")
        print("=" * 60)
        for stride in [2, 1]:
            bench_memory(args.r, args.s, stride, args.rounds, args.shots, args.noise)
        if args.bell:
            for stride in [2, 1]:
                bench_bell(args.r, args.s, stride, args.rounds,
                          args.shots if args.shots <= 300 else 200, args.noise)
    elif args.bell:
        bench_bell(args.r, args.s, 1, args.rounds, args.shots, args.noise)
    else:
        bench_memory(args.r, args.s, 1, args.rounds, args.shots, args.noise)
