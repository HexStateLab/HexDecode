#!/usr/bin/env python3
"""
hexdecode_stride.py — Stride-generalized Bell test for IBM hardware

Supports arbitrary (1+x^g)(1+y^g) polynomials via --stride parameter.
Default stride=1 (standard nearest-neighbor toric code).

For high-rate codes:
  --grid 6 6 --stride 3   → (1+x^3)(1+y^3), 75% rate, d=2
  --grid 12 12 --stride 4 → (1+x^4)(1+y^4), 55.6% rate, d=3
  --grid 20 20 --stride 10 → (1+x^10)(1+y^10), 75% rate, d=2

Build: requires libstridecodec.so in same directory or ../ 
"""

import sys, os, time, argparse, getpass
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'Heron-R2'))
sys.path.insert(0, os.path.join(_here, '..'))
sys.path.insert(0, _here)

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
except ImportError:
    print("qiskit_ibm_runtime not installed"); sys.exit(1)
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from pw_opt import build_circuit, all_syndromes_opt
from hexdecode.stridecodec_bindings import decode as stride_decode
from hexdecode.stridecodec_bindings import is_stabilizer as stride_is_stab
from hexdecode.stridecodec_bindings import params as stride_params
from hexdecode.decoder import virtual_decode


def main():
    ap = argparse.ArgumentParser(description="HexDecode stride-generalized IBM Bell test")
    ap.add_argument("--backend", "-b", type=str, default="ibm_fez")
    ap.add_argument("--grid", type=int, nargs=2, default=(6, 8))
    ap.add_argument("--stride", "-g", type=int, default=1,
                    help="Polynomial stride for (1+x^g)(1+y^g)")
    ap.add_argument("--virtual-stride", "-vg", type=int, default=None,
                    help="Virtual QEC stride from nullspace partitioning. "
                         "E.g. --stride 6 --virtual-stride 3 on 6x6 "
                         "gives 75% rate with d=2, 0 ancilla CX.")
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--token", type=str, default=None)
    opts = ap.parse_args()

    r, s = opts.grid
    g = opts.stride

    if r % g != 0 or s % g != 0:
        print(f"ERROR: r={r}, s={s} not divisible by stride={g}")
        return 1

    k, d, rate = stride_params(r, s, g)
    hr, hs = r // g, s // g
    sectors = g * g
    n_data = r * s
    n_anc = (r - g) * s

    print(f"\n{'='*60}")
    print(f"  HexDecode Stride-{g} Bell  {r}×{s}  rounds={opts.rounds}")
    print(f"  Polynomial: (1+x^{g})(1+y^{g})")
    if opts.virtual_stride and opts.virtual_stride < g:
        vg = opts.virtual_stride
        vk, vd, vr = stride_params(r, s, vg)
        print(f"  Virtual QEC: stride-{g} → stride-{vg}")
        print(f"  Physical: stride-{g} ({k} nullspace, d≈{d}, {rate:.1f}% rate)")
        print(f"  Virtual:  stride-{vg} ({vk} logicals, d≈{vd}, {vr:.1f}% rate)")
        print(f"  Circuit: 0 ancilla CX (virtual syndrome from data)")
    else:
        print(f"  Sectors: {sectors} × ({hr}×{hs})  d≈{d}")
        print(f"  Physical: {n_data}d + {n_anc}a = {n_data+n_anc}q")
        print(f"  Logical Z-operators: {k}  rate: {rate:.1f}%")
    print(f"{'='*60}")

    # ── Backend ──
    if opts.dry_run:
        from qiskit_ibm_runtime.fake_provider.backends.fez.fake_fez import FakeFez
        be = FakeFez()
        print(f"  Dry run on FakeFez")
    else:
        token = opts.token or os.environ.get("IBM_QUANTUM_TOKEN") or \
            getpass.getpass("IBM token: ").strip()
        svc = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
        be = svc.backend(opts.backend)
        print(f"  Backend: {be.name} ({be.num_qubits}q)")

    # ── Build circuits ──
    ff = (opts.rounds == 1)
    use_virtual = (opts.virtual_stride is not None and opts.virtual_stride < g)

    if use_virtual:
        from hexdecode_virtual import build_virtual_circuit
        qzz = build_virtual_circuit(r, s, g, partial_x=False)
        qxx = build_virtual_circuit(r, s, g, partial_x=True)
        n_anc_check = 0
    else:
        kw = dict(logical_state="00", bell=True, bell_ancilla=True,
                  stabilizer_basis="Z", full_stabilizer=False,
                  no_reset=True, periodic=True, compact=True,
                  free_final_round=ff, stride=g)
        qzz, _, lq0, lq1, n_anc_check = build_circuit(r, s, opts.rounds,
                                                        bell_measure=False, **kw)
        qxx, *_ = build_circuit(r, s, opts.rounds,
                                 bell_measure=False, partial_x=True, **kw)

    for lab, qc in [("ZZ", qzz), ("XX", qxx)]:
        ops = qc.count_ops()
        print(f"  {lab}: {qc.num_qubits}q  CX={ops.get('cx',0)}  depth={qc.depth()}")

    # ── Transpile ──
    def _best(qc, seeds=4):
        best_t, best_c = None, float('inf')
        for sd in range(seeds):
            pm = generate_preset_pass_manager(backend=be, optimization_level=3,
                                               seed_transpiler=sd)
            t = pm.run(qc)
            c = sum(v for k, v in t.count_ops().items()
                    if k in ('cz', 'ecr', 'cx'))
            if c < best_c: best_t, best_c = t, c
        return best_t

    print(f"  Transpiling...")
    qzz_t = _best(qzz, 4)
    qxx_t = _best(qxx, 4)
    for lab, t in [("ZZ", qzz_t), ("XX", qxx_t)]:
        ops = t.count_ops()
        cz = sum(v for k, v in ops.items() if k in ('cz', 'ecr', 'cx'))
        sw = ops.get('swap', 0)
        print(f"  {lab}: {t.num_qubits}q  CZ={cz}  SWAP={sw}  depth={t.depth()}")

    if opts.dry_run:
        print("  Dry run complete."); return

    # ── Submit ──
    sampler = SamplerV2(mode=be)
    job = sampler.run([qzz_t, qxx_t], shots=opts.shots)
    print(f"\n  Job: {job.job_id()}")
    print(f"  https://quantum.ibm.com/jobs/{job.job_id()}")
    print("  Waiting...")
    try:
        res = job.result()
    except KeyboardInterrupt:
        print("\n  Detached."); return

    # ── Decode & Witness ──
    shots = opts.shots
    out = {}
    for arm, pub in zip(("ZZ", "XX"), res):
        db = pub.data.data.to_bool_array(order='little').astype(np.uint8)
        data = db.reshape(-1, r, s)[:shots]
        nsh = data.shape[0]

        m = np.zeros(nsh, dtype=np.uint8)
        if hasattr(pub.data, 'bell'):
            m = pub.data.bell.to_bool_array(order='little')[:, 0].astype(np.uint8)[:shots]

        if arm == "ZZ":
            z1 = data[:, 0, :].sum(axis=1) % 2
            z2 = data[:, :, 0].sum(axis=1) % 2
            bits = z1 ^ z2
            framed = bits

            # Decode with stride-generalized decoder (or virtual QEC)
            corr = np.zeros_like(data)
            t0 = time.perf_counter()
            if use_virtual:
                for shot in range(nsh):
                    corr[shot] = virtual_decode(data[shot], r, s, g,
                                                 opts.virtual_stride)
            else:
                syn = all_syndromes_opt(pub, opts.rounds, r, s, n_anc,
                                        no_reset=True, free_final_round=ff,
                                        data_raw=data if ff else None,
                                        full_stabilizer=False, periodic=True, stride=g)
                for shot in range(nsh):
                    corr[shot] = stride_decode(r, s, g, syn[shot, -1])
            dt = time.perf_counter() - t0
            fixed = data ^ corr

            z1c = fixed[:, 0, :].sum(axis=1) % 2
            z2c = fixed[:, :, 0].sum(axis=1) % 2
            raw = 2.0 * (z1 == z2).mean() - 1.0
            corr_val = 2.0 * (z1c == z2c).mean() - 1.0
            print(f"  <ZZ> raw={raw:+.4f}  corrected={corr_val:+.4f}  decode={dt*1e3:.0f}ms")
            bits = z1c ^ z2c
        else:
            xx_par = np.zeros(nsh, dtype=np.uint8)
            for j in range(s): xx_par ^= data[:, 0, j]
            for i in range(r): xx_par ^= data[:, i, 0]
            bits = xx_par
            framed = xx_par ^ m
            raw = 2.0 * (bits == 0).mean() - 1.0
            frame_val = 2.0 * (framed == 0).mean() - 1.0
            print(f"  <XX> raw={raw:+.4f}  frame-corrected={frame_val:+.4f}")

        out[arm] = (bits, framed)

    zz_bits, _ = out["ZZ"]
    xx_bits, xx_framed = out["XX"]
    nz = int((1 - zz_bits).sum())
    nx = int((1 - xx_framed).sum())
    zz = 2.0 * nz / shots - 1.0
    xx = 2.0 * nx / shots - 1.0
    W = zz + xx

    def _ci(ok, N):
        if N < 2: return -1, 1
        p = ok / N; z = 1.96; d = 1 + z * z / N; c = p + z * z / (2 * N)
        m = z * (p * (1 - p) / N + z * z / (4 * N * N)) ** 0.5
        return max(-1, (c - m) / d), min(1, (c + m) / d)

    zz_lo, zz_hi = _ci(nz, shots)
    W_lo = max(-2, 2 * zz_lo - 1 + xx)
    W_hi = min(2, 2 * zz_hi - 1 + xx)

    print(f"\n  W = ⟨Z₁Z₂⟩ + ⟨X₁X₂⟩_frame = {zz:+.4f} + {xx:+.4f} = {W:+.4f}")
    print(f"  95% CI: [{W_lo:.4f}, {W_hi:.4f}]")
    if W_lo > 1.0: print(f"  ✓ ENTANGLED")
    elif W > 1.0: print(f"  ~ marginal")
    else: print(f"  ✗ separable")
    print()


if __name__ == "__main__":
    main()
