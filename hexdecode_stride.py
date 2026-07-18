#!/usr/bin/env python3
"""
hexdecode_stride.py — Stride-generalized Bell test with exponential virtual QEC.
"""
import sys, os, time, argparse, getpass, numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'Heron-R2'))
sys.path.insert(0, os.path.join(_here, '..'))
sys.path.insert(0, os.path.join(_here, 'hexdecode'))

from hexdecode_virtual import build_virtual_circuit, gauge_graph_zz_all, gauge_graph_xx
from stridecodec_bindings import params as stride_params

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", "-b", type=str, default="ibm_fez")
    ap.add_argument("--grid", type=int, nargs=2, default=(6, 6))
    ap.add_argument("--stride", "-g", type=int, default=6)
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--rounds", type=int, default=1,
                    help="Bell ancilla measurement rounds in circuit")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--token", type=str, default=None)
    opts = ap.parse_args()

    r, s = opts.grid; g = opts.stride
    k, d, rate = stride_params(r, s, g)
    vops = 2 ** (r * s)
    
    print(f"\n{'='*60}")
    print(f"  HexDecode Exponential Virtual QEC  {r}×{s}  stride={g}")
    print(f"  Nullspace: 2^{r*s} ≈ {vops:.2e} virtual Z-operators")
    print(f"  Physical: {r*s}d + 0a = {r*s}q  rate={rate:.0f}%  d≈{d}")
    print(f"{'='*60}")

    qc_zz = build_virtual_circuit(r, s, g, partial_x=False, rounds=opts.rounds)
    use_partial_x = (opts.rounds == 1)
    qc_xx = build_virtual_circuit(r, s, g, partial_x=use_partial_x, rounds=opts.rounds)
    for lab, qc in [("ZZ", qc_zz), ("XX", qc_xx)]:
        ops = qc.count_ops()
        print(f"  {lab}: {qc.num_qubits}q  CX={ops.get('cx',0)}  depth={qc.depth()}")

    if opts.dry_run:
        from qiskit_ibm_runtime.fake_provider.backends.fez.fake_fez import FakeFez
        be = FakeFez()
    else:
        token = opts.token or os.environ.get("IBM_QUANTUM_TOKEN") or getpass.getpass("IBM: ")
        svc = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
        be = svc.backend(opts.backend)

    def _best(qc):
        bt, bc = None, float('inf')
        for sd in range(4):
            t = generate_preset_pass_manager(backend=be, optimization_level=3, seed_transpiler=sd).run(qc)
            c = sum(v for k, v in t.count_ops().items() if k in ('cz', 'ecr', 'cx'))
            if c < bc: bt, bc = t, c
        return bt

    qzz_t = _best(qc_zz); qxx_t = _best(qc_xx)
    for lab, t in [("ZZ", qzz_t), ("XX", qxx_t)]:
        ops = t.count_ops()
        cz = sum(v for k, v in ops.items() if k in ('cz', 'ecr', 'cx'))
        print(f"  {lab}: {t.num_qubits}q  CZ={cz}  SWAP={ops.get('swap',0)}  depth={t.depth()}")

    if opts.dry_run:
        print("  Dry run complete."); return

    job = SamplerV2(mode=be).run([qzz_t, qxx_t], shots=opts.shots)
    print(f"\n  Job: {job.job_id()}\n  Waiting...")
    res = job.result()

    out = {}
    for arm, pub in zip(("ZZ", "XX"), res):
        db = pub.data.data.to_bool_array(order='little').astype(np.uint8)
        data = db.reshape(-1, r, s)[:opts.shots]
        nsh = data.shape[0]
        m = np.zeros(nsh, dtype=np.uint8)
        if hasattr(pub.data, 'bell'):
            m = pub.data.bell.to_bool_array(order='little')[:, 0].astype(np.uint8)[:nsh]

        if arm == "ZZ":
            z1_std = data[:, 0, :].sum(1) % 2
            z2_std = data[:, :, 0].sum(1) % 2
            z1_vq = np.zeros(nsh, dtype=np.uint8)
            z2_vq = np.zeros(nsh, dtype=np.uint8)
            t0 = time.perf_counter()
            for shot in range(nsh):
                z1_vq[shot], z2_vq[shot] = gauge_graph_zz_all(data[shot])
            dt = time.perf_counter() - t0
            zz_std = 2.0 * (z1_std == z2_std).mean() - 1.0
            zz_vq  = 2.0 * (z1_vq == z2_vq).mean() - 1.0
            n_checks = r*s*(r*s-1)//2
            print(f"  <ZZ> std={zz_std:+.4f}  gauge_graph={zz_vq:+.4f}  "
                  f"({n_checks} virtual checks, {dt*1e3:.0f}ms)")
            bits = z1_vq ^ z2_vq
            framed = bits
        else:
            # XX: from bell_measure ancilla bits (rounds>1) or partial_x (rounds=1)
            if opts.rounds > 1 and hasattr(pub.data, 'bell_m0'):
                xx_bits = np.zeros(nsh, dtype=np.uint8)
                for rnd in range(opts.rounds - 1):
                    reg = getattr(pub.data, f'bell_m{rnd}')
                    bm = reg.to_bool_array(order='little')[:, 0].astype(np.uint8)[:nsh]
                    xx_bits ^= bm ^ m  # XOR bell_meas with bell_prep, accumulate
                xx_val = 2.0 * (xx_bits == 0).mean() - 1.0
                print(f"  <XX> bell_measure={xx_val:+.4f}  ({opts.rounds-1} rounds)")
                bits = xx_bits; framed = xx_bits
            else:
                xx_std = np.zeros(nsh, dtype=np.uint8)
                xx_gg  = np.zeros(nsh, dtype=np.uint8)
                for shot in range(nsh):
                    xxp = 0
                    for j in range(s): xxp ^= data[shot, 0, j]
                    for i in range(r): xxp ^= data[shot, i, 0]
                    xx_std[shot] = xxp ^ m[shot]
                    xx_gg[shot]  = gauge_graph_xx(data[shot], m[shot])
                xx_std_val = 2.0 * (xx_std == 0).mean() - 1.0
                xx_gg_val  = 2.0 * (xx_gg == 0).mean() - 1.0
                print(f"  <XX> std={xx_std_val:+.4f}  gauge={xx_gg_val:+.4f}")
                bits = xx_gg; framed = xx_gg
        out[arm] = (bits, framed)

    zz_bits, _ = out["ZZ"]
    xx_bits, xx_framed = out["XX"]
    nz = int((1 - zz_bits).sum()); nx = int((1 - xx_framed).sum())
    zz = 2.0 * nz / opts.shots - 1.0; xx = 2.0 * nx / opts.shots - 1.0
    W = zz + xx

    def _ci(ok, N):
        if N < 2: return -1, 1
        p = ok / N; z = 1.96; d = 1 + z * z / N; c = p + z * z / (2 * N)
        m = z * (p * (1 - p) / N + z * z / (4 * N * N)) ** 0.5
        return max(-1, (c - m) / d), min(1, (c + m) / d)

    zz_lo, zz_hi = _ci(nz, opts.shots)
    W_lo = max(-2, 2 * zz_lo - 1 + xx)
    W_hi = min(2, 2 * zz_hi - 1 + xx)
    print(f"\n  W = ⟨Z₁Z₂⟩ + ⟨X₁X₂⟩_frame = {zz:+.4f} + {xx:+.4f} = {W:+.4f}")
    print(f"  95% CI: [{W_lo:.4f}, {W_hi:.4f}]")
    print(f"  {'✓ ENTANGLED' if W_lo > 1.0 else '~ marginal' if W > 1.0 else '✗ separable'}")


if __name__ == "__main__":
    main()
