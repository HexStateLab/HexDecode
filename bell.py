# bell.py — Stride-1 Bell witness for HexDecode
#
# Adapts the pattern from Heron-R2/bell_complex.py (MIT licensed).
# Uses stride-1 circuits with our C decoder for ~20× faster decode.

import sys, os, random, numpy as np, time

from .decoder import prep, solve, S_of, check_logical


def data_syndrome(data, r, s, stride=1):
    """Compute stride-1 stabilizer syndrome from data readout.
    V(i,j) = E[i][j] ⊕ E[(i+stride)%r][j]
    S(i,j) = V(i,j) ⊕ V(i,(j+stride)%s)
    """
    g = stride
    V = data.astype(np.uint8) ^ np.roll(data.astype(np.uint8), shift=-g, axis=0)
    return V ^ np.roll(V, shift=-g, axis=1)


def _wilson_ci(ok, N, z=1.96):
    if N < 2:
        return -1, 1
    p = ok / N
    d = 1 + z * z / N
    c = p + z * z / (2 * N)
    m = z * (p * (1 - p) / N + z * z / (4 * N * N)) ** 0.5
    return max(-1, (c - m) / d), min(1, (c + m) / d)


def bell_witness(zz_bits, xx_bits):
    """Compute CHSH witness W = ⟨Z₁Z₂⟩ + ⟨X₁X₂⟩ from measurement bits.

    zz_bits: array of (z1 ^ z2) per shot (1 = anti-correlated)
    xx_bits: array of (bell_meas ^ bell_prep) per shot (1 = flipped)
    Returns (W, W_lo, W_hi, ZZ, XX, zz_ci_lo, zz_ci_hi)
    """
    shots = len(zz_bits)
    n_zz_ok = int((1 - zz_bits).sum())  # shots where Z1 == Z2
    n_xx_ok = int((1 - xx_bits).sum())  # shots where parity preserved

    zz = 2.0 * n_zz_ok / shots - 1.0
    xx = 2.0 * n_xx_ok / shots - 1.0

    zz_lo, zz_hi = _wilson_ci(n_zz_ok, shots)
    W = zz + xx
    W_lo = max(-2, 2 * zz_lo - 1 + xx)
    W_hi = min(2, 2 * zz_hi - 1 + xx)

    return W, W_lo, W_hi, zz, xx, zz_lo, zz_hi


def decode_all_shots(syndromes, r, s, stride=1, rot_decode=False):
    """Decode all shots. syndromes: (shots, rounds, r, s).

    Returns (corrections, n_ok) where corrections is (shots, r, s) and
    n_ok is number of valid (stabilizer) corrections.
    """
    shots = syndromes.shape[0]
    corr = np.zeros((shots, r, s), dtype=np.uint8)
    n_ok = 0

    for shot in range(shots):
        syn_last = syndromes[shot, -1].copy().astype(np.uint8)

        if rot_decode:
            # Try shift-enumeration: pick minimum-weight valid correction
            from .decoder import tesseract_decode_rot
            c = tesseract_decode_rot(syndromes[shot], r, s)
        else:
            prep(syn_last, r, s)
            c = solve(syn_last, r, s)

        corr[shot] = c
        if check_logical(c, r, s):
            n_ok += 1

    return corr, n_ok
