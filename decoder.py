# decoder.py — Stride-1 decoder pipeline for HexDecode
#
# Adapts the pattern from Heron-R2/decoder.py (MIT licensed).
# Uses our stride-1 C library instead of stride-2 plane_warp binary.
# Stride-1 has a single-sector kernel with r+s-1 nullspace dim (no sector
# decomposition needed), making the decoder ~20× faster.

import numpy as np

from .bindings import (solve_plane as _c_solve, preprocess_syndrome as _c_prep,
                         syndrome_of as _c_syn_of, is_stabilizer as _c_is_stab)

# Stride-generalized decoder
try:
    from .stridecodec_bindings import decode as _stride_decode
    from .stridecodec_bindings import syndrome_of as _stride_syndrome
    from .stridecodec_bindings import is_stabilizer as _stride_is_stab
    from .stridecodec_bindings import params as _stride_params
    _has_stride = True
except (ImportError, OSError):
    _has_stride = False


def prep(syn, r, s):
    """Preprocess syndrome in-place (measurement fault repair)."""
    _c_prep(r, s, syn)


def solve(syn, r, s):
    """Minimum-weight correction from syndrome via C decoder."""
    return _c_solve(r, s, syn)


def solve_stride(syn, r, s, stride):
    """Decode with (1+x^stride)(1+y^stride) polynomial codec."""
    if _has_stride:
        return _stride_decode(r, s, stride, syn)
    raise RuntimeError("libstridecodec.so not available")


def virtual_decode(data, r, s, physical_stride, virtual_stride):
    """Virtual QEC: reconstruct virtual syndrome from data readout.
    
    For codes where physical_stride ≥ virtual_stride, the nullspace
    of the physical code contains the virtual code structure.
    Syndrome is reconstructed via:
      V[i][j] = data[i][j] ⊕ data[(i+g)%r][j]
      S[i][j] = V[i][j] ⊕ V[i][(j+g)%s]
    where g = virtual_stride.
    
    No ancilla qubits needed — syndrome comes from data correlations.
    """
    import numpy as np
    r2, s2 = r, s
    g = virtual_stride
    data = np.asarray(data, dtype=np.uint8).reshape(r2, s2)
    V = data ^ np.roll(data, shift=-g, axis=0)
    S = V ^ np.roll(V, shift=-g, axis=1)
    return _stride_decode(r2, s2, g, S.astype(np.uint8))


def virtual_params(r, s, physical_g, virtual_g):
    """Return effective code parameters for virtual QEC."""
    return _stride_params(r, s, virtual_g)


def S_of(E, r, s):
    """Compute stride-1 syndrome from error pattern."""
    return _c_syn_of(r, s, E)


def check_logical(corr, r, s):
    """True if correction is a stabilizer (no logical error)."""
    return _c_is_stab(r, s, corr)


def tesseract_decode_ffinal(syndromes, r, s):
    """Single-shot decode: last round's syndrome → correction."""
    syn = syndromes[-1].copy().astype(np.uint8)
    prep(syn, r, s)
    return solve(syn, r, s)


def tesseract_decode(syndromes, r, s):
    """Multi-round AND-vote + fallback to last round."""
    rr = syndromes.shape[0]

    # AND-vote: intersect all rounds (persistent errors survive)
    syn_and = np.ones((r, s), dtype=np.uint8)
    for t in range(rr):
        syn_and &= syndromes[t]

    # Check viability: all row and column parity must be even
    viable = True
    for i in range(r):
        if syn_and[i].sum() % 2:
            viable = False
            break
    if viable:
        for j in range(s):
            if syn_and[:, j].sum() % 2:
                viable = False
                break

    syn = syn_and.copy() if viable else syndromes[-1].copy()
    prep(syn, r, s)
    return solve(syn, r, s)


def _translation_grid(r, s):
    """Diverse grid-matched translations for rotation-enumeration decode."""
    r4, s4 = max(1, r // 4), max(1, s // 4)
    return [(dx % r, dy % s) for dx, dy in [
        (0, 0), (r4, 0), (0, s4), (r4, s4),
        (r4 * 2, s4 * 2), (r // 2, 0), (0, s // 2)
    ]]


def translate_syn(syn, dx, dy):
    return np.roll(syn, shift=(-dx, -dy), axis=(0, 1))


def untranslate_corr(corr, dx, dy):
    return np.roll(corr, shift=(dx, dy), axis=(0, 1))


def tesseract_decode_rot(syndromes, r, s):
    """Best-translation decode: try shifts, pick minimum-weight valid."""
    syn = syndromes[-1].copy().astype(np.uint8)
    best_wt = r * s + 1
    best_corr = np.zeros((r, s), dtype=np.uint8)

    for dx, dy in _translation_grid(r, s):
        syn_t = translate_syn(syn, dx, dy)
        prep(syn_t, r, s)
        corr_t = solve(syn_t, r, s)
        corr = untranslate_corr(corr_t, dx, dy)
        if not check_logical(corr, r, s):
            continue
        wt = int(corr.sum())
        if wt < best_wt:
            best_wt = wt
            best_corr = corr

    return best_corr
