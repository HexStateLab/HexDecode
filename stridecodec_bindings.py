"""
stridecodec_bindings.py — Python ctypes wrapper for libstridecodec.so

Stride-generalized (1+x^g)(1+y^g) polynomial codec.
Supports arbitrary stride g where r%g==0 and s%g==0.

For (1+x^10)(1+y^10) on 20×20: 300 logical Z-operators, 50% total rate.
"""
import ctypes, os, numpy as np

_lib = None

def _load():
    global _lib
    if _lib is not None: return _lib
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ["libstridecodec.so", "../libstridecodec.so"]:
        path = os.path.join(here, name)
        if os.path.exists(path):
            _lib = ctypes.cdll.LoadLibrary(path)
            break
    if _lib is None:
        raise RuntimeError("libstridecodec.so not found")
    _lib.stride_decode.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.POINTER(ctypes.c_uint8),
                                    ctypes.POINTER(ctypes.c_uint8)]
    _lib.stride_decode.restype = ctypes.c_int
    _lib.stride_syndrome.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                      ctypes.POINTER(ctypes.c_uint8),
                                      ctypes.POINTER(ctypes.c_uint8)]
    _lib.stride_syndrome.restype = None
    _lib.stride_is_stabilizer.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                           ctypes.POINTER(ctypes.c_uint8)]
    _lib.stride_is_stabilizer.restype = ctypes.c_int
    return _lib


def decode(r, s, stride, syndrome):
    """Decode (r,s) syndrome for (1+x^stride)(1+y^stride) code."""
    lib = _load()
    syn = np.ascontiguousarray(syndrome, dtype=np.uint8).flatten()
    out = np.zeros(r * s, dtype=np.uint8)
    lib.stride_decode(r, s, stride,
                       syn.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                       out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
    return out.reshape(r, s)


def syndrome_of(r, s, stride, error):
    """Compute syndrome from error pattern."""
    lib = _load()
    err = np.ascontiguousarray(error, dtype=np.uint8).flatten()
    syn = np.zeros(r * s, dtype=np.uint8)
    lib.stride_syndrome(r, s, stride,
                         err.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                         syn.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
    return syn.reshape(r, s)


def is_stabilizer(r, s, stride, correction):
    """True if correction is a stabilizer (no logical error)."""
    lib = _load()
    corr = np.ascontiguousarray(correction, dtype=np.uint8).flatten()
    return bool(lib.stride_is_stabilizer(r, s, stride,
                corr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))))


def params(r, s, stride):
    """Return (k_logical, distance, rate_percent) for the code."""
    hr, hs = r // stride, s // stride
    k = stride * stride * (hr + hs - 1)
    d = min(hr, hs)
    if d < 2: d = 2
    rate = k / (r * s) * 100
    return k, d, rate
