# bindings.py — ctypes wrapper for libplane_s1.so (stride-1 decoder)
#
# Adapts the pattern from Heron-R2/pw_qiskit.py (MIT licensed) to stride-1.

import ctypes
import os
import numpy as np

# Load the stride-1 shared library
_lib_dir = os.path.dirname(os.path.abspath(__file__))
_lib_path = os.path.join(_lib_dir, "..", "libplane_s1.so")
_lib = ctypes.CDLL(_lib_path)

# Expose the library handle for globals access
libplane_s1 = _lib

# -- set argument / return types for each exported C function --

# void preprocess_syndrome(int r, int s, uint8_t *syn)
_lib.preprocess_syndrome.argtypes = [ctypes.c_int, ctypes.c_int,
                                      ctypes.POINTER(ctypes.c_uint8)]
_lib.preprocess_syndrome.restype = None

# void syndrome_of(int r, int s, uint8_t *err, uint8_t *syn)
_lib.syndrome_of.argtypes = [ctypes.c_int, ctypes.c_int,
                              ctypes.POINTER(ctypes.c_uint8),
                              ctypes.POINTER(ctypes.c_uint8)]
_lib.syndrome_of.restype = None

# int solve_plane(int r, int s, uint8_t *syn, uint8_t *out)
_lib.solve_plane.argtypes = [ctypes.c_int, ctypes.c_int,
                              ctypes.POINTER(ctypes.c_uint8),
                              ctypes.POINTER(ctypes.c_uint8)]
_lib.solve_plane.restype = ctypes.c_int

# int solve_plane_layered(int r, int s, uint8_t *syn, uint8_t *out)
_lib.solve_plane_layered.argtypes = [ctypes.c_int, ctypes.c_int,
                                      ctypes.POINTER(ctypes.c_uint8),
                                      ctypes.POINTER(ctypes.c_uint8)]
_lib.solve_plane_layered.restype = ctypes.c_int

# int solve_plane_fast(int r, int s, uint8_t *syn, uint8_t *out)
_lib.solve_plane_fast.argtypes = [ctypes.c_int, ctypes.c_int,
                                   ctypes.POINTER(ctypes.c_uint8),
                                   ctypes.POINTER(ctypes.c_uint8)]
_lib.solve_plane_fast.restype = ctypes.c_int

# void canonicalize(int r, int s, uint8_t *corr)
_lib.canonicalize.argtypes = [ctypes.c_int, ctypes.c_int,
                               ctypes.POINTER(ctypes.c_uint8)]
_lib.canonicalize.restype = None

# int is_stabilizer(int r, int s, uint8_t *diff)
_lib.is_stabilizer.argtypes = [ctypes.c_int, ctypes.c_int,
                                ctypes.POINTER(ctypes.c_uint8)]
_lib.is_stabilizer.restype = ctypes.c_int

# int decode_Z(int r, int s, uint8_t *err_z, uint8_t *dec_z)
_lib.decode_Z.argtypes = [ctypes.c_int, ctypes.c_int,
                           ctypes.POINTER(ctypes.c_uint8),
                           ctypes.POINTER(ctypes.c_uint8)]
_lib.decode_Z.restype = ctypes.c_int


class PlaneWarp:
    """ctypes wrapper for the stride-1 plane_warp decoder."""

    def __init__(self, fast=False, singleshot=True, escape=True,
                 weight_cap=0, cap_auto_rate=0.0):
        self.fast = fast
        # Stride-1 decoder globals (silently skip if not exposed)
        try:
            g_fast = ctypes.c_int.in_dll(_lib, "g_fast")
            g_fast.value = 1 if fast else 0
        except ValueError:
            pass
        try:
            g_singleshot = ctypes.c_int.in_dll(_lib, "g_singleshot")
            g_singleshot.value = 1 if singleshot else 0
        except ValueError:
            pass
        try:
            g_escape = ctypes.c_int.in_dll(_lib, "g_escape_enabled")
            g_escape.value = 1 if escape else 0
        except ValueError:
            pass
        try:
            cap = ctypes.c_int.in_dll(_lib, "g_weight_cap")
            cap.value = weight_cap
        except ValueError:
            pass
        try:
            cap_r = ctypes.c_double.in_dll(_lib, "g_cap_auto_rate")
            cap_r.value = cap_auto_rate
        except ValueError:
            pass

    def solve(self, syn, r, s):
        out = np.zeros(r * s, dtype=np.uint8)
        syn_arr = np.ascontiguousarray(syn, dtype=np.uint8).flatten()
        _lib.solve_plane(r, s,
                         syn_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                         out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
        return out.reshape(r, s)

    def preprocess(self, syn, r, s):
        syn_arr = np.ascontiguousarray(syn, dtype=np.uint8).flatten()
        _lib.preprocess_syndrome(r, s,
                                  syn_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
        return syn_arr.reshape(r, s)

    def syndrome_of(self, E, r, s):
        out = np.zeros(r * s, dtype=np.uint8)
        err_arr = np.ascontiguousarray(E, dtype=np.uint8).flatten()
        _lib.syndrome_of(r, s,
                          err_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                          out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
        return out.reshape(r, s)

    def is_stabilizer(self, corr, r, s):
        corr_arr = np.ascontiguousarray(corr, dtype=np.uint8).flatten()
        return bool(_lib.is_stabilizer(r, s,
                    corr_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))))

    def decode(self, syndromes, r, s):
        syn = syndromes[-1].copy().astype(np.uint8)
        self.preprocess(syn, r, s)
        return self.solve(syn, r, s)


# Convenience functions
def solve_plane(r, s, syn):
    syn_arr = np.ascontiguousarray(syn, dtype=np.uint8).flatten()
    out = np.zeros(r * s, dtype=np.uint8)
    _lib.solve_plane(r, s,
                     syn_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                     out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
    return out.reshape(r, s)


def solve_plane_layered(r, s, syn):
    syn_arr = np.ascontiguousarray(syn, dtype=np.uint8).flatten()
    out = np.zeros(r * s, dtype=np.uint8)
    _lib.solve_plane_layered(r, s,
                              syn_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                              out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
    return out.reshape(r, s)


def solve_plane_fast(r, s, syn):
    syn_arr = np.ascontiguousarray(syn, dtype=np.uint8).flatten()
    out = np.zeros(r * s, dtype=np.uint8)
    _lib.solve_plane_fast(r, s,
                           syn_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                           out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
    return out.reshape(r, s)


def syndrome_of(r, s, err):
    err_arr = np.ascontiguousarray(err, dtype=np.uint8).flatten()
    out = np.zeros(r * s, dtype=np.uint8)
    _lib.syndrome_of(r, s,
                      err_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                      out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
    return out.reshape(r, s)


def preprocess_syndrome(r, s, syn):
    syn_arr = np.ascontiguousarray(syn, dtype=np.uint8).flatten()
    _lib.preprocess_syndrome(r, s,
                              syn_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
    return syn_arr.reshape(r, s)


def is_stabilizer(r, s, diff):
    diff_arr = np.ascontiguousarray(diff, dtype=np.uint8).flatten()
    return bool(_lib.is_stabilizer(r, s,
                diff_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))))


def decode_Z(r, s, err_z):
    err_arr = np.ascontiguousarray(err_z, dtype=np.uint8).flatten()
    out = np.zeros(r * s, dtype=np.uint8)
    _lib.decode_Z(r, s,
                   err_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                   out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
    return out.reshape(r, s)
