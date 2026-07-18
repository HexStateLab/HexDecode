# HexDecode — Stride-1 toric code decoder for Heavy-Hex architectures.
#
# License: MIT (c) 2026 HexStateLab
#
# Based on the standard toric code (Kitaev 1997) with stride-1 nearest-neighbor
# checks — native to the degree-3 heavy-hex connectivity of IBM Heron R2.
#
# Key differences from stride-2:
#   - V-check: P(i,j) ⊕ P(i+1,j)  (adjacent, no skip)
#   - Each ancilla touches exactly 2 data qubits → fits degree-2 nodes natively
#   - No flag qubits needed — the ancilla IS the flag
#   - Single-sector decoder (full lattice) — 2× the distance at same grid size
#   - ~20–30× faster decode via compiled C (libplane_s1.so)

from .bindings import (solve_plane, solve_plane_layered, solve_plane_fast,
                        syndrome_of, preprocess_syndrome, is_stabilizer,
                        decode_Z, PlaneWarp, libplane_s1)
from .decoder import (tesseract_decode, tesseract_decode_ffinal, prep,
                       S_of, check_logical, solve, solve_stride,
                       virtual_decode, virtual_params)

__version__ = "1.0.0"
