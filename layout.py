#!/usr/bin/env python3
"""
hexdecode_layout.py — Optimal HeavyHex layout for stride-1 toric code.

Stride-1 V-check: V(i,j) = P(i,j) ⊕ P(i+1,j) — two adjacent data qubits.

On heavy-hex (IBM Heron R2), the lattice has:
  - Degree-3 nodes (hexagon corners) → data qubits
  - Degree-2 nodes (hexagon edges between adjacent layers) → ancillas

The optimal layout places each ancilla on the degree-2 edge BETWEEN
its two data qubits in the coupling graph. This eliminates all routing
overhead: every logical CX maps to exactly one physical CZ.
"""

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


def _neighbors_of(coupling_map, q):
    """All qubits adjacent to q in the coupling map."""
    nb = set()
    for a, b in coupling_map:
        if a == q: nb.add(b)
        elif b == q: nb.add(a)
    return nb


def stride1_heavyhex_layout(backend, r, s, stride=1):
    """Find an optimal initial_layout for stride-1 on HeavyHex.

    Strategy: data qubits go on degree-3+ nodes. Each ancilla goes on
    a degree-2 node that is adjacent to BOTH its data-qubit neighbors
    in the coupling map. This ensures 1:1 CX→CZ mapping.

    Returns initial_layout dict {logical_qubit_index: physical_qubit_index}
    or None if no embedding found.
    """
    coupling = backend.coupling_map
    if coupling is None:
        return None

    edges = [(a, b) for a, b in coupling]
    # Build adjacency
    adj = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    n_data = r * s
    n_anc = (r - stride) * s
    total = n_data + n_anc

    phys_nodes = sorted(adj.keys())
    if len(phys_nodes) < total:
        return None

    # Classify by degree: data → deg≥3, ancilla → deg=2
    deg3 = sorted([p for p in phys_nodes if len(adj[p]) >= 3])
    deg2 = sorted([p for p in phys_nodes if len(adj[p]) == 2])

    # We need n_data degree-3 nodes and n_anc degree-2 nodes
    # arranged such that each ancilla is adjacent to its 2 data qubits
    if len(deg3) < n_data or len(deg2) < n_anc:
        return None

    # For stride=1: data at (i,j), ancilla at (i,j) between data(i,j) and data(i+1,j)
    # Each column j forms a chain: d[0,j] — a[0,j] — d[1,j] — a[1,j] — ... — d[r-1,j]
    # Each edge in the chain must be a physical edge in the coupling map.

    # Greedy: for each column j, try to find a degree-3 → degree-2 → degree-3 chain
    layout = {}
    used = set()

    def find_chain_pair(start_d, start_a, target_d):
        """Find physical qubits d1-a-d2 forming a 3-qubit chain where
        d1 is adjacent to a, and a is adjacent to d2."""
        for a in deg2:
            if a in used: continue
            nb = adj[a]
            if start_d in nb:
                # a is adjacent to start_d; find another degree-3 neighbor
                for d2 in nb:
                    if d2 == start_d: continue
                    if d2 in used: continue
                    if d2 in deg3:
                        return a, d2
        return None, None

    # Build chains column by column
    for j in range(s):
        # First data qubit in column j
        d0 = (0, j)
        d0_idx = d0[0] * s + d0[1]

        # Find a degree-3 node for d0
        for p0 in deg3:
            if p0 in used: continue
            layout[d0_idx] = p0
            used.add(p0)
            break
        else:
            return None

        # Chain downward: for each row i, find ancilla a between d[i] and d[i+1]
        for i in range(r - stride):
            ai = (i, j)
            ai_idx = n_data + i * s + j
            di1 = (i + stride, j)
            di1_idx = di1[0] * s + di1[1]

            curr_d = layout[d0_idx + i * stride * s]  # phys index of current data
            if curr_d is None: return None

            a_phys, d_next = find_chain_pair(curr_d, None, None)
            if a_phys is None and d_next is None:
                # Try alternative: any degree-2 adjacent to curr_d
                for a in deg2:
                    if a in used: continue
                    if curr_d in adj[a]:
                        # Find degree-3 neighbor of a (not curr_d)
                        for d2 in adj[a]:
                            if d2 != curr_d and d2 in deg3 and d2 not in used:
                                a_phys, d_next = a, d2
                                break
                        if a_phys is not None:
                            break
                if a_phys is None:
                    return None

            layout[ai_idx] = a_phys
            used.add(a_phys)

            if i < r - stride - 1:
                # More rows below: need d_next to be data qubit
                pass
            layout.setdefault(di1_idx, None)
            if layout[di1_idx] is None:
                layout[di1_idx] = d_next
                used.add(d_next)

    # Fill remaining bell ancilla and extra qubits
    extra_start = n_data + n_anc
    for extra_idx in range(extra_start, total if total > extra_start else extra_start):
        for p in phys_nodes:
            if p not in used:
                layout[extra_idx] = p
                used.add(p)
                break

    if len(layout) < total:
        # Try a simpler approach: let the transpiler handle it
        return None

    return list(layout.get(i, 0) for i in range(total))


def transpile_with_layout(qc, backend, layout=None, seeds=8):
    """Transpile with best Sabre seed, optionally with initial_layout."""
    best_t, best_c = None, None
    for sd in range(max(1, seeds)):
        kw = {}
        if layout is not None:
            kw['initial_layout'] = layout
        pm = generate_preset_pass_manager(
            backend=backend, optimization_level=3, seed_transpiler=sd, **kw)
        t = pm.run(qc)
        c = sum(v for k, v in t.count_ops().items()
                if k in ('cz', 'ecr', 'cx', 'swap'))
        if best_c is None or c < best_c:
            best_t, best_c = t, c
    return best_t
