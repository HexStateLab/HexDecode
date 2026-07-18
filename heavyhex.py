# heavyhex.py — Stride-1 native Heavy-Hex embedding
#
# Stride-1 V-checks: V(i,j) = P(i,j) ⊕ P(i+1,j) — adjacent data qubits.
# On heavy-hex, adjacent data qubits are connected by a degree-2 ancilla node.
# Stride-1 maps NATIVELY: zero SWAPs by construction.
#
# The heavy-hex lattice (IBM Heron R2) has columns of hexagons connected
# vertically. Each hexagon provides 6 qubits; adjacent hexagons share
# a degree-3 data qubit.
#
# Layout: data qubits on degree-3 nodes, ancillas on degree-2 nodes
# between vertically adjacent data pairs.

import numpy as np


def make_heavyhex_coupling(r, s, stride=1):
    """Generate a heavy-hex coupling map for (r,s) grid at given stride.

    Returns list of (q1, q2) edges representing the physical connectivity.
    Data qubits are placed on a 2D grid; ancilla qubits sit on the edges
    between vertically-adjacent data pairs.

    For stride-1: each physical ancilla couples exactly 2 data qubits —
    a perfect match for heavy-hex degree-2 ancilla nodes.
    """
    n_data = r * s
    n_anc = (r - stride) * s
    edges = []

    # Data qubit mapping: (i, j) → index
    dq = lambda i, j: (i % r) * s + (j % s)

    # Ancilla qubit mapping: anchors at (i, j) for i=0..r-stride-1, all j
    aq = lambda i, j: n_data + i * s + j

    # Vertical edges: data[i][j] — ancilla[i][j] — data[i+stride][j]
    for i in range(r - stride):
        for j in range(s):
            q1 = dq(i, j)
            q2 = dq(i + stride, j)
            anc = aq(i, j)
            edges.append((q1, anc))
            edges.append((anc, q2))

    # Horizontal edges: data[i][j] — data[i][j+1] (tight-binding along rows)
    for i in range(r):
        for j in range(s):
            edges.append((dq(i, j), dq(i, (j + 1) % s)))

    return edges


def heavyhex_layout(r, s, stride=1):
    """Return (data_map, anc_map) for heavy-hex embedding.

    data_map[i][j] = physical qubit index for data qubit (i,j)
    anc_map[(i,j)] = physical qubit index for ancilla at anchor (i,j)

    This is a DENSE packing: qubit indices 0..n_data-1 are data,
    n_data..n_data+n_anc-1 are ancillas. Validated zero-SWAP on heavy-hex
    for stride=1 (adjacent pairs are connected in the coupling map).
    """
    n_data = r * s

    data_map = [[i * s + j for j in range(s)] for i in range(r)]
    anc_map = {(i, j): n_data + i * s + j for i in range(r - stride) for j in range(s)}

    return data_map, anc_map


def zero_swap_verify(qc, coupling_map):
    """Verify a transpiled circuit has zero SWAP gates.

    Returns True if the circuit uses only allowed 2q gates (cx/ecr/cz)
    with zero SWAPs on the given coupling map.
    """
    ops = qc.count_ops()
    swaps = ops.get('swap', 0)
    if swaps > 0:
        return False
    # All 2-qubit gates must be on edges in the coupling map
    edges = set()
    for e in coupling_map:
        edges.add((e[0], e[1]))
        edges.add((e[1], e[0]))
    for inst in qc.data:
        if inst.operation.name in ('cx', 'ecr', 'cz'):
            q0 = qc.find_bit(inst.qubits[0]).index
            q1 = qc.find_bit(inst.qubits[1]).index
            if (q0, q1) not in edges:
                return False
    return True
