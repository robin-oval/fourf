#!/usr/bin/env python3

from __future__ import division
from __future__ import print_function

from compas.topology import adjacency_from_edges, dijkstra_distances


__all__ = []


def quadmesh_polyedge_assembly_sequence(quadmesh, pkey2type):
    """
    """
    pkey2step = {pkey: None for pkey in quadmesh.polyedges()}

    for pkey, ptype in pkey2type.items():
        if ptype == 'support':
            pkey2step[pkey] = -1
        elif ptype == 'spine':
            pkey2step[pkey] = 0

    vkey2subpkey = {}
    for pkey, polyedge in quadmesh.polyedges(data=True):
        if pkey2type[pkey] == 'spine' or pkey2type[pkey] == 'span':
            for vkey in polyedge:
                vkey2subpkey[vkey] = pkey

    pkey_adjacency_edges = set()
    for pkey, polyedge in quadmesh.polyedges(data=True):
        if pkey2type[pkey] == 'spine' or pkey2type[pkey] == 'span':
            for vkey in polyedge:
                for nbr in quadmesh.vertex_neighbors(vkey):
                    pkey2 = vkey2subpkey[nbr]
                    if pkey != pkey2:
                        if pkey2type[pkey2] == 'spine' or pkey2type[pkey2] == 'span':
                            a, b = min(pkey, pkey2), max((pkey, pkey2))
                            pkey_adjacency_edges.add((a, b))
    pkey_adjacency_edges = set([tuple(pkey if not pkey2type[pkey] == 'spine' else -1 for pkey in pkeys) for pkeys in pkey_adjacency_edges])
    pkey_adjacency_edges = set([(min(pkeys), max(pkeys)) for pkeys in pkey_adjacency_edges if pkeys[0] != pkeys[1]])

    adjacency = adjacency_from_edges(pkey_adjacency_edges)
    weights = {edge: 1.0 for edge in pkey_adjacency_edges}
    weights.update({tuple(reversed(edge)): 1.0 for edge in pkey_adjacency_edges})
    target = -1
    for pkey, step in dijkstra_distances(adjacency, weights, target).items():
        if pkey != -1:
            pkey2step[pkey] = step

    return pkey2step
