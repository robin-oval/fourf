from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from compas.utilities import pairwise

__all__ = []


def support_shortest_boundary_polyedges(quadmesh):
	bdrypkey2length = {}
	for pkey, polyedge in quadmesh.polyedges(data=True):
	    if quadmesh.is_edge_on_boundary(polyedge[0], polyedge[1]):
	        bdrypkey2length[pkey] = sum([quadmesh.edge_length(u, v) for u, v in pairwise(polyedge)])
	avrg_length = sum(bdrypkey2length.values()) / len(bdrypkey2length)
	return [pkey for pkey, length in bdrypkey2length.items() if length < avrg_length]

def polyedge_types(quadmesh, supported_pkeys):

	supports = [vkey for pkey in supported_pkeys for vkey in quadmesh.polyedge_vertices(pkey)]

	pkey2type = {pkey: None for pkey in quadmesh.polyedges()}

	# support
	for pkey in supported_pkeys:
	    pkey2type[pkey] = 'support'

	# spine
	for pkey, polyedge in quadmesh.polyedges(data=True):
		if not quadmesh.is_edge_on_boundary(*polyedge[:2]):
		    start, end = polyedge[0], polyedge[-1]
		    cdt0 = start in supports and quadmesh.is_vertex_singular(end)
		    cdt1 = end in supports and quadmesh.is_vertex_singular(start)
		    if cdt0 or cdt1:
		        pkey2type[pkey] = 'spine'
	# span
	for pkey, polyedge in quadmesh.polyedges(data=True):
	    if pkey2type[pkey] is None:
	        if polyedge[0] in supports and polyedge[-1] in supports:
	            pkey2type[pkey] = 'span'
	# cantilever
	for pkey in quadmesh.polyedges():
	    if pkey2type[pkey] is None:
	        pkey2type[pkey] = 'cantilever'

	return pkey2type
