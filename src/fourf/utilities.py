from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from jax_fdm.datastructures import FDNetwork


__all__ = []

def mesh_to_fdnetwork(mesh, supported_vkeys, area_load, init_forcedensity):

	network = FDNetwork()

	for vkey in mesh.vertices():
	    x, y, z = mesh.vertex_coordinates(vkey)
	    network.add_node(x=x, y=y, z=z, key=vkey)

	for u, v in mesh.edges():
	    if u not in supported_vkeys or v not in supported_vkeys:
	        network.add_edge(u, v)

	for node in supported_vkeys:
	    network.node_support(node)

	mesh_area = mesh.area()
	for node in network.nodes():
	    vertex_area = mesh.vertex_area(node)
	    network.node_load(node, load=[0.0, 0.0, vertex_area * area_load])

	for edge in network.edges():
	    network.edge_forcedensity(edge, init_forcedensity)

	return network