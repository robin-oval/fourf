import os
from math import fabs
from math import log
from math import log10

from compas.datastructures import Mesh
from compas.datastructures import network_find_cycles

from compas.colors import Color
from compas.colors import ColorMap

from compas.utilities import remap_values

from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import fdm
from jax_fdm.visualization import Viewer

from fourf import DATA


# ==========================================================================
# Import datastructures
# ==========================================================================

FILE_IN = os.path.abspath(os.path.join(DATA, "tripod_network_dual_spine_3d.json"))
# FILE_IN = os.path.abspath(os.path.join(DATA, "tripod_network_dual_3d_full_step_5.json"))
network = FDNetwork.from_json(FILE_IN)

FILE_IN = os.path.abspath(os.path.join(DATA, "tripod_network_dual_3d.json"))
network_final = FDNetwork.from_json(FILE_IN)

# ==========================================================================
# Create 2D projection network for shadow effect
# ==========================================================================

network_2d = network.copy()
network_2d.nodes_attribute("z", 0.0)

# ==========================================================================
# Create mesh for visualization
# ==========================================================================

cycles = network_find_cycles(network)
vertices = {vkey: network.node_coordinates(vkey) for vkey in network.nodes()}
mesh = Mesh.from_vertices_and_faces(vertices, cycles)
mesh.delete_face(0)
mesh.cull_vertices()

# ==========================================================================
# Delete doubly supported edges on the ground, if any
# ==========================================================================

# ztol = 0.1
# deletable = []
# for edge in network.edges():
#     u, v = edge
#     if network.is_node_support(u) and network.is_node_support(v):
#         if network.node_attribute(u, "z") < ztol and network.node_attribute(v, "z") < ztol:
#             deletable.append(edge)

# for u, v in deletable:
#     network.delete_edge(u, v)

# ==========================================================================
# Static equilibrium
# ==========================================================================

# network = fdm(network)
network.print_stats()

# ==========================================================================
# Visualization
# ==========================================================================

edges = list(network.edges())
forces = [fabs(network.edge_force(edge)) for edge in edges]
forces_final = [fabs(network_final.edge_force(edge)) for edge in network_final.edges()]

# edge colors
cmap = ColorMap.from_mpl("viridis")
ratios = remap_values(forces, original_min=0.0, original_max=max(forces_final))
edgecolor = {edge: cmap(ratio) for edge, ratio in zip(edges, ratios)}

# edge widths
width_min, width_max = 0.001, 0.07
widths = remap_values(forces, width_min, width_max, original_min=0.0, original_max=max(forces_final))
edgewidth = {edge: width for edge, width in zip(edges, widths)}

viewer = Viewer(width=1600, height=900, show_grid=False)

viewer.add(network,
           edgecolor=edgecolor,
           edgewidth=edgewidth,
           show_loads=False,
           loadscale=1.0)

viewer.add(network_2d,
           as_wireframe=True,
           show_points=False,
           opacity=0.75)

viewer.add(mesh,
           show_points=False,
           show_lines=False,
           opacity=0.4)

viewer.show()
