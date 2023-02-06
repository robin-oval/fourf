# the essentials
import os
import numpy as np
from scipy.spatial.distance import directed_hausdorff

from itertools import combinations

# compas
from compas.colors import Color
from compas.geometry import Line
from compas.colors import Color, ColorMap
from compas.geometry import Line, Polyline, distance_point_point, length_vector, length_vector_xy, sum_vectors, cross_vectors, rotate_points, bounding_box
from compas.geometry import angle_vectors, project_point_plane
from compas.geometry import add_vectors, subtract_vectors, scale_vector, normalize_vector
from compas.utilities import pairwise

from compas_quad.datastructures import QuadMesh
from compas_quad.coloring import quad_mesh_polyedge_2_coloring

# jax fdm
from jax_fdm.datastructures import FDNetwork

from jax_fdm.equilibrium import fdm
from jax_fdm.equilibrium import constrained_fdm

from jax_fdm.optimization import SLSQP, LBFGSB
from jax_fdm.optimization import OptimizationRecorder

from jax_fdm.parameters import EdgeForceDensityParameter, NodeAnchorXParameter, NodeAnchorYParameter

from jax_fdm.goals import EdgeLengthGoal, NodePlaneGoal, EdgeDirectionGoal, EdgesLengthEqualGoal
from jax_fdm.goals import NodePointGoal, NodeYCoordinateGoal, NodeZCoordinateGoal, NodeXCoordinateGoal, NetworkLoadPathGoal

from jax_fdm.losses import RootMeanSquaredError, SquaredError, MeanSquaredError, MeanAbsoluteError, AbsoluteError, PredictionError
from jax_fdm.losses import Loss

from jax_fdm.visualization import LossPlotter
from jax_fdm.visualization import Viewer

from fourf import DATA
from fourf.utilities import mesh_to_fdnetwork
from fourf.topology import threefold_vault
from fourf.topology import quadmesh_densification
from fourf.support import support_shortest_boundary_polyedges
from fourf.support import polyedge_types
from fourf.sequence import quadmesh_polyedge_assembly_sequence

# ==========================================================================
# Parameters
# ==========================================================================


brick_length, brick_width, brick_thickness = 0.225, 0.1, 0.03  # [m]
brick_layers = 3  # [-]
brick_density = 16.5  # [kN/m3]

dead_load = 0.0  # additional dead load [kN/m2]
pz = brick_density * brick_thickness * brick_layers + dead_load  # vertical area load (approximated self-weight + uniform dead load) [kN/m2]
print("pz", pz)
error = MeanSquaredError

qmin, qmax = -20.0, -0.0  # min and max force densities
optimizer = LBFGSB  # SLSQP  # the optimization algorithm
maxiter = 1000  # optimizer maximum iterations
tol = 1e-6  # optimizer tolerance

# point constraint weights
w_start = 10.0  # 10
w_end = 1.0  # 1
w_factor_boundary = 1.0  # 0.25

w_xy = 1.0
w_z = 5.0

export = True  # export result to JSON

# ==========================================================================
# Import target mesh
# ==========================================================================

FILE_IN = os.path.abspath(os.path.join(DATA, "tripod_mesh_dual_3d.json"))
mesh = QuadMesh.from_json(FILE_IN)
mesh.collect_strips()
mesh.collect_polyedges()

FILE_IN = os.path.abspath(os.path.join(DATA, "tripod_network_dual_3d.json"))
network = FDNetwork.from_json(FILE_IN)

# ==========================================================================
# Update network
# ==========================================================================

for node in network.nodes():
    vertex_area = mesh.vertex_area(node)
    network.node_load(node, load=[0.0, 0.0, vertex_area * pz * -1.0])

# ==========================================================================
# Define optimization parameters
# ==========================================================================

parameters = []
for edge in network.edges():
    parameter = EdgeForceDensityParameter(edge, qmin, qmax)
    parameters.append(parameter)

# ==========================================================================
# Define goals
# ==========================================================================

# edge lengths
goals = []

# NOTE: non-spine nodes best fit XYZ
for node in network.nodes_free():
    x, y, z = network.node_coordinates(node)
    goals.append(NodeXCoordinateGoal(node, target=x, weight=w_xy))
    goals.append(NodeYCoordinateGoal(node, target=y, weight=w_xy))
    goals.append(NodeZCoordinateGoal(node, target=z, weight=w_z))


# for polyedge in span_polyedges_split + spine_polyedges:
#     n = len(polyedge)
#     for i, node in enumerate(polyedge):

#         weight = w_start + (w_end - w_start) * (i / (n - 1))
#         if mesh.is_vertex_on_boundary(node):
#             weight = weight * w_factor_boundary
#         point = network.node_coordinates(node)
#         goals.append(NodePointGoal(node, target=point, weight=weight))

# ==========================================================================
# Combine error functions and regularizer into custom loss function
# ==========================================================================

error = error(goals, alpha=1.0)
loss = Loss(error)

# ==========================================================================
# Form-find network
# ==========================================================================

network0 = network.copy()
# network = fdm(network)
# network_fd = network.copy()

# print(f"Load path: {round(network.loadpath(), 3)}")

# network = fdm(network)

# ==========================================================================
# Solve constrained form-finding problem
# ==========================================================================

optimizer = optimizer()

network = constrained_fdm(network,
                          optimizer=optimizer,
                          loss=loss,
                          parameters=parameters,
                          maxiter=maxiter,
                          tol=tol)

# ==========================================================================
# Delete edges at the supports
# ==========================================================================

# deletable = []
# for u, v in network.edges():
#     if u in supports and v in supports:
#         deletable.append((u, v))

# for u, v in deletable:
#     network.delete_edge(u, v)

# network = fdm(network)

# ==========================================================================
# Export JSON
# ==========================================================================

if export:
    filepath = os.path.join(DATA, "tripod_network_bestfit_subd_3d.json")
    network.to_json(filepath)
    for vkey in mesh.vertices():
        xyz = network.node_coordinates(vkey)
        mesh.vertex_attributes(vkey, "xyz", xyz)
    filepath = os.path.join(DATA, "tripod_mesh_bestfit_subd_3d.json")
    mesh.to_json(filepath)
    print("\nExported network and mesh JSON files!")

# ==========================================================================
# Hausdorff distance
# ==========================================================================

U = np.array([network0.node_coordinates(node) for node in network0.nodes()])
V = np.array([network.node_coordinates(node) for node in network.nodes()])
directed_u = directed_hausdorff(U, V)[0]
directed_v = directed_hausdorff(V, U)[0]
hausdorff = max(directed_u, directed_v)

print(f"Hausdorff distances: U: {directed_u:.2f}\tV: {directed_v:.2f}\tUV: {round(hausdorff, 2)}")

# ==========================================================================
# Report stats
# ==========================================================================

network.print_stats()

# ==========================================================================
# Visualization
# ==========================================================================

viewer = Viewer(width=1600, height=900, show_grid=False)

# modify view
# viewer.view.camera.zoom(-35)  # number of steps, negative to zoom out
viewer.view.camera.rotation[2] = 0.0  # set rotation around z axis to zero

# color strip edges according to sequence
# maxs = max_step
# cmap = ColorMap.from_mpl("viridis")
# for i in range(maxs):

#     for strip in profile_strips[i]:
#         for edge in strip:
#             line = Line(*network.edge_coordinates(*edge))
#             viewer.add(line, linecolor=cmap(i, maxval=maxs), linewidth=5.)

#         for polyedge in zip(*strip):
#             polyline = Polyline([network.node_coordinates(n) for n in polyedge])
#             viewer.add(polyline)

# viewer.add(mesh, opacity=0.5, show_points=False)

# optimized network
viewer.add(network,
           edgewidth=(0.01, 0.03),
           edgecolor="force",
           loadscale=1.0)

# # reference network
viewer.add(network0,
           as_wireframe=True,
           show_points=False,
           linewidth=1.0,
           color=Color.grey().darkened())

# draw lines to target
for node in network.nodes():
    pt = network.node_coordinates(node)
    line = Line(pt, network0.node_coordinates(node))
    viewer.add(line, color=Color.grey())

# show le crème
viewer.show()
