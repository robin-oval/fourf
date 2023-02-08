# the essentials
import os
import numpy as np
from scipy.spatial.distance import directed_hausdorff

from itertools import combinations

# compas
from compas.colors import Color
from compas.geometry import Line
from compas.colors import Color, ColorMap
from compas.datastructures import Mesh
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

from jax_fdm.parameters import EdgeForceDensityParameter
from jax_fdm.parameters import NodeAnchorZParameter

from jax_fdm.goals import NodeResidualForceGoal
from jax_fdm.goals import NodePointGoal

from jax_fdm.losses import RootMeanSquaredError, SquaredError, MeanSquaredError, MeanAbsoluteError, AbsoluteError, PredictionError
from jax_fdm.losses import Loss

from jax_fdm.visualization import Viewer

# fourf
from fourf import DATA

# ==========================================================================
# Parameters
# ==========================================================================

# controls
export = False
view = True

# brick hollow properties
brick_hollow_thickness = 0.025  # [m]
brick_hollow_layers = 2  # [-]
brick_hollow_density = 11.0  # [kN/m3]

# brick solid properties
brick_solid_thickness = 0.04  # [m]
brick_solid_layers = 1  # [-]
brick_solid_density = 18.0  # [kN/m3]

# white mortar properties
mortar_thickness = 0.012  # [m]
mortar_layers = 2
mortar_density = 20.0  # [kN/m3]

# vertical area load (approximated self-weight) [kN/m2]
brick_hollow_pz = brick_hollow_density * brick_hollow_thickness * brick_hollow_layers
brick_solid_pz = brick_solid_density * brick_solid_thickness * brick_solid_layers
mortar_pz = mortar_density * mortar_thickness * mortar_layers
pz = brick_hollow_pz + brick_solid_pz + mortar_pz

print(f"Area load: {pz:.2f} [kN/m2] (Brick hollow:  {brick_hollow_pz:.2f} [kN/m2]\tBrick solid:  {brick_solid_pz:.2f} [kN/m2]\tMortar {mortar_pz:.2f} [kN/m2])")

ztol = 0.2

# optimization
qmin, qmax = -20.0, -0.0  # min and max force densities
optimizer = LBFGSB   # the optimization algorithm
error = SquaredError
maxiter = 10000  # optimizer maximum iterations
tol = 1e-6  # optimizer tolerance

# point goal weights
w_xy = 1.0
w_z = 1.0

# ==========================================================================
# Import target mesh
# ==========================================================================

FILE_IN = os.path.abspath(os.path.join(DATA, "tripod_mesh_dual_3d.json"))
mesh = Mesh.from_json(FILE_IN)

FILE_IN = os.path.abspath(os.path.join(DATA, "tripod_network_dual_3d.json"))
network = FDNetwork.from_json(FILE_IN)

# ==========================================================================
# Update network
# ==========================================================================

for node in network.nodes():
    vertex_area = mesh.vertex_area(node)
    network.node_load(node, load=[0.0, 0.0, vertex_area * pz * -1.0])

# ==========================================================================
# Add bracing edges
# ==========================================================================

bracing_edges = []
for fkey in mesh.faces():
    vertices = mesh.face_vertices(fkey)
    if len(vertices) != 4:
        continue
    a, b, c, d = vertices
    edge = network.add_edge(a, c)
    network.edge_forcedensity(edge, -0.1)
    edge = network.add_edge(b, d)
    network.edge_forcedensity(edge, -0.1)

# ==========================================================================
# Define optimization parameters
# ==========================================================================

parameters = []
for edge in network.edges():
    parameter = EdgeForceDensityParameter(edge, qmin, qmax)
    parameters.append(parameter)

spine_nodes = []
# for node in network.nodes_supports():
#     x, y, z = network.node_coordinates(node)
#     if z < 0.1:
#         continue
#     spine_nodes.append(node)
#     parameter = NodeAnchorZParameter(node, z - ztol, z + ztol)
#     parameters.append(parameter)

# ==========================================================================
# Define goals
# ==========================================================================

# edge lengths
goals = []
for node in network.nodes_free():
    point = network.node_coordinates(node)
    goals.append(NodePointGoal(node, target=point))

# for node in spine_nodes:
#     goals.append(NodeResidualForceGoal(node, target=0.0))

# ==========================================================================
# Combine error functions and regularizer into custom loss function
# ==========================================================================

error = error(goals, alpha=1.0)
loss = Loss(error)

# ==========================================================================
# Form-find network
# ==========================================================================

network0 = network.copy()

# ==========================================================================
# Solve constrained form-finding problem
# ==========================================================================

network = constrained_fdm(network,
                          optimizer=optimizer(),
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

if view:
    viewer = Viewer(width=1600, height=900, show_grid=False)

    viewer.view.color = (0.1, 0.1, 0.1, 1)  # change background to black

    # best-fit network
    viewer.add(network,
               edgewidth=(0.001, 0.05),
               edgecolor="force",
               show_loads=False,
               loadscale=1.0)

    # target network
    viewer.add(network0,
               as_wireframe=True,
               show_points=False,
               linewidth=1.0,
               color=Color.grey().darkened())

    # draw lines between best-fit and target networks
    for node in network.nodes():
        pt = network.node_coordinates(node)
        line = Line(pt, network0.node_coordinates(node))
        viewer.add(line, color=Color.grey())

    # show le crème
    viewer.show()
