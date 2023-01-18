
from math import pi, cos, sin

from compas.datastructures import Mesh
from compas_singular.datastructures import QuadMesh, CoarseQuadMesh

from compas.topology import adjacency_from_edges, dijkstra_distances

from compas.geometry import midpoint_point_point, circle_from_points
from compas.geometry import Line

from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import fdm, constrained_fdm
from jax_fdm.optimization import LBFGSB

from jax_fdm.parameters import EdgeForceDensityParameter

from jax_fdm.goals import EdgeLengthGoal, NodeTangentAngleGoal, NodeXCoordinateGoal, NodeYCoordinateGoal

from jax_fdm.losses import SquaredError, Loss

from compas.utilities import pairwise

from compas_view2.app import App

### INPUTS ###

r = 10.0 # circumcircle radius [m]
pos_angles = 0.0, 3 * pi / 4, 3 * pi / 2 # triangle parameterisation angle [radians]
wid_angles = pi / 12, pi / 12, pi / 12 # triangle to hexagon parameterisation angle [radians]
target_edge_length = 1.0

pz = -10.0 # total vertical load
q0 = -1.0

opt = LBFGSB
qmin, qmax = None, -1e-3
maxiter = 1000
tol = 1e-6

add_edge_length_goal = False
factor_length = 1.0
weight_edge_length_goal = 0.1

add_vertex_tangent_goal = False
angle = 30 / 180 * 2 * pi
weight_vertex_tangent_goal = 1.0

add_projection_goal = True
weight_projection_goal = 1.0

view = True

### COARSE MESH ###

vertices = []
for a, da in zip(pos_angles, wid_angles):
    for angle in [a - da / 2, a + da / 2]:
        vertices.append([r * cos(angle), r * sin(angle), 0.0])
vertices.append([0.0, 0.0, 0.0])

k = len(vertices) - 1
corners = vertices[:-1]
for i in range(k):
    vertices.insert(2 * i + 1, midpoint_point_point(corners[i], corners[(i + 1) % k]))
# print('vertices', len(vertices), vertices)

g, _, _ = circle_from_points(vertices[3], vertices[7], vertices[11])
vertices[-1] = g

n = len(vertices)
faces = []
for i in range(k):
    faces.append([n - 1, (2 * i + 1) % (n - 1), (2 * i + 2) % (n - 1), (2 * i + 3) % (n - 1)])
# print('faces', faces)

### DENSE MESH ###

mesh = CoarseQuadMesh.from_vertices_and_faces(vertices, faces)
mesh.collect_strips()
mesh.set_strips_density_target(target_edge_length)
mesh.densification()
mesh = mesh.get_quad_mesh()
mesh.collect_polyedges()

### SUPPORTS ###

pkey2type = {pkey: None for pkey in mesh.polyedges()}

# supports
supports = set()
threshold_length = sum([mesh.edge_length(u, v) for u, v in pairwise(mesh.vertices_on_boundary())]) / 6
for pkey, polyedge in mesh.polyedges(data=True):
    if mesh.is_edge_on_boundary(polyedge[0], polyedge[1]):
        length = sum([mesh.edge_length(u, v) for u, v in pairwise(polyedge)])
        if length < threshold_length:
            pkey2type[pkey] = 'support'
            supports.update(polyedge)
# spine
for pkey, polyedge in mesh.polyedges(data=True):
    start, end = polyedge[0], polyedge[-1]
    cdt0 = start in supports and mesh.vertex_degree(end) == 6
    cdt1 = end in supports and mesh.vertex_degree(start) == 6
    if cdt0 or cdt1:
        pkey2type[pkey] = 'spine'
# span
for pkey, polyedge in mesh.polyedges(data=True):
    if pkey2type[pkey] is None:
        if polyedge[0] in supports and polyedge[-1] in supports:
            pkey2type[pkey] = 'span'
# cantilever
for pkey in mesh.polyedges():
    if pkey2type[pkey] is None:
        pkey2type[pkey] = 'cantilever'


### SEQUENCE ###

pkey2step = {pkey: None for pkey in mesh.polyedges()}
for pkey, ptype in pkey2type.items():
    if ptype == 'support':
        pkey2step[pkey] = -1
    elif ptype == 'spine':
        pkey2step[pkey] = 0

vkey2subpkey = {}
for pkey, polyedge in mesh.polyedges(data=True):
    if pkey2type[pkey] == 'spine' or pkey2type[pkey] == 'span':
        for vkey in polyedge:
            vkey2subpkey[vkey] = pkey
# print(vkey2subpkey)

pkey_adjacency_edges = set()
for pkey, polyedge in mesh.polyedges(data=True):
    if pkey2type[pkey] == 'spine' or pkey2type[pkey] == 'span':
        for vkey in polyedge:
            for nbr in mesh.vertex_neighbors(vkey):
                pkey2 = vkey2subpkey[nbr]
                if pkey != pkey2:
                    if pkey2type[pkey2] == 'spine' or pkey2type[pkey2] == 'span':
                        a, b = min(pkey, pkey2), max((pkey, pkey2))
                        pkey_adjacency_edges.add((a, b))
pkey_adjacency_edges = set([tuple(pkey if not pkey2type[pkey] == 'spine' else -1 for pkey in pkeys) for pkeys in pkey_adjacency_edges])
pkey_adjacency_edges = set([(min(pkeys), max(pkeys)) for pkeys in pkey_adjacency_edges if pkeys[0] != pkeys[1]])
# print(pkey_adjacency_edges)

adjacency = adjacency_from_edges(pkey_adjacency_edges)
weights = {edge: 1.0 for edge in pkey_adjacency_edges}
weights.update({tuple(reversed(edge)): 1.0 for edge in pkey_adjacency_edges})
target = -1
for pkey, step in dijkstra_distances(adjacency, weights, target).items():
    if pkey != -1:
        pkey2step[pkey] = step
max_step = max([step for step in pkey2step.values() if step is not None])
# print('max_step', max_step)

### SMOOTH ###
mesh.smooth_centroid(fixed=supports, kmax=10)
# mesh.smooth_centroid(fixed=[vkey for vkey in mesh.vertices_on_boundary() if mesh.vertex_degree(vkey) == 2], kmax=5)
# mesh.smooth_centroid(fixed=mesh.vertices_on_boundary(), kmax=100)

### FORM FINDING ###

# data
network = FDNetwork()

for vkey in mesh.vertices():
    x, y, z = mesh.vertex_coordinates(vkey)
    network.add_node(x=x, y=y, z=z, key=vkey)

for u, v in mesh.edges():
    if u not in supports or v not in supports:
        network.add_edge(u, v)

for node in supports:
    network.node_support(node)

mesh_area = mesh.area()
for node in network.nodes():
    vertex_area = mesh.vertex_area(node)
    network.node_load(node, load=[0.0, 0.0, vertex_area / mesh_area * pz])

for edge in network.edges():
    network.edge_forcedensity(edge, q0)

network = fdm(network)

### FORM OPTIMIZATION ###

# STEP 1 #

parameters = []

for edge in network.edges():
    parameter = EdgeForceDensityParameter(edge, qmin, qmax)
    parameters.append(parameter)

# # edge length goal
# goals_length = []
# target_length = factor_length * sum([network.edge_length(*edge) for edge in network.edges()]) / network.number_of_edges()
# for edge in network.edges():
#     goal = EdgeLengthGoal(edge, target=target_length, weight=weight_edge_length_goal)
#     goals_length.append(goal)

# projection
goals_projection = []
for node in mesh.vertices_on_boundary():
    if node not in supports:
        x, y, z = network.node_coordinates(node)
        for goal, coord in ((NodeXCoordinateGoal, x), (NodeYCoordinateGoal, y)):
            goal = goal(node, coord)
            goals_projection.append(goal)

# # vertex tangent goal
# goals_tangent = []
# for node in network.nodes():
#     goal = NodeTangentAngleGoal(node, vector=[0.0, 0.0, 1.0], target=angle, weight=weight_vertex_tangent_goal)
#     goals_tangent.append(goal)

loss = Loss(
            # SquaredError(goals=goals_length, name='EdgeLengthGoal', alpha=1.0),
            SquaredError(goals=goals_projection, name='NodeCoordinateGoal', alpha=1.0),
            # SquaredError(goals=goals_tangent, name='NodeTangentAngleGoal', alpha=1.0),
            )

# network = constrained_fdm(network,
#                              optimizer=opt(),
#                              parameters=parameters,
#                              loss=loss,
#                              maxiter=maxiter,
#                              tol=tol,
#                              )

# STEP 2 #

parameters = []

for edge in network.edges():
    parameter = EdgeForceDensityParameter(edge, qmin, qmax)
    parameters.append(parameter)

# edge length goal
goals_length = []
target_length = sum([network.edge_length(*edge) for edge in network.edges()]) / network.number_of_edges()
for edge in network.edges():
    goal = EdgeLengthGoal(edge, target=target_length, weight=weight_edge_length_goal)
    goals_length.append(goal)

# projection
goals_projection = []
for node in mesh.vertices_on_boundary():
    if node not in supports:
        x, y, z = network.node_coordinates(node)
        for goal, coord in ((NodeXCoordinateGoal, x), (NodeYCoordinateGoal, y)):
            goal = goal(node, coord)
            goals_projection.append(goal)

# # vertex tangent goal
# goals_tangent = []
# for node in network.nodes():
#     goal = NodeTangentAngleGoal(node, vector=[0.0, 0.0, 1.0], target=angle, weight=weight_vertex_tangent_goal)
#     goals_tangent.append(goal)

# loss = Loss(
#             SquaredError(goals=goals_length, name='EdgeLengthGoal', alpha=1.0),
#             SquaredError(goals=goals_projection, name='NodeCoordinateGoal', alpha=0.1),
#             # SquaredError(goals=goals_tangent, name='NodeTangentAngleGoal', alpha=1.0),
#             )

# network = constrained_fdm(network,
#                              optimizer=opt(),
#                              parameters=parameters,
#                              loss=loss,
#                              maxiter=maxiter,
#                              tol=tol,
#                              )

### VIEWER ###

if view:
    viewer = App(width=1600, height=900, show_grid=False)

    ptype2color = {
        'support': [1.0, 0.0, 0.0],
        'spine': [1.0, 0.0, 1.0],
        'span': [0.0, 0.0, 1.0],
        'cantilever': [0.0, 0.0, 0.0],
    }

    edge2color = {}
    for pkey, polyedge in mesh.polyedges(data=True):
        ptype = pkey2type[pkey]
        color = ptype2color[ptype]
        if ptype == 'span':
            step = pkey2step[pkey]
            lamba = step / max_step
            color = [1 - lamba, 0, lamba]
        for u, v in pairwise(polyedge):
            edge2color[(u, v)] = color

    for edge, color in edge2color.items():
        viewer.add(Line(*mesh.edge_coordinates(*edge)), linecolor=color)
    
    for edge in network.edges():
        viewer.add(Line(*network.edge_coordinates(*edge)))

    # viewer.add(mesh)
    # viewer.add(network)
    
    viewer.show()






