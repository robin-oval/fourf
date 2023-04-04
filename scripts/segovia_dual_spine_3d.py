import os

from collections import defaultdict
from itertools import combinations

from math import pi, radians, degrees

from numpy import mean, std

from operator import itemgetter

from compas.datastructures import network_find_cycles, Mesh, Network

from compas.colors import Color, ColorMap
from compas.geometry import Line, Polyline, distance_point_point, length_vector, length_vector_xy, sum_vectors, cross_vectors, rotate_points, bounding_box
from compas.geometry import add_vectors, scale_vector, normalize_vector

from compas_quad.datastructures import CoarseQuadMesh
from compas_quad.datastructures import QuadMesh
from compas_quad.coloring import quad_mesh_polyedge_2_coloring

from fourf import DATA

from fourf.topology import threefold_vault
from fourf.topology import quadmesh_densification
from fourf.support import support_shortest_boundary_polyedges
from fourf.support import polyedge_types
from fourf.sequence import quadmesh_polyedge_assembly_sequence
from fourf.utilities import mesh_to_fdnetwork
from fourf.view import view_f4

from jax_fdm.datastructures import FDNetwork

from jax_fdm.equilibrium import fdm, constrained_fdm
from jax_fdm.optimization import LBFGSB, SLSQP
from jax_fdm.parameters import EdgeForceDensityParameter, NodeSupportXParameter, NodeSupportYParameter
from jax_fdm.parameters import NodeLoadXParameter, NodeLoadYParameter, NodeLoadZParameter

from jax_fdm.goals import NodePointGoal
from jax_fdm.goals import EdgeLengthGoal
from jax_fdm.goals import EdgeDirectionGoal
from jax_fdm.goals import NodeTangentAngleGoal
from jax_fdm.goals import NodeXCoordinateGoal
from jax_fdm.goals import NodeYCoordinateGoal
from jax_fdm.goals import NodeResidualForceGoal
from jax_fdm.goals import NodeZCoordinateGoal
from jax_fdm.goals import NodePlaneGoal
from jax_fdm.goals import NetworkLoadPathGoal
from jax_fdm.goals import EdgesLengthEqualGoal

from jax_fdm.constraints import EdgeLengthConstraint, EdgeForceConstraint, NodeZCoordinateConstraint
from jax_fdm.losses import SquaredError, Loss, PredictionError
from jax_fdm.visualization import Viewer

from compas.utilities import pairwise

from compas.datastructures import mesh_weld

# ==========================================================================
# Parameters
# ==========================================================================

# point load
uniform_load = False
point_load = 0.1

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

# spine description
spine_height = 2.30  # [m]

# optimization
opt = SLSQP  # optimization solver
qmin, qmax = -10.0, 0.0  # bounds on force densities [kN/m]
maxiter = 5000  # maximum number of iterations
tol = 1e-6  # optimization tolerance

# controls
view = True
export = True

# ==========================================================================
# Import datastructures
# ==========================================================================

network = FDNetwork.from_json(os.path.join(DATA, "tripod_network_dual_2d.json"))
mesh = QuadMesh.from_json(os.path.abspath(os.path.join(DATA, "tripod_mesh_dual_2d.json")))

# ==========================================================================
# Modify dual mesh
# ==========================================================================

deletable = []
for fkey in mesh.faces():
    if len(mesh.face_vertices(fkey)) != 4:
        deletable.append(fkey)

assert len(deletable) == 1

for dkey in deletable:
    hexagon = mesh.face_vertices(dkey)
    mesh.delete_face(dkey)

mesh.collect_strips()

for vertices in mesh.vertices_on_boundaries():
    if len(vertices) == 7:
        mesh.add_face(vertices[:-1])

mesh = mesh_weld(mesh)
mesh.collect_polyedges()

# ==========================================================================
# Supports
# ==========================================================================

bdrypkey2length = {}
for pkey, polyedge in mesh.polyedges(data=True):
    if mesh.is_edge_on_boundary(polyedge[0], polyedge[1]):
        bdrypkey2length[pkey] = mesh.polyedge_length(polyedge)
avrg_length = sum(bdrypkey2length.values()) / len(bdrypkey2length)
supported_pkeys = set([pkey for pkey, length in bdrypkey2length.items() if length < avrg_length])
print('supported_pkeys', supported_pkeys)

# support polyedges
support_polyedges = [polyedge for pkey, polyedge in mesh.polyedges(data=True) if pkey in supported_pkeys]

# support nodes
supported_vkeys = set([vkey for pkey in supported_pkeys for vkey in mesh.polyedge_vertices(pkey)])
supports = supported_vkeys

# ==========================================================================
# Spine
# ==========================================================================

# spine polyedges
spine_pkeys = set()
spine_polyedges = []
for pkey, polyedge in mesh.polyedges(data=True):
    if polyedge[0] in supported_vkeys:
        for u, v in pairwise(polyedge):
            cdt0 = mesh.halfedge[u][v] is not None and len(mesh.face_vertices(mesh.halfedge[u][v])) == 6
            cdt1 = mesh.halfedge[v][u] is not None and len(mesh.face_vertices(mesh.halfedge[v][u])) == 6
            if cdt0 or cdt1:
                spine_polyedges.append(polyedge)
                spine_pkeys.add(pkey)
print('spine_pkeys', spine_pkeys)

# spine nodes
spine_nodes = set()
for polyedge in spine_polyedges:
    spine_nodes.update(polyedge)

# spine strips
spine_strip_edges = []
for fkey in mesh.faces():
    if mesh.face_degree(fkey) == 6:
        for u, v in mesh.face_halfedges(fkey):
            strip_edges = mesh.collect_strip(v, u, both_sides=False)
            if strip_edges[-1][-1] in supported_vkeys:
                spine_strip_edges.append(strip_edges)

# spine-split supports polyedge
support_polyedges_spine_split = []
for polyedge in support_polyedges:
    i = 0
    side_a = []
    side_b = []
    for node in polyedge:
        if node in spine_nodes:
            i += 1
        if i <= 1:
            side_a.append(node)
        else:
            side_b.append(node)
    side_a = list(reversed(side_a))
    support_polyedges_spine_split.append([side_a, side_b])

# ==========================================================================
# Profile
# ==========================================================================

# profile polyedges
profile_polyedges = []
for fkey in mesh.faces():
    if len(mesh.face_vertices(fkey)) == 6:
        for u0, v0 in mesh.face_halfedges(fkey):
            for u, v in ([u0, v0], [v0, u0]):
                polyedge = mesh.collect_polyedge(u, v, both_sides=False)
                if polyedge[-1] not in supported_vkeys:
                    profile_polyedges.append(polyedge)

# profile nodes
profile_nodes = set()
for polyedge in profile_polyedges:
    profile_nodes.update(polyedge)

# profile edges
profile_edges = set([edge for polyedge in profile_polyedges for edge in pairwise(polyedge)])

# ==========================================================================
# Assembly sequence
# ==========================================================================

pkey2type = polyedge_types(mesh, supported_pkeys, dual=True)
pkey2step = quadmesh_polyedge_assembly_sequence(mesh, pkey2type)
vkey2step = {vkey: step for pkey, step in pkey2step.items() for vkey in mesh.polyedge_vertices(pkey) if step is not None}
steps = set([step for step in pkey2step.values() if step is not None])
min_step, max_step = int(min(steps)), int(max(steps))

edge2step = {}
for u, v in mesh.edges():
    edge2step[tuple(sorted((u, v)))] = max([vkey2step[u], vkey2step[v]])

step2edges = {step: [] for step in range(min_step, max_step + 1)}
for u, v in mesh.edges():
    step = max([vkey2step[u], vkey2step[v]])
    step2edges[step].append((u, v))

pkey2color = quad_mesh_polyedge_2_coloring(mesh)
color0 = pkey2color[next(iter(spine_pkeys))]
pkeys0 = set([pkey for pkey in mesh.polyedges() if pkey2color[pkey] == color0])
pkeys02step = {pkey: pkey2step[pkey] for pkey in pkeys0}

edges0 = [edge for pkey in mesh.polyedges() for edge in mesh.polyedge_edges(pkey) if pkey2color[pkey] == color0 and edge not in spine_strip_edges and edge not in profile_edges]
edges1 = [edge for pkey in mesh.polyedges() for edge in mesh.polyedge_edges(pkey) if pkey2color[pkey] != color0 and edge not in spine_strip_edges and edge not in profile_edges]

# ==========================================================================
# Update node loads
# ==========================================================================

for node in network.nodes():
    vertex_area = mesh.vertex_area(node)
    load = vertex_area * pz * -1.0
    if uniform_load:
        load = point_load * -1.0
    network.node_load(node, load=[0.0, 0.0, load])

network0 = network.copy()

# ==========================================================================
# Form-finding
# ==========================================================================

network_eq = fdm(network)

# ==========================================================================
# Constrained form-finding of the spine
# ==========================================================================

network_spine = FDNetwork()

# add nodes
for node in spine_nodes:
    x, y, z = network.node_coordinates(node)
    load = network.node_load(node)
    network_spine.add_node(node, x=x, y=y, z=z)
    network_spine.node_load(node, load)
    if node in supports:
        network_spine.node_support(node)

# add edges
for strip in spine_strip_edges:
    for u, v in strip:
        if network_spine.has_edge(u, v) or network_spine.has_edge(v, u):
            continue
        if not network.has_edge(u, v):
            u, v = v, u
        edge = (u, v)
        network_spine.add_edge(u, v)
        q = network.edge_forcedensity(edge)
        network_spine.edge_forcedensity(edge, q)

for polyedge in spine_polyedges:
    for u, v in pairwise(polyedge):
        if network_spine.has_edge(u, v) or network_spine.has_edge(v, u):
            continue
        if not network.has_edge(u, v):
            u, v = v, u
        edge = (u, v)
        network_spine.add_edge(u, v)
        q = network.edge_forcedensity(edge)
        network_spine.edge_forcedensity(edge, q)

# form finding
network_spine_eq = fdm(network_spine)

# ==========================================================================
# Constrained form-finding
# ==========================================================================

# parameters
parameters = []
for edge in network_spine.edges():
    parameter = EdgeForceDensityParameter(edge, -10.0, 0.0)
    parameters.append(parameter)

for node in network_spine.nodes_free():
    for param in [NodeLoadYParameter]:
        parameter = param(node, -0.2, 0.2)
        parameters.append(parameter)

# goals
goals = []
constraints = []
for node in network_spine.nodes_free():
    x, y, z = network_spine.node_coordinates(node)
    for xy, goal in zip((x, y), (NodeXCoordinateGoal, NodeYCoordinateGoal)):
        goals.append(goal(node, target=xy, weight=10.0))

    constraint = NodeZCoordinateConstraint(node, 0.0, spine_height)
    constraints.append(constraint)

for strip in spine_strip_edges:
    for u, v in strip:
        if not network_spine.has_edge(u, v):
            u, v = v, u
        edge = (u, v)
        vector = network_spine.edge_vector(u, v)
        goal = EdgeDirectionGoal(edge, vector)
        goals.append(goal)

# loss
loss = Loss(SquaredError(goals))

network_spine0 = network_spine.copy()
network_spine = constrained_fdm(network_spine,
                                optimizer=opt(),
                                loss=loss,
                                parameters=parameters,
                                maxiter=maxiter,
                                constraints=constraints,
                                tol=1e-9)

network_spine.print_stats()
print(f"\n***Spine max height: {max(network_spine.nodes_attribute(name='z')):.2f}***\n")

# ==========================================================================
# Generate mesh
# ==========================================================================

cycles = network_find_cycles(network_spine)
vertices = {vkey: network_spine.node_coordinates(vkey) for vkey in network_spine.nodes()}
mesh = Mesh.from_vertices_and_faces(vertices, cycles)
mesh.delete_face(0)

# ==========================================================================
# Export
# ==========================================================================

if export:

    print("Setting load y component back to 0.0 for export...")
    ns = network_spine.copy()
    for node in network_spine.nodes():
        py = ns.node_attribute(node, "py")
        ns.node_attribute(node, "py", 0.0)
        # if not network.is_node_support(node):
            # ns.node_attribute(node, "ry", py)

    for name, datastruct in {"network": ns, "mesh": mesh}.items():

        filepath = os.path.join(DATA, f"tripod_{name}_dual_spine_3d.json")
        datastruct.to_json(filepath)

    # add edges
    spine_center = Network()
    for strip in spine_strip_edges:
        nodes = []
        for u, v in strip:
            x, y, z = network_spine.edge_midpoint(u, v)
            node = spine_center.add_node(x=x, y=y, z=z)
            nodes.append(node)
        for u, v in pairwise(nodes):
            spine_center.add_edge(u, v)

    filepath = os.path.join(DATA, f"tripod_network_dual_spine_center_3d.json")
    spine_center.to_json(filepath)
    print("\nExported JSON files!")

# ==========================================================================
# Visualization
# ==========================================================================

if view:

    viewer = Viewer(width=1600, height=900, show_grid=False)
    viewer.view.color = (0.1, 0.1, 0.1, 1)  # change background to black

    viewer.add(network_spine0, as_wireframe=True, show_points=False)
    viewer.add(network_spine, edgecolor="force", edgewidth=(0.01, 0.03))

    viewer.show()
