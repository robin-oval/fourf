# the essentials
import os
from math import fabs

# scientific computing
import numpy as np
from scipy.spatial.distance import directed_hausdorff

# compas
from compas.colors import Color
from compas.geometry import Line
from compas.geometry import add_vectors
from compas.geometry import scale_vector
from compas.geometry import normalize_vector
from compas.geometry import dot_vectors
from compas.utilities import pairwise

# quadness
from compas_quad.datastructures import QuadMesh
from compas_quad.coloring import quad_mesh_polyedge_2_coloring

# construction stability
from fourf import DATA
from fourf.support import polyedge_types
from fourf.sequence import quadmesh_polyedge_assembly_sequence

# jax fdm
from jax_fdm.datastructures import FDNetwork

from jax_fdm.equilibrium import constrained_fdm

from jax_fdm.optimization import LBFGSB

from jax_fdm.parameters import EdgeForceDensityParameter
from jax_fdm.parameters import NodeSupportXParameter
from jax_fdm.parameters import NodeSupportYParameter

from jax_fdm.goals import NodePointGoal
from jax_fdm.goals import EdgeLengthGoal

from jax_fdm.losses import SquaredError
from jax_fdm.losses import L2Regularizer
from jax_fdm.losses import Loss

from jax_fdm.visualization import Viewer


# ==========================================================================
# Parameters
# ==========================================================================

# controls
export = False
view = True

# network modifiers
add_bracing = True
add_bracing_to_spine = True
delete_supported_edges = False
offset_normal = 0.0  # 0.025 / 2 (to offset mesh based on the normal of final geometry)

# brick hollow properties
brick_hollow_thickness = 0.025  # [m]
brick_hollow_layers = 2  # 2 [-]
brick_hollow_density = 11.0  # [kN/m3]

# brick solid properties
brick_solid_thickness = 0.04  # [m]
brick_solid_layers = 1  # 1 [-]
brick_solid_density = 18.0  # [kN/m3]

# white mortar properties
mortar_thickness = 0.012  # 0.015, [m]
mortar_layers = 2  # 2
mortar_density = 19.0  # 19 [kN/m3]

# Additional dead load
super_dead_pz = 0.0  # [kN/m2]

# vertical area load (approximated self-weight) [kN/m2]
brick_hollow_pz = brick_hollow_density * brick_hollow_thickness * brick_hollow_layers
brick_solid_pz = brick_solid_density * brick_solid_thickness * brick_solid_layers
mortar_pz = mortar_density * mortar_thickness * mortar_layers
pz = brick_hollow_pz + brick_solid_pz + mortar_pz + super_dead_pz

print(f"Area load: {pz:.2f} [kN/m2] (Brick hollow:  {brick_hollow_pz:.2f} [kN/m2]\tBrick solid:  {brick_solid_pz:.2f} [kN/m2]\tMortar {mortar_pz:.2f} [kN/m2]\tExtra dead{super_dead_pz:.2f} [kN/m2])")

# sequential form finding
start_step = 1
max_step_sequential = 5  # 5
max_step_sequential_short = 3  # 3

# optimization
qmin, qmax = -20.0, -1e-8  # min and max force densities
optimizer = LBFGSB   # the optimization algorithm
error_fn = SquaredError  # error term in the loss function
maxiter = 10000  # optimizer maximum iterations
tol = 1e-8  # optimizer tolerance

# update force densities of original network after every optimization step
update_qs_after_optimization = True

# add supports to the optimization parameters list
parametrize_supports = False
xytol = 0.025  # the amount supports can wiggle on x and y from their current position

weight_goals_per_step = True  # prioritize xyz matching of vertices at current step
alpha_xyz = 1.0  # target xyz goal
alpha_length = 0.0  # target lengths goal
alpha_reg = 0.0  # regularizer

# ==========================================================================
# Import target mesh
# ==========================================================================

FILE_IN = os.path.abspath(os.path.join(DATA, "tripod_mesh_dual_3d.json"))
mesh = QuadMesh.from_json(FILE_IN)

FILE_IN = os.path.abspath(os.path.join(DATA, "tripod_network_dual_3d.json"))
network = FDNetwork.from_json(FILE_IN)

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

# spine edges
spine_polyedge_edges = set()
for polyedge in spine_polyedges:
    for edge in pairwise(polyedge):
        spine_polyedge_edges.add(edge)

# spine strips
spine_strip_edges = []
spine_strip_edges_edges = set()
for fkey in mesh.faces():
    if mesh.face_degree(fkey) == 6:
        for u, v in mesh.face_halfedges(fkey):
            strip_edges = mesh.collect_strip(v, u, both_sides=False)
            if strip_edges[-1][-1] in supported_vkeys:
                spine_strip_edges.append(strip_edges)
                spine_strip_edges_edges.update(strip_edges)

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

# profile strips
profile_strips = []
for fkey in mesh.faces():
    if mesh.face_degree(fkey) == 6:
        for u, v in mesh.face_halfedges(fkey):
            strip_edges = mesh.collect_strip(v, u, both_sides=False)
            if strip_edges[-1][-1] not in supported_vkeys:
                profile_strips.append(strip_edges)
assert len(profile_strips) == 3

profile_polyedges_from_strips = []
profile_span_short_polyedges = []

for strip in profile_strips:
    polyedges = list(zip(*strip))
    profile_polyedges_from_strips.append(polyedges)

    test_edge = polyedges[0][:2]
    vr = normalize_vector(mesh.edge_vector(*test_edge))
    dot = dot_vectors(vr, [1., 0., 0.])
    if fabs(1.0 - dot) < 0.1:
        profile_span_short_polyedges.extend(polyedges)

assert len(profile_span_short_polyedges) == 2
profile_edges_span_short = set([edge for polyedge in profile_span_short_polyedges for edge in pairwise(polyedge)])

# ==========================================================================
# Assembly sequence
# ==========================================================================

pkey2type = polyedge_types(mesh, supported_pkeys, dual=True)
pkey2step = quadmesh_polyedge_assembly_sequence(mesh, pkey2type)
vkey2step = {vkey: step for pkey, step in pkey2step.items() for vkey in mesh.polyedge_vertices(pkey) if step is not None and step >= 0}
steps = set([step for step in pkey2step.values() if step is not None])
min_step, max_step = int(min(steps)), int(max(steps))

edge2step = {}
for u, v in mesh.edges():
    edge2step[tuple(sorted((u, v)))] = max([vkey2step[u], vkey2step[v]])

step2edges = {step: [] for step in range(min_step, max_step + 1)}
for u, v in mesh.edges():
    step = max([vkey2step[u], vkey2step[v]])
    step2edges[step].append((u, v))

pkey2color = quad_mesh_polyedge_2_coloring(mesh, skip_singularities=False)
color0 = pkey2color[next(iter(spine_pkeys))]
pkeys0 = set([pkey for pkey in mesh.polyedges() if pkey2color[pkey] == color0])
pkeys02step = {pkey: pkey2step[pkey] for pkey in pkeys0}

edges0 = [edge for pkey in mesh.polyedges() for edge in mesh.polyedge_edges(pkey) if pkey2color[pkey] == color0 and edge not in spine_strip_edges and edge not in profile_edges]
edges1 = [edge for pkey in mesh.polyedges() for edge in mesh.polyedge_edges(pkey) if pkey2color[pkey] != color0 and edge not in spine_strip_edges and edge not in profile_edges]

# ==========================================================================
# Short span
# ==========================================================================

edges_span_short = set()
nodes_span_short = set()
for polyedge in profile_span_short_polyedges:
    for u, v in pairwise(polyedge):
        strip = mesh.collect_strip(u, v)
        for edge in strip:
            edges_span_short.add(edge)
            for node in edge:
                nodes_span_short.add(node)

for pkey, polyedge in mesh.polyedges(True):
    if pkey2type[pkey] != "span":
        continue
    if any((node in nodes_span_short for node in polyedge)):
        edges = list(pairwise(polyedge))
        edges_span_short.update(edges)

# ==========================================================================
# Offset mesh
# ==========================================================================

if offset_normal:

    for vkey in mesh.vertices():
        xyz = mesh.vertex_coordinates(vkey)
        normal = normalize_vector(mesh.vertex_normal(vkey))
        xyz_new = add_vectors(xyz, scale_vector(normal, offset_normal))
        mesh.vertex_attributes(vkey, "xyz", xyz_new)
        network.node_attributes(vkey, "xyz", xyz_new)

# ==========================================================================
# Update node loads
# ==========================================================================

for node in network.nodes():
    vertex_area = mesh.vertex_area(node)
    network.node_load(node, load=[0.0, 0.0, vertex_area * pz * -1.0])

# ==========================================================================
# Add bracing edges
# ==========================================================================

bracing_edges = []
if add_bracing:
    print("\nAdding bracing edges")
    for fkey in mesh.faces():
        vertices = mesh.face_vertices(fkey)

        # skip faces in spine
        if not add_bracing_to_spine:
            if all(vkey2step[vkey] == 0 for vkey in vertices):
                continue

        if len(vertices) != 4:
            continue

        a, b, c, d = vertices

        edge = network.add_edge(a, c)
        network.edge_attribute(edge, "type", "bracing")
        network.edge_forcedensity(edge, -0.1)

        edge = network.add_edge(b, d)
        network.edge_attribute(edge, "type", "bracing")
        network.edge_forcedensity(edge, -0.1)

# ==========================================================================
# Delete supported edges
# ==========================================================================

if delete_supported_edges:
    print("\nDeleting supported edges")
    deletable = []
    for edge in network.edges():
        u, v = edge
        if network.is_node_support(u) and network.is_node_support(v):
            deletable.append(edge)

    for u, v in deletable:
        network.delete_edge(u, v)

# ==========================================================================
# Change of variables
# ==========================================================================

network0 = network.copy()

# ==========================================================================
# Sequential constrained form-finding
# ==========================================================================

for step in range(start_step, max_step_sequential + 1):

    print(f"\n***Step: {step}***")

    network = network0.copy()

    deletable = []
    for node in network.nodes():
        node_step = vkey2step[node]

        if node in nodes_span_short:
            if node_step > max_step_sequential_short or node_step > step:
                deletable.append(node)
        else:
            if node_step > step:
                deletable.append(node)

    for dnode in deletable:
        network.delete_node(dnode)

# ==========================================================================
# Define optimization parameters
# ==========================================================================

    parameters = []
    for edge in network.edges():
        parameter = EdgeForceDensityParameter(edge, qmin, qmax)
        parameters.append(parameter)

    if parametrize_supports:

        print("\nAdding supports' XY coordinates as optimization parameters")
        for node in network.nodes_supports():
            node_step = vkey2step[node]
            if node_step == 0:
                continue
            else:
                x, y, _ = network0.node_coordinates(node)
                parameter = NodeSupportXParameter(node, x-xytol, x+xytol)
                parameters.append(parameter)
                parameter = NodeSupportYParameter(node, y-xytol, y+xytol)
                parameters.append(parameter)


# ==========================================================================
# Define goals
# ==========================================================================

    goals_xyz = []
    for node in network.nodes_free():

        point = network0.node_coordinates(node)
        weight_xyz = 1.0

        if weight_goals_per_step:
            node_step = vkey2step[node]
            weight_xyz = node_step / step

        goals_xyz.append(NodePointGoal(node, target=point, weight=weight_xyz))

    goals_length = []
    for edge in network.edges():
        length = network0.edge_length(*edge)
        goals_length.append(EdgeLengthGoal(edge, target=length))

# ==========================================================================
# Combine error functions and regularizer into custom loss function
# ==========================================================================

    loss = Loss(error_fn(goals_xyz, alpha_xyz),
                error_fn(goals_length, alpha_length),
                L2Regularizer(alpha_reg))

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
# Report stats
# ==========================================================================

    network.print_stats()

# ==========================================================================
# Update force densities
# ==========================================================================

    if update_qs_after_optimization:
        for edge in network.edges():
            q = network.edge_forcedensity(edge)
            network0.edge_forcedensity(edge, q)

# ==========================================================================
# Hausdorff distance
# ==========================================================================

    U = np.array([network0.node_coordinates(node) for node in network.nodes_free()])
    V = np.array([network.node_coordinates(node) for node in network.nodes_free()])
    directed_u = directed_hausdorff(U, V)[0]
    directed_v = directed_hausdorff(V, U)[0]
    hausdorff = max(directed_u, directed_v)

    print(f"\nHausdorff distances: U: {directed_u:.3f}\tV: {directed_v:.3f}\tUV: {round(hausdorff, 3)}")

# ==========================================================================
# Export JSON
# ==========================================================================

    if export:
        filename = f"tripod_network_dual_bestfit_3d_step_{step}.json"
        if add_bracing:
            filename = f"tripod_network_dual_bestfit_braced_3d_step_{step}.json"
        filepath = os.path.join(DATA, filename)
        network.to_json(filepath)
        print(f"\nExported network JSON file to {filepath}!")

# ==========================================================================
# Visualization
# ==========================================================================

    if view:
        print("\nViewing")
        viewer = Viewer(width=1600, height=900, show_grid=False)

        viewer.view.color = (0.1, 0.1, 0.1, 1)  # change background to dark gray

        # best-fit network
        viewer.add(network,
                   edgewidth=(0.001, 0.05),
                   edgecolor="force",
                   show_loads=True,
                   loadscale=2.0)

        # target network
        viewer.add(network0,
                   as_wireframe=True,
                   show_points=False,
                   linewidth=1.0,
                   color=Color.white().darkened())

        # draw lines between best-fit and target networks
        for node in network.nodes():
            pt = network.node_coordinates(node)
            line = Line(pt, network0.node_coordinates(node))
            viewer.add(line, color=Color.grey())

        # show le crème
        viewer.show()
