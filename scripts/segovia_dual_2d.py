import os

from collections import defaultdict
from itertools import combinations

from math import pi, radians, degrees

from numpy import mean, std

from operator import itemgetter

from compas.colors import Color, ColorMap
from compas.geometry import Line, Polyline, distance_point_point, length_vector, length_vector_xy, sum_vectors, cross_vectors, rotate_points, bounding_box
from compas.geometry import add_vectors, scale_vector, normalize_vector

from compas_quad.datastructures import CoarseQuadMesh
from compas_quad.coloring import quad_mesh_polyedge_2_coloring

from fourf import DATA

from fourf.topology import threefold_vault
from fourf.topology import quadmesh_densification
from fourf.support import support_shortest_boundary_polyedges
from fourf.support import polyedge_types
from fourf.sequence import quadmesh_polyedge_assembly_sequence
from fourf.utilities import mesh_to_fdnetwork
from fourf.view import view_f4

from jax_fdm.equilibrium import fdm, constrained_fdm
from jax_fdm.optimization import LBFGSB, SLSQP
from jax_fdm.parameters import EdgeForceDensityParameter
from jax_fdm.parameters import NodeSupportXParameter
from jax_fdm.parameters import NodeSupportYParameter

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


brick_length = 0.225   # [m]
brick_width = 0.1   # [m]
brick_thickness = 0.025  # [m]
brick_layers = 3  # [-]
brick_density = 16.5  # [kN/m3]
comp_strength = 5.0  # [MPa]

target_edge_length = 0.25  # 0.25 [m] for mesh subdivision

dead_load = 1.0  # dead load [kN/m2]
pz = 0.0
q0 = -1.0  # initial force densities [kN/m]

opt = LBFGSB  # optimization solver
qmin, qmax = None, -1e-1  # bound on force densities [kN/m]
maxiter = 5000  # maximum number of iterations
tol = 1e-6  # optimization tolerance

# aim for target positions
add_node_spine_xyz_goal = True
weight_node_spine_xyz_goal = 100.0

# # keep spine planar
add_spine_planarity_goal = False
weight_spine_planarity_goal = 10.0

# # keep profile planar
add_profile_planarity_goal = False  # True
weight_profile_planarity_goal = 10.0

# all edges length goal
add_edge_length_goal = True
length_factor = 0.8
weight_edge_length_goal = 1.0

# edge length goal to obtain constant brick course widths
add_edge_length_profile_goal = False
weight_edge_length_profile_goal = 1.0

# edge length goal to obtain constant brick course widths
add_edge_length_strips_goal = False
weight_edge_length_strips_goal = 0.1  # 0.1

# edge equalize length goal to obtain constant brick course widths
add_edge_length_equal_strips_goal = False  # True
weight_edge_length_equal_strips_goal = 0.1

# edge equalize length goals to polyedges parallel to spine
add_edge_length_equal_polyedges_goal = False  # True
weight_edge_length_equal_polyedges_goal = 0.1

# controls
optimize = False
view = True
results = False
export = False

# ==========================================================================
# Import dual mesh
# ==========================================================================

filepath = os.path.abspath(os.path.join(DATA, "tripod_dual_2d_subd_mesh.json"))
mesh = CoarseQuadMesh.from_json(filepath)

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

for skey, strip in mesh.strips(True):
    if strip[0][0] in hexagon or strip[-1][0] in hexagon:
        mesh.strip_density(skey, 1)
    else:
        mesh.set_strip_density_target(skey, target_edge_length)

mesh.densification()

mesh = mesh.dense_mesh()

assert len(mesh.vertices_on_boundaries()) == 2, f"{len(mesh.vertices_on_boundaries())} boundaries!"

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
# Create FD network
# ==========================================================================

network = mesh_to_fdnetwork(mesh, supported_vkeys, pz, q0)
network0 = network.copy()

# ==========================================================================
# Parameters
# ==========================================================================

if optimize:

    parameters = []

    for edge in network.edges():
        parameter = EdgeForceDensityParameter(edge, qmin, qmax)
        parameters.append(parameter)

# ==========================================================================
# Goals I
# ==========================================================================

    print()

    # spine xyz
    goals_spine_xyz = []
    if add_node_spine_xyz_goal:
        for node in spine_nodes:
            point = network.node_coordinates(node)
            goal = NodePointGoal(node, target=point, weight=weight_node_spine_xyz_goal)
            goals_spine_xyz.append(goal)
        print('{} SpineNodesPointGoal'.format(len(goals_spine_xyz)))

    # spine planarity
    goals_spine_planarity = []
    if add_spine_planarity_goal:
        for polyedge in spine_polyedges:
            vector = mesh.edge_vector(polyedge[0], polyedge[1])
            origin = mesh.vertex_coordinates(polyedge[0])
            normal = cross_vectors(vector, [0.0, 0.0, 1.0])
            plane = (origin, normal)
            for node in polyedge[1:]:
                goal = NodePlaneGoal(node, plane, weight_spine_planarity_goal)
                goals_spine_planarity.append(goal)
        print('{} SpinePlanarityGoal'.format(len(goals_spine_planarity)))

    # profile planarity
    goals_profile_planarity = []
    if add_profile_planarity_goal:
        for polyedge in profile_polyedges:
            vector = mesh.edge_vector(polyedge[0], polyedge[1])
            origin = mesh.vertex_coordinates(polyedge[0])
            normal = cross_vectors(vector, [0.0, 0.0, 1.0])
            plane = (origin, normal)
            for node in polyedge[1:]:
                goal = NodePlaneGoal(node, plane, weight_profile_planarity_goal)
                goals_profile_planarity.append(goal)
        print('{} ProfilePlanarityGoal'.format(len(goals_spine_planarity)))

    # length of profile curves edges
    goals_length_profile = []
    if add_edge_length_profile_goal:
        for polyedge in profile_polyedges:

            for i, edge in enumerate(pairwise(polyedge)):
                factor = 1.0
                # if i == 0:
                #     factor = 2.0

                if not network.has_edge(u, v):
                    u, v = v, u
                edge = (u, v)
                goal = EdgeLengthGoal(edge, target=target_edge_length * factor, weight=weight_edge_length_profile_goal)

                goals_length_profile.append(goal)

        print('{} EdgeProfileLengthGoal'.format(len(goals_length_profile)))

    # edge equalize length goals to polyedges parallel to spine
    goals_length_equal_polyedges = []
    if add_edge_length_equal_polyedges_goal:
        for pkey, polyedge in mesh.polyedges(True):
            ptype = pkey2type[pkey]
            if ptype != "span":
                continue
            edges = []
            for u, v in pairwise(polyedge):
                if not network.has_edge(u, v):
                    u, v = v, u
                edges.append((u, v))
            goal = EdgesLengthEqualGoal(edges, weight=weight_edge_length_equal_polyedges_goal)
            goals_length_equal_polyedges.append(goal)
        print('{} EdgePolyedgesEqualLengthGoal'.format(len(goals_length_equal_polyedges)))

    goals_length_equal_strips = []
    if add_edge_length_equal_strips_goal:

        for i in range(0, max_step + 1):
            edges = step2edges[i]
            # edges = [edge for edge in step2edges[i] if edge in edges1]
            goal = EdgesLengthEqualGoal(edges, weight=weight_edge_length_equal_strips_goal)
            goals_length_equal_strips.append(goal)

        print('{} EdgeStripsEqualLengthGoal'.format(len(goals_length_equal_strips)))

    goals_length = []
    if add_edge_length_goal:
        for u, v in edges0 + edges1:
            if (u, v) in profile_edges or (v, u) in profile_edges:
                continue
            if not network.has_edge(u, v):
                u, v = v, u
            edge = (u, v)
            length = mesh.edge_length(*edge) * length_factor
            goal = EdgeLengthGoal(edge, length)#  weight_edge_length_goal)
            goals_length.append(goal)

        print('{} EdgeReducedLengthGoal'.format(len(goals_length)))

# ==========================================================================
# Loss function
# ==========================================================================

    loss = Loss(
                SquaredError(goals=goals_spine_xyz, name='NodeSpineXYZGoal', alpha=1.0),
                SquaredError(goals=goals_spine_planarity, name='EdgeSpinePlanarityGoal', alpha=1.0),
                SquaredError(goals=goals_profile_planarity, name='EdgeProfilePlanarityGoal', alpha=1.0),
                SquaredError(goals=goals_length_profile, name='EdgeProfileLengthGoal', alpha=1.0),
                PredictionError(goals=goals_length_equal_polyedges, name='EdgesLengthEqualPolyedgesGoal', alpha=1.0),
                PredictionError(goals=goals_length_equal_strips, name='EdgesLengthEqualStripsGoal', alpha=1.0),
                SquaredError(goals=goals_length, name='EdgesLengthGoal', alpha=1.0),
                )

# ==========================================================================
# Constrained form-finding
# ==========================================================================

    network = constrained_fdm(network,
                              optimizer=opt(),
                              parameters=parameters,
                              loss=loss,
                              maxiter=maxiter,
                              tol=tol
                              )

    cnetwork1 = network.copy()

# ==========================================================================
# Export
# ==========================================================================

    # update mesh coordinates
    for vkey in mesh.vertices():
        xyz = network.node_coordinates(vkey)
        mesh.vertex_attributes(vkey, names="xyz", values=xyz)

    if export:
        for name, datastruct in {"network": network, "mesh": mesh}.items():
            filepath = os.path.join(DATA, f"tripod_{name}_dual_2d.json")
            datastruct.to_json(filepath)
        print("Exported JSON files!")

# ==========================================================================
# Visualizationj
# ==========================================================================

if view:
    viewer = Viewer(width=1600, height=900, show_grid=False)

    viewer.view.color = (0.1, 0.1, 0.1, 1)  # change background to black

    # viewer.add(network0, as_wireframe=True, show_points=False)

    # profile lines
    # for line in profile_lines:
    #     viewer.add(line)

    for vkey in mesh.vertices():
        xyz = network.node_coordinates(vkey)
        mesh.vertex_attributes(vkey, names="xyz", values=xyz)

    # viewer.add(mesh)

    # viewer.add(eqnetwork, as_wireframe=True)

    # viewer.add(cnetwork1, as_wireframe=True, show_points=False)

    # for polyedge in spine_polyedges:
    #     for i, edge in enumerate(pairwise(polyedge)):
    #         line = Line(*mesh.edge_coordinates(*edge))
    #         viewer.add(line, linewidth=4.)

    mins = 1
    maxs = max_step + 1
    cmap = ColorMap.from_mpl("viridis")
    for i in range(mins, maxs):
        edges = step2edges[i]
        for edge in edges:
            line = Line(*network.edge_coordinates(*edge))
            # viewer.add(line, linecolor=cmap(i, minval=mins, maxval=maxs), linewidth=2.)

    #     for strip in profile_strips[i]:
    #         for edge in strip:
    #             line = Line(*network.edge_coordinates(*edge))
    #             viewer.add(line, linecolor=cmap(i, maxval=maxs), linewidth=5.)

    #         for polyedge in zip(*strip):
    #             polyline = Polyline([network.node_coordinates(n) for n in polyedge])
    #             viewer.add(polyline)

    spine_strip_edges_flat = [edge for edges in spine_strip_edges for edge in edges]

    # support polyedges
    for pkey in supported_pkeys:
        for edge in pairwise(mesh.polyedge_vertices(pkey)):
            if edge in spine_strip_edges_flat:
                continue
            viewer.add(Line(*mesh.edge_coordinates(*edge)), linecolor=(1.0, 0.5, 0.5), linewidth=2.)
    # spine strips
    for strip in spine_strip_edges:
        for edge in strip:
            viewer.add(Line(*mesh.edge_coordinates(*edge)), linecolor=(1.0, 0.0, 1.0), linewidth=2.)

    # polyedges
    for polyedge in profile_polyedges:
        for edge in pairwise(polyedge):
            x = edge2step[tuple(sorted(edge))] / max_step
            viewer.add(Line(*mesh.edge_coordinates(*edge)), linecolor=(0.0, 0.2 + 0.8 * x, 0.2 + 0.8 * x), linewidth=2.0)

    # edges 0
    for edge in edges0:
        x = edge2step[tuple(sorted(edge))] / max_step
        viewer.add(Line(*mesh.edge_coordinates(*edge)), linecolor=(0.5 * x, 0.5 * x, 0.5 * x), linewidth=2.)

    # edges 1
    for edge in edges1:
        x = edge2step[tuple(sorted(edge))] / max_step
        linecolor = (0.8 * x, 1.0, 0.8 * x)
        viewer.add(Line(*mesh.edge_coordinates(*edge)), linecolor=linecolor, linewidth=2.)

    viewer.show()
