import os

from collections import defaultdict
from itertools import combinations

from math import pi, radians, degrees, fabs

from numpy import mean, std

from operator import itemgetter

from compas.datastructures import network_find_cycles, Mesh, Network

from compas.colors import Color, ColorMap
from compas.geometry import Line, Polyline, distance_point_point, length_vector, length_vector_xy, sum_vectors, cross_vectors, rotate_points, bounding_box
from compas.geometry import add_vectors, scale_vector, normalize_vector, angle_vectors, project_point_plane, normalize_vector, subtract_vectors
from compas.geometry import dot_vectors, Point

from compas_view2.shapes import Arrow

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
from jax_fdm.optimization import LBFGSB, SLSQP, OptimizationRecorder
from jax_fdm.parameters import EdgeForceDensityParameter, NodeAnchorXParameter, NodeAnchorYParameter, NodeAnchorZParameter
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
from jax_fdm.goals import EdgeAngleGoal
from jax_fdm.goals import NodeResidualForceGoal

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

target_edge_length = 0.25  # 0.25 [m]

dead_load = 1.0  # additional dead load [kN/m2]
pz = brick_density * brick_thickness * brick_layers + dead_load  # vertical area load (approximated self-weight + uniform dead load) [kN/m2]
print(f"Area load: {pz} [kN/m2]")

freeze_spine = True  # add supports to the nodes of the spine
update_loads = False  # recompute node loads at every assembly step based on vertex area of form-found mesh

opt = LBFGSB  # optimization solver
qmin, qmax = -20.0, -1e-1  # bounds on force densities [kN/m]
maxiter = 5000  # maximum number of iterations
tol = 1e-9  # optimization tolerance

# keep horizontal projection fixed
add_node_xy_goal = True
weight_node_xy_goal = 5.0

# best-fit nodes
add_node_bestfit_goal = True
weight_node_bestfit_goal = 5.0

# node tangent goal
add_node_tangent_goal = False
weight_node_tangent_goal = 1.0

# edge length goal to obtain constant brick course widths
# add_edge_length_profile_goal = False
# weight_edge_length_profile_goal = 1.0

# profile edges direction goal TODO: it is angle!
add_edge_direction_profile_goal = True
l_start, l_end, l_exp = radians(45), radians(15), 1.0  # long spans, angle bounds and variation exponent [-]
s_start, s_end, s_exp = radians(30), radians(15), 1.0  # short spans, angle bounds and variation exponent [-]
weight_edge_direction_profile_goal = 1.0

# equalize length of edges parallel to the spine
add_edge0_length_equal_goal = False
weight_edge0_length_equal_goal = 1.0

# edge equalize length of edges perpendicular to the spine
add_edge1_length_equal_goal = True
weight_edge1_length_equal_goal = 1.0  # 0.1

# reduce reaction forces at the spine
add_node_spine_reaction_goal = False
weight_node_spine_reaction_goal = 0.01  # 0.1

# controls
export = False
results = False
view = True
view_node_tangents = False

# sequential form finding
max_step_sequential = 5  # 5
max_step_sequential_short = 3  # 3

# ==========================================================================
# Import datastructures
# ==========================================================================

network = FDNetwork.from_json(os.path.join(DATA, "tripod_network_dual_2d.json"))
mesh = QuadMesh.from_json(os.path.abspath(os.path.join(DATA, "tripod_mesh_dual_2d.json")))
network_spine = FDNetwork.from_json(os.path.join(DATA, "tripod_network_dual_spine_3d.json"))

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
# Update network
# ==========================================================================

for node in network.nodes():
    vertex_area = mesh.vertex_area(node)
    network.node_load(node, load=[0.0, 0.0, vertex_area * pz * -1.0])

network_base = network.copy()
network0 = network.copy()

for node in spine_nodes:
    z = network_spine.node_attribute(node, "z")
    network0.node_attribute(node, "z", z)

# store data
network = network_spine.copy()

# ==========================================================================
# Freeze spine nodes in the air
# ==========================================================================

if freeze_spine:
    print("\nFreezing spine in the air!")
    for node in spine_nodes:
        network.node_support(node)

# ==========================================================================
# Sequential constrained form-finding
# ==========================================================================

print(network)
networks = {}

for step in range(1, max_step_sequential + 1):

    print(f"\n***Step: {step}***")

    # TODO: update loads
    if update_loads:
        cycles = network_find_cycles(network)
        vertices = {vkey: network.node_coordinates(vkey) for vkey in network.nodes()}
        _mesh = Mesh.from_vertices_and_faces(vertices, cycles)
        _mesh.delete_face(0)
        _mesh.cull_vertices()

        print("Updating loads on network from recomputed mesh")
        for node in network.nodes():
            vertex_area = _mesh.vertex_area(node)
            network.node_load(node, load=[0.0, 0.0, vertex_area * pz * -1.0])

    # add edges
    nodes_step = set()
    for edges in (edges0, edges1, profile_edges):
        for edge in edges:
            # take edge only if it belongs to the current assembly step
            if edge2step.get(edge, edge2step.get((edge[1], edge[0]))) != step:
                continue
            if edge in edges_span_short or (edge[1], edge[0]) in edges_span_short:
                if step > max_step_sequential_short:
                    continue
            for node in edge:
                # skip node if it already exists
                if network.has_node(node):
                    continue
                # add node at position xyz
                x, y, z = network0.node_coordinates(node)
                network.add_node(node, x=x, y=y, z=z)
                nodes_step.add(node)
                # add node load
                network.node_load(node, load=network0.node_load(node))
                # add support
                if node in supports or node in spine_nodes:
                    network.node_support(node)

            # correct edge orientation
            u, v = edge
            if not network0.has_edge(u, v):
                u, v = v, u
            if network.has_edge(u, v):
                continue
            # add edge
            edge = network.add_edge(u, v)
            # assign edge force density
            q = network0.edge_forcedensity(edge)
            network.edge_forcedensity(edge, q)

    print(network)

# ==========================================================================
# Angle of the assembly step
# ==========================================================================

    angle_long = l_start + (l_end - l_start) * ((step - 1) / (max_step - 1)) ** l_exp
    angle_short = s_start + (s_end - s_start) * ((step - 1) / (max_step - 1)) ** s_exp
    print(f"Angle long span: {degrees(angle_long):.2f}")
    print(f"Angle short span: {degrees(angle_short):.2f}")

# ==========================================================================
# Optimization parameters
# ==========================================================================

    parameters = []

    for edge in network.edges():
        parameter = EdgeForceDensityParameter(edge, qmin, qmax)
        parameters.append(parameter)

    # ztol = 0.05
    # for node in spine_nodes:
    #     if node in supported_vkeys:
    #         continue
    #     x, y, z = network.node_coordinates(node)
    #     parameter = NodeAnchorZParameter(node, z - ztol, z + ztol)
    #     parameters.append(parameter)

# ==========================================================================
# Goals
# ==========================================================================

    # goals
    goals = []

    # maintain horizontal projection
    points = []
    if add_node_xy_goal:
        for node in nodes_step:  # NOTE: nodes added at this assembly step
            x, y, z = network_base.node_coordinates(node)
            points.append(Point(x, y, z))
            for goal, xy in zip((NodeXCoordinateGoal, NodeYCoordinateGoal), (x, y)):
                goal = goal(node, xy, weight_node_xy_goal)
                goals.append(goal)

    # best-fit history
    if add_node_bestfit_goal:
        for node in network.nodes_free():  # NOTE: nodes added at all previous steps
            if node in nodes_step:
                continue
            xyz = network.node_coordinates(node)
            goal = NodePointGoal(node, xyz, weight_node_bestfit_goal)
            goals.append(goal)

    # node tangent goal
    if add_node_tangent_goal:
        for node in nodes_step:
            if node in supports:
                continue
            goal = NodeTangentAngleGoal(node, vector=[0.0, 0.0, 1.0], target=pi * 0.5 - (angle))
            goals.append(goal)

    # edge direction goal
    profile_lines = []
    if add_edge_direction_profile_goal:
        for edge in profile_edges:
            # take edge only if it belongs to the current assembly step
            if edge2step.get(edge, edge2step.get((edge[1], edge[0]))) != step:
                continue

            if edge in edges_span_short or (edge[1], edge[0]) in edges_span_short:
                if step > max_step_sequential_short:
                    continue

            angle = angle_long

            if edge in profile_edges_span_short:
                angle = angle_short

            u, v = edge
            factor = 1.0
            if not network.has_edge(u, v):
                u, v = v, u
                factor = -1.0
            edge = (u, v)

            vector0 = mesh.edge_vector(u, v)
            ortho = cross_vectors(vector0, [0.0, 0.0, 1.0])
            vector = rotate_points([vector0], angle * factor, axis=ortho, origin=[0.0, 0.0, 0.0])[0]
            goal = EdgeDirectionGoal(edge, target=vector, weight=weight_edge_direction_profile_goal)
            # vector_ref = [0.0, 0.0, 1.0]
            # goal = EdgeAngleGoal(edge, [0.0, 0.0, 1.0], pi * 0.5 - (angle * factor))
            goals.append(goal)

            # for viz
            start = network_base.node_coordinates(u)
            end = add_vectors(start, scale_vector(normalize_vector(vector), target_edge_length))
            line = Line(start, end)
            profile_lines.append(line)

    # equalize length of edges parallel to the spine
    if add_edge0_length_equal_goal:
        _edges = []
        for edge in edges0:
            # take edge only if it belongs to the current assembly step
            if edge2step.get(edge, edge2step.get((edge[1], edge[0]))) != step:
                continue
            if edge in edges_span_short or (edge[1], edge[0]) in edges_span_short:
                if step > max_step_sequential_short:
                    continue
            u, v = edge
            if not network.has_edge(u, v):
                u, v = v, u
            _edges.append((u, v))

        goal = EdgesLengthEqualGoal(_edges, weight=weight_edge0_length_equal_goal)
        goals.append(goal)

    # equalize length of edges perpendicular to the spine
    if add_edge1_length_equal_goal:
        _edges = []
        for edges in (edges1, profile_edges):
            for edge in edges:
                # take edge only if it belongs to the current assembly step
                if edge2step.get(edge, edge2step.get((edge[1], edge[0]))) != step:
                    continue
                if edge in edges_span_short or (edge[1], edge[0]) in edges_span_short:
                    if step > max_step_sequential_short:
                        continue

                u, v = edge
                if not network.has_edge(u, v):
                    u, v = v, u
                _edges.append((u, v))

        goal = EdgesLengthEqualGoal(_edges, weight=weight_edge1_length_equal_goal)
        goals.append(goal)

    # reduce reaction forces at spine nodes
    if freeze_spine and add_node_spine_reaction_goal:
        for node in spine_nodes:
            if node in supported_vkeys:
                continue
            goal = NodeResidualForceGoal(node, 0.0, weight_node_spine_reaction_goal)
            goals.append(goal)

# ==========================================================================
# Loss function
# ==========================================================================

    loss = Loss(SquaredError(goals))

# ==========================================================================
# Constrained form-finding
# ==========================================================================

    network = constrained_fdm(network,
                              optimizer=opt(),
                              loss=loss,
                              parameters=parameters,
                              maxiter=maxiter,
                              tol=tol,
                              callback=None)

    # print out network statistics
    network.print_stats()

# ==========================================================================
# Store data
# ==========================================================================

    # store network
    networks[step] = network.copy()

# ==========================================================================
# Export assembly step
# ==========================================================================

    print("\nExporting assembly step networks...")
    print("Deleting supported edges...")

    _network = network.copy()
    deletable = []
    for edge in _network.edges():
        u, v = edge
        if _network.is_node_support(u) and _network.is_node_support(v):
            deletable.append(edge)
    for u, v in deletable:
        _network.delete_edge(u, v)

    _network = fdm(_network)

    filepath = os.path.join(DATA, f"tripod_network_dual_3d_step_{step}.json")
    _network.to_json(filepath)
    print("\nExported assembly JSON file")

# ==========================================================================
# Generate mesh
# ==========================================================================

cycles = network_find_cycles(network)
vertices = {vkey: network.node_coordinates(vkey) for vkey in network.nodes()}
mesh = Mesh.from_vertices_and_faces(vertices, cycles)
mesh.delete_face(0)
mesh.cull_vertices()

# ==========================================================================
# Export
# ==========================================================================

if export:
    for name, datastruct in {"network": network, "mesh": mesh}.items():

        filepath = os.path.join(DATA, f"tripod_{name}_dual_3d.json")
        datastruct.to_json(filepath)

    print("\nExported JSON files!")

# ==========================================================================
# Visualization
# ==========================================================================

if view:

    viewer = Viewer(width=1600, height=900, show_grid=False)

    viewer.view.color = (0.1, 0.1, 0.1, 1)  # change background to black

    viewer.add(network, edgecolor="force", show_loads=True, show_reactions=True, edgewidth=(0.01, 0.03))
    viewer.add(mesh, show_points=False, show_lines=False, opacity=0.5)

    # viewer.add(network, as_wireframe=True, show_points=False)

    network_base = network.copy()
    for node in network_base.nodes():
        network_base.node_attribute(node, "z", 0.0)
    viewer.add(network_base, as_wireframe=True, show_points=False)

    # # profile lines
    # for line in profile_lines:
    #     viewer.add(line, linewidth=5.0, linecolor=Color.red())

    # for edge in edges_span_short:
    #     viewer.add(Line(*network_base.edge_coordinates(*edge)), color=Color.green(), linewidth=5.0)

    # cmap = ColorMap.from_mpl("magma")
    # for strip in profile_strips:
    #     for i, edge in enumerate(strip):
    #         viewer.add(Line(*network_base.edge_coordinates(*edge)), color=cmap(i, maxval=len(strip)), linewidth=5.0)

    # for polyedges in profile_polyedges_from_strips:
    #     for polyedge in polyedges:
    #         for i, edge in enumerate(pairwise(polyedge)):
    #             viewer.add(Line(*network_base.edge_coordinates(*edge)), color=cmap(i, maxval=len(polyedge)), linewidth=5.0)

    # for vkey in mesh.vertices():
    #     xyz = network.node_coordinates(vkey)
    #     mesh.vertex_attributes(vkey, names="xyz", values=xyz)

    # viewer.add(mesh)

    # viewer.add(eqnetwork, as_wireframe=True)

    # viewer.add(cnetwork1, as_wireframe=True, show_points=False)

    # for polyedge in spine_polyedges:
    #     for i, edge in enumerate(pairwise(polyedge)):
    #         line = Line(*mesh.edge_coordinates(*edge))
    #         viewer.add(line, linewidth=4.)

    # mins = 1
    # maxs = max_step + 1
    # cmap = ColorMap.from_mpl("viridis")
    # for i in range(mins, maxs):
    #     edges = step2edges[i]
    #     for edge in edges:
    #         line = Line(*network.edge_coordinates(*edge))
            # viewer.add(line, linecolor=cmap(i, minval=mins, maxval=maxs), linewidth=2.)

    #     for strip in profile_strips[i]:
    #         for edge in strip:
    #             line = Line(*network.edge_coordinates(*edge))
    #             viewer.add(line, linecolor=cmap(i, maxval=maxs), linewidth=5.)

    #         for polyedge in zip(*strip):
    #             polyline = Polyline([network.node_coordinates(n) for n in polyedge])
    #             viewer.add(polyline)

    # spine_strip_edges_flat = [edge for edges in spine_strip_edges for edge in edges]

    # # support polyedges
    # for pkey in supported_pkeys:
    #     for edge in pairwise(mesh.polyedge_vertices(pkey)):
    #         if edge in spine_strip_edges_flat:
    #             continue
    #         viewer.add(Line(*mesh.edge_coordinates(*edge)), linecolor=(1.0, 0.5, 0.5), linewidth=2.)
    # spine strips
    # for strip in spine_strip_edges:
    #     for edge in strip:
    #         viewer.add(Line(*network_spine.edge_coordinates(*edge)), linecolor=(1.0, 0.0, 1.0), linewidth=2.)

    # # polyedges
    # for polyedge in profile_polyedges:
    #     for edge in pairwise(polyedge):
    #         x = edge2step[tuple(sorted(edge))] / max_step
    #         viewer.add(Line(*mesh.edge_coordinates(*edge)), linecolor=(0.0, 0.2 + 0.8 * x, 0.2 + 0.8 * x), linewidth=2.0)

    # # edges 0
    # for edge in edges0:
    #     x = edge2step[tuple(sorted(edge))] / max_step
    #     viewer.add(Line(*mesh.edge_coordinates(*edge)), linecolor=(0.5 * x, 0.5 * x, 0.5 * x), linewidth=2.)

    # # edges 1
    # for edge in edges1:
    #     x = edge2step[tuple(sorted(edge))] / max_step
    #     linecolor = (0.8 * x, 1.0, 0.8 * x)
    #     viewer.add(Line(*mesh.edge_coordinates(*edge)), linecolor=linecolor, linewidth=2.)

    # node tangent arrows
    if view_node_tangents:
        angles_mesh = []
        tangent_angles_mesh = []
        arrows = []
        tangent_arrows = []
        vkeys = []

        # for vkey in mesh.vertices():
        for vkey in nodes_step:
            vkeys.append(vkey)

            xyz = mesh.vertex_coordinates(vkey)

            if vkey in supports or vkey in spine_nodes:
                continue

            normal = scale_vector(normalize_vector(mesh.vertex_normal(vkey)), 0.25)
            z_vector = [0., 0., 1.]

            angle = angle_vectors(z_vector, normal, deg=True)

            ppoint = project_point_plane(add_vectors(xyz, z_vector), (xyz, normal))
            tangent_vector = normalize_vector(subtract_vectors(ppoint, xyz))
            tangent_arrow = Arrow(xyz, scale_vector(tangent_vector, 0.25))

            tangent_arrows.append(tangent_arrow)

            angles_mesh.append(angle)
            tangent_angles_mesh.append(angle_vectors(z_vector, tangent_vector, deg=True))

            arrow = Arrow(xyz, normal)
            arrows.append(arrow)

        cmap = ColorMap.from_mpl("plasma")
        min_angle = min(tangent_angles_mesh)
        max_angle = max(tangent_angles_mesh)
        print(f"\nTangent angle\tMin: {min_angle:.2f}\tMax: {max_angle:.2f}\tMean: {sum(tangent_angles_mesh)/len(tangent_angles_mesh):.2f}\n")

        for vkey, angle, tangent_angle, arrow, tarrow in zip(vkeys, angles_mesh, tangent_angles_mesh, arrows, tangent_arrows):
            # color = cmap(tangent_angle, minval=min_angle, maxval=max_angle)
            color = cmap(tangent_angle, minval=min_angle, maxval=max_angle)
            # viewer.add(arrow, show_edges=False, opacity=0.5)
            viewer.add(tarrow, facecolor=color, show_edges=False, opacity=0.8)
            # print(f"node: {vkey}\tangle: {angle:.2f}\ttangent angle: {tangent_angle:.2f}\ttangent angle 2: {90-angle:.2f}")


    viewer.show()
