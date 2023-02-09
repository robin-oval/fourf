import os

from math import pi, radians, degrees, fabs

from compas.datastructures import network_find_cycles, Mesh, Network, mesh_weld

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

from jax_fdm.losses import L2Regularizer, SquaredError, Loss, PredictionError, AbsoluteError, MeanSquaredError, MeanAbsoluteError

from jax_fdm.visualization import Viewer

from compas.utilities import pairwise

# ==========================================================================
# Parameters
# ==========================================================================

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
mortar_density = 19.0  # [kN/m3]

# vertical area load (approximated self-weight) [kN/m2]
brick_hollow_pz = brick_hollow_density * brick_hollow_thickness * brick_hollow_layers
brick_solid_pz = brick_solid_density * brick_solid_thickness * brick_solid_layers
mortar_pz = mortar_density * mortar_thickness * mortar_layers
pz = brick_hollow_pz + brick_solid_pz + mortar_pz

print(f"Area load: {pz:.2f} [kN/m2] (Brick hollow:  {brick_hollow_pz:.2f} [kN/m2]\tBrick solid:  {brick_solid_pz:.2f} [kN/m2]\tMortar {mortar_pz:.2f} [kN/m2])")

# controls
export = True
view = True

# spine parameters
freeze_spine = True  # add supports to the nodes of the spine
update_loads = True  # recompute node loads at every assembly step based on vertex area of form-found mesh
fofin_last = False  # form-finding last generated network with updated loads
bestfit_last = False  # approximate shape with updated node loads

# sequential form finding
max_step_sequential = 5  # 5
max_step_sequential_short = 3  # 3

# angle bounds
l_start, l_end, l_exp = radians(45), radians(15), 1.0  # long spans, angle bounds and variation exponent [-]
s_start, s_end, s_exp = radians(30), radians(15), 1.0  # short spans, angle bounds and variation exponent [-]

# constrained form-finding
opt = LBFGSB  # optimization solver
qmin, qmax = -15.0, -1e-1  # bounds on force densities [kN/m]
maxiter = 20000  # maximum number of iterations
tol = 1e-9  # 1e-9  optimization tolerance
parametrize_z_spine = True  # add z coordinate of spine supports as optimization parameters
ztol = 0.05  # 0.1
alpha = 1e-2  # 1e-2, 5e-3  alpha coefficient for L2 regularizer

# best-fit past nodes
add_node_bestfit_goal = True
weight_node_bestfit_goal = 5.0

# keep horizontal projection fixed
add_node_xy_goal = True
weight_node_xy_goal = 3.0  # 3.

# profile edges direction goal
add_edge_direction_profile_goal = True
weight_edge_direction_profile_goal = 10.0  # 5.0

# equalize length of edges perpendicular to the spine
add_edge1_length_equal_goal = True
weight_edge1_length_equal_goal = 1.0

# equalize length of edges parallel to the spine
add_edge0_length_equal_goal = True
weight_edge0_length_equal_goal = 0.2

# edge target length goal
add_edge_length_profile_goal = False
weight_edge_length_profile_goal = 1.0
edge_length_factor = 1.0

# reduce reaction forces at the spine
add_node_spine_reaction_goal = True
weight_node_spine_reaction_goal = 15.  # 5

# spine nodes no torsion goal
add_node_spine_notorsion_goal = True
weight_node_spine_notorsion_goal = 1.0

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

# spine_center = Network()
# for strip in spine_strip_edges:
#     nodes = []
#     for u, v in strip:
#         x, y, z = network.edge_midpoint(u, v)
#         node = spine_center.add_node(x=x, y=y, z=z)
#         nodes.append(node)
#     for u, v in pairwise(nodes):
#         spine_center.add_edge(u, v)

# filepath = os.path.join(DATA, f"tripod_network_dual_spine_center_corrected_3d.json")
# spine_center.to_json(filepath)

# raise
# ==========================================================================
# Updates
# ==========================================================================

# update loads on 2d mesh
for node in network.nodes():
    vertex_area = mesh.vertex_area(node)
    network.node_load(node, load=[0.0, 0.0, vertex_area * pz * -1.0])

# make copies
network_base = network.copy()
network0 = network.copy()

# lift up spine nodes
for node in spine_nodes:
    z = network_spine.node_attribute(node, "z")
    network0.node_attribute(node, "z", z)

# ==========================================================================
# Initial network
# ==========================================================================

# make a copy of the initial spine-modified network
network_spine = network_spine.copy()
network = network_spine.copy()

# almost zero-out force densities in spine edges
for edge in network.edges():
    if edge in spine_strip_edges_edges or (edge[1], edge[0]) in spine_strip_edges_edges:
        network.edge_forcedensity(edge, -1e-3)

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

networks = {}
nodes_xyz_history = {}

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
            # stop adding edges in short span if condition is fulfilled
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
            edge = (u, v)
            network.add_edge(u, v)
            # assign edge force density
            q = network0.edge_forcedensity(edge)
            network.edge_forcedensity(edge, q)

# ==========================================================================
# Angle of the assembly step
# ==========================================================================

    angle_long = l_start + (l_end - l_start) * ((step - 1) / (max_step - 1)) ** l_exp
    print(f"Angle long span: {degrees(angle_long):.2f}")

    if step <= max_step_sequential_short:
        angle_short = s_start + (s_end - s_start) * ((step - 1) / (max_step - 3)) ** s_exp
        print(f"Angle short span: {degrees(angle_short):.2f}")

# ==========================================================================
# Optimization parameters
# ==========================================================================

    parameters = []

    for edge in network.edges():
        _qmax = qmax
        if step >= 3:
            _qmax = qmax * 1.0
        parameter = EdgeForceDensityParameter(edge, qmin, _qmax)
        parameters.append(parameter)

    if parametrize_z_spine and freeze_spine:
        print("Adding z coordinate of spine supports as optimization parameters...")
        for node in spine_nodes:
            if node in supported_vkeys:
                continue
            z = network.node_attribute(node, "z")
            parameter = NodeAnchorZParameter(node, z - ztol, z + ztol)
            parameters.append(parameter)

# ==========================================================================
# Goals
# ==========================================================================

    # goals
    goals = []
    print()

    # maintain horizontal projection
    if add_node_xy_goal:
        goals_xy = []
        # for node in nodes_step:  # NOTE: nodes added at this assembly step
        for node in network.nodes_free():
            x, y, z = network_base.node_coordinates(node)
            for goal, xy in zip((NodeXCoordinateGoal, NodeYCoordinateGoal), (x, y)):
                goal = goal(node, xy, weight_node_xy_goal)
                goals_xy.append(goal)
        print(f"{int(len(goals_xy) / 2)} NodeXCoordinateGoal")
        print(f"{int(len(goals_xy) / 2)} NodeYCoordinateGoal")

        goals.extend(goals_xy)

    # best-fit history
    if add_node_bestfit_goal:
        goals_bestfit = []
        for node in network.nodes_free():  # NOTE: nodes added at all previous steps
            if node in nodes_step: # or node in hexagon:
                continue
            xyz = network.node_coordinates(node)
            goal = NodePointGoal(node, xyz, weight_node_bestfit_goal)
            goals_bestfit.append(goal)
        print(f"{len(goals_bestfit)} NodeBestfitGoal")

        goals.extend(goals_bestfit)

    # edge direction goal
    profile_lines = []
    goals_edge_direction = []
    if add_edge_direction_profile_goal:
        for edge in profile_edges:
            # take edge only if it belongs to the current assembly step
            # if edge2step.get(edge, edge2step.get((edge[1], edge[0]))) != step:
            edge_step = edge2step.get(edge, edge2step.get((edge[1], edge[0])))
            if edge_step > step or edge_step == 0:
                continue

            # if edge_step != step:
                # continue

            if edge in edges_span_short or (edge[1], edge[0]) in edges_span_short:
                if edge_step > max_step_sequential_short:
                    continue

            angle = l_start + (l_end - l_start) * ((edge_step - 1) / (max_step - 1)) ** l_exp

            if edge in profile_edges_span_short:
                angle = s_start + (s_end - s_start) * ((edge_step - 1) / (max_step - 3)) ** s_exp

            u, v = edge
            factor = 1.0
            if not network.has_edge(u, v):
                u, v = v, u
                factor = -1.0

            assert network.has_edge(u, v), f"Edge {edge} at step {edge_step} not found"

            edge = (u, v)

            vector0 = mesh.edge_vector(u, v)
            ortho = cross_vectors(vector0, [0.0, 0.0, 1.0])
            vector = rotate_points([vector0], angle * factor, axis=ortho, origin=[0.0, 0.0, 0.0])[0]

            goal = EdgeAngleGoal((u, v), vector=[0.0, 0.0, 1.0], target= pi * 0.5 - angle * factor)
            goals_edge_direction.append(goal)

            # goal = EdgeDirectionGoal((u, v), target=vector, weight=weight_edge_direction_profile_goal)
            # goals_edge_direction.append(goal)

            # for viz
            start = network_base.node_coordinates(u)
            end = add_vectors(start, scale_vector(normalize_vector(vector), 0.25))
            line = Line(start, end)
            profile_lines.append(line)

        print(f"{len(goals_edge_direction)} EdgeDirectionGoal")
        goals.extend(goals_edge_direction)

    # equalize length of edges parallel to the spine
    if add_edge0_length_equal_goal:
        goals_edge0_length_equal = []
        _edges = []
        for edge in edges0:

            edge_step = edge2step.get(edge, edge2step.get((edge[1], edge[0])))

            # edge is not in current step
            if edge_step != step:
                continue

            if edge in edges_span_short or (edge[1], edge[0]) in edges_span_short:
                if edge_step > max_step_sequential_short:
                    continue

            u, v = edge
            if not network.has_edge(u, v):
                u, v = v, u
            _edges.append((u, v))

        if len(_edges) > 0:
            goal = EdgesLengthEqualGoal(_edges, weight=weight_edge0_length_equal_goal)
            goals_edge0_length_equal.append(goal)

        print(f"{len(goals_edge0_length_equal)} Edges0EqualLengthGoal")
        goals.extend(goals_edge0_length_equal)

    # equalize length of edges perpendicular to the spine
    if add_edge1_length_equal_goal:
        goals_edge1_length_equal = []
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

        if len(_edges) > 0:
            goal = EdgesLengthEqualGoal(_edges, weight=weight_edge1_length_equal_goal)
            goals_edge1_length_equal.append(goal)

        print(f"{len(goals_edge1_length_equal)} Edges1EqualLengthGoal")
        goals.extend(goals_edge1_length_equal)

    # reduce reaction forces at spine nodes
    if freeze_spine and add_node_spine_reaction_goal:
        goals_spine_reactions = []
        for node in spine_nodes:
            if node in supported_vkeys:
                continue
            goal = NodeResidualForceGoal(node, 0.0, weight_node_spine_reaction_goal)
            goals_spine_reactions.append(goal)

        print(f"{len(goals_spine_reactions)} NodeSpineReactionGoal")
        goals.extend(goals_spine_reactions)

    # spine nodes no torsion goal
    if add_node_spine_notorsion_goal:
        goals_nodes_spine_notorsion = []
        for strip in spine_strip_edges:
            for u, v in strip:
                if u in hexagon or v in hexagon:
                    continue
                if not network.has_edge(u, v):
                    u, v = v, u
                edge = (u, v)
                vector = network_spine.edge_vector(u, v)
                goal = EdgeDirectionGoal(edge, vector, weight_node_spine_notorsion_goal)
                goals_nodes_spine_notorsion.append(goal)

        print(f"{len(goals_nodes_spine_notorsion)} NodesSpineNoTorsionGoal")
        goals.extend(goals_nodes_spine_notorsion)

# ==========================================================================
# Loss function
# ==========================================================================

    loss = Loss(SquaredError(goals), L2Regularizer(alpha))

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

    if export:
        print("\nExporting assembly step networks...")
        print("Deleting supported edges...")
        print("Zeroing out loads on supported spine nodes...")

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
# Generate mesh for final step
# ==========================================================================

# create mesh from last network
cycles = network_find_cycles(network)
vertices = {vkey: network.node_coordinates(vkey) for vkey in network.nodes()}
mesh = Mesh.from_vertices_and_faces(vertices, cycles)
mesh.delete_face(0)
mesh.cull_vertices()

# store assembly steps at the vertices of the mesh
for vkey in mesh.vertices():
    vertex_step = vkey2step[vkey]
    mesh.vertex_attribute(vkey, "step", vertex_step)

# ==========================================================================
# Update loads
# ==========================================================================

if fofin_last or bestfit_last:
    print("Updating loads on network from recomputed mesh")
    for node in network.nodes():
        vertex_area = mesh.vertex_area(node)
        network.node_load(node, load=[0.0, 0.0, vertex_area * pz * -1.0])

# ==========================================================================
# Update loads
# ==========================================================================

if fofin_last:
    network = fdm(network)

# ==========================================================================
# Optimization parameters
# ==========================================================================

if bestfit_last:

    parameters = []

    for edge in network.edges():
        parameter = EdgeForceDensityParameter(edge, qmin, qmax)
        parameters.append(parameter)

# ==========================================================================
# Goals
# ==========================================================================

    # goals
    goals = []

    # best-fit history
    for node in network.nodes_free():  # NOTE: nodes added at all previous steps
        xyz = network.node_coordinates(node)
        goal = NodePointGoal(node, xyz)
        goals.append(goal)

# ==========================================================================
# Loss function
# ==========================================================================

    loss = Loss(MeanSquaredError(goals), L2Regularizer(0.1))

# ==========================================================================
# Constrained form-finding
# ==========================================================================

    print("\nBest fitting last surface with updated loads")
    network = constrained_fdm(network,
                              optimizer=opt(),
                              loss=loss,
                              parameters=parameters,
                              maxiter=maxiter,
                              tol=1e-9,
                              callback=None)

    # print out network statistics
    network.print_stats()

# ==========================================================================
# Export
# ==========================================================================

if export:
    for name, datastruct in {"network": network, "mesh": mesh}.items():

        filepath = os.path.join(DATA, f"tripod_{name}_dual_3d.json")
        datastruct.to_json(filepath)

    # spine center
    spine_center = Network()
    for strip in spine_strip_edges:
        nodes = []
        for u, v in strip:
            x, y, z = network.edge_midpoint(u, v)
            node = spine_center.add_node(x=x, y=y, z=z)
            nodes.append(node)
        for u, v in pairwise(nodes):
            spine_center.add_edge(u, v)

    filepath = os.path.join(DATA, f"tripod_network_dual_spine_center_corrected_3d.json")
    spine_center.to_json(filepath)

    print("\nExported JSON files!")

# ==========================================================================
# Visualization
# ==========================================================================

if view:

    viewer = Viewer(width=1600, height=900, show_grid=False)

    viewer.view.color = (0.1, 0.1, 0.1, 1)  # change background to black

    viewer.add(network, edgecolor="force", show_loads=False, show_reactions=True, edgewidth=(0.01, 0.03))
    viewer.add(mesh, show_points=False, show_lines=False, opacity=0.5)

    viewer.add(network_spine, as_wireframe=True, show_points=False)
    viewer.add(network_base, as_wireframe=True, show_points=False)

    viewer.show()
