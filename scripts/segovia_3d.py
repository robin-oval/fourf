import os

from math import pi, radians, degrees

from numpy import mean, std

from compas.colors import Color, ColorMap
from compas.geometry import Line, Polyline, distance_point_point, length_vector, length_vector_xy, sum_vectors, cross_vectors, rotate_points, bounding_box
from compas.geometry import angle_vectors, project_point_plane
from compas.geometry import add_vectors, subtract_vectors, scale_vector, normalize_vector

from compas.utilities import pairwise

from compas_quad.datastructures import CoarseQuadMesh, QuadMesh
from compas_quad.coloring import quad_mesh_polyedge_2_coloring

from compas_view2.shapes import Arrow

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

from jax_fdm.parameters import EdgeForceDensityParameter, NodeAnchorXParameter, NodeAnchorYParameter

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

from jax_fdm.visualization import Viewer, LossPlotter


# ==========================================================================
# Project description
# ==========================================================================

"""
system: masonry vault built using thin-tile vaulting of a herringbone pattern
design: threefold arch like monkey saddle with creases
objectives/constraints to integrate:
- fit desired space -> target XY projection
- planar parts of the spine for simple centering -> plane constraint
- starting assembly without centering from spine -> succession of rising arches parallel to the spine
  (directly using the edge slope objective or indirectly using the residual force objective)
- regular width of the brick courses -> constant length of the edges in the transverse direction of the spine

TODO:

- constrain spine polyedges to be straight with plane goals
- sequential form finding
- add the X and Y coordinates of the supported nodes (except those of the spine) as parameters
and add goals to guide them on radial lines
- add loads as parameters to update them during form finding to match the self-weight
"""

# ==========================================================================
# Inputs
# ==========================================================================

r = 2.5  # circumcircle radius [m]
pos_angles = 0.0, 3 * pi / 4, 3 * pi / 2  # triangle parameterisation angle [radians]
wid_angles = pi / 6, pi / 6, pi / 6  # triangle to hexagon parameterisation angle [radians]
offset1, offset2 = 0.9, 0.95  # offset factor the unsupported and supported edges inward respectively [-]
target_edge_length = 0.25  # 0.25 [m]

brick_length, brick_width, brick_thickness = 0.24, 0.125, 0.04  # [m]
brick_layers = 4  # [-]
brick_density = 12.0  # [kN/m3]
comp_strength = 6.0  # [MPa]

course_width = 0.125 * 1.5

dead_load = 0.0  # dead load [kN/m2]
pz = brick_density * brick_thickness * brick_layers + dead_load  # vertical area load (approximated self-weight + uniform dead load) [kN/m2]

qmin, qmax = None, -1e-1  # bound on force densities [kN/m]
add_supports_as_parameters = True
ctol = 0.5  # 0.5  bound on supports X and Y positions

opt = LBFGSB  # optimization solver
maxiter = 10000  # maximum number of iterations
tol = 1e-6  # optimization tolerance

# aim for target positions
add_singularity_xyz_goal = True
cross_height = 2.2  # height of spine cross [m]
weight_node_target_goal = 10.0

# keep spine nodes xy position
add_spine_xy_goal = True
weight_spine_xy_goal = 10.0

# keep spine planar
add_spine_planarity_goal = True  # True
weight_spine_planarity_goal = 10.0

# vertex projection goal to cover the desire space
add_horizontal_projection_goal = True  # True
weight_projection_goal = 10.0

# edge length goal to obtain constant brick course widths
add_edge_length_profile_goal = True  # True
weight_edge_length_profile_goal = 1.0

# profile edges direction goal
add_edge_direction_profile_goal = True
s_start, s_end, s_exp = radians(80), radians(30), 1.0  # 70, 30 minimum and maximum angles and variation exponent [-]
weight_edge_direction_profile_goal = 1.0

# edge length goal to obtain constant brick course widths
add_edge_length_strips_goal = False  # False
weight_edge_length_strips_goal = 1.0

# edge equalize length goal to obtain constant brick course widths
add_edge_length_equal_strips_goal = True
weight_edge_length_equal_strips_goal = 10.0  # 1.0

# edge equalize length goals to polyedges parallel to spine
add_edge_length_equal_polyedges_goal = False
weight_edge_length_equal_polyedges_goal = 1.0

# shape the normal of the polyedges running from the singularity to the unsupported boundary
# NOTE: currently not working properly! do not use!
add_node_tangent_goal = False
t_start, t_end, t_exp = radians(70), radians(30), 1.0  # minimum and maximum angles from Z and variation exponent [-]
weight_node_tangent_goal = 10.0

# plane goal
add_edge_plane_goal = False
weight_edge_plane_goal = 10.0

# controls
optimize = True
record = False
add_constraints = False
view = True
view_node_tangents = False
results = False
export = True

# ==========================================================================
# Load FD network
# ==========================================================================

network = FDNetwork.from_json(os.path.join(DATA, "tripod_network_2d.json"))
network0 = network.copy()

# ==========================================================================
# Load mesh
# ==========================================================================

vertices, faces = threefold_vault(r, pos_angles, wid_angles, offset1=offset1, offset2=offset2)
mesh = CoarseQuadMesh.from_vertices_and_faces(vertices, faces)
mesh = quadmesh_densification(mesh, target_edge_length)

# ==========================================================================
# Update node loads
# ==========================================================================

for vkey in mesh.vertices():
    xyz = network.node_coordinates(vkey)
    mesh.vertex_attributes(vkey, names="xyz", values=xyz)

for node in network.nodes():
    vertex_area = mesh.vertex_area(node)
    network.node_load(node, load=[0.0, 0.0, vertex_area * pz * -1.0])

# ==========================================================================
# Network parts
# ==========================================================================

supp_pkeys = support_shortest_boundary_polyedges(mesh)
supports = set([vkey for pkey in supp_pkeys for vkey in mesh.polyedge_vertices(pkey)])

pkey2type = polyedge_types(mesh, supp_pkeys)

edge2color = quad_mesh_polyedge_2_coloring(mesh, edge_output=True)
color_ortho, color_paral = None, None
for pkey, ptype in pkey2type.items():
    if ptype == 'spine':
        color_paral = edge2color[tuple(mesh.polyedge_vertices(pkey)[:2])]
        color_ortho = 1 - color_paral
        break
edges_ortho = [edge for edge, color in edge2color.items() if color == color_ortho]

### ASSEMBLY SEQUENCE ###

pkey2step = quadmesh_polyedge_assembly_sequence(mesh, pkey2type)
steps = set([step for step in pkey2step.values() if step is not None])
min_step, max_step = int(min(steps)), int(max(steps))

# singularity node
snode = [node for node in network.nodes() if network.degree(node) == 6].pop()

# spine polyedges (from singularity to supported boundary)
spine_polyedges = []
for pkey, polyedge in mesh.polyedges(data=True):
    cdt1 = mesh.vertex_degree(polyedge[0]) == 6 and polyedge[-1] in supports
    cdt2 = mesh.vertex_degree(polyedge[-1]) == 6 and polyedge[0] in supports
    if cdt1 or cdt2:
        polyedge = list(reversed(polyedge)) if mesh.vertex_degree(polyedge[-1]) == 6 else polyedge
        spine_polyedges.append(polyedge)

# spine nodes
spine_nodes = set()
for polyedge in spine_polyedges:
    spine_nodes.update(polyedge)

# profile curves (from singularity to unsupported boundary)
profile_polyedges = []
for pkey, polyedge in mesh.polyedges(data=True):
    cdt1 = mesh.vertex_degree(polyedge[0]) == 6 and polyedge[-1] not in supports
    cdt2 = mesh.vertex_degree(polyedge[-1]) == 6 and polyedge[0] not in supports
    if cdt1 or cdt2:
        polyedge = list(reversed(polyedge)) if mesh.vertex_degree(polyedge[-1]) == 6 else polyedge
        profile_polyedges.append(polyedge)

# profile nodes
profile_nodes = set()
for polyedge in profile_polyedges:
    profile_nodes.update(polyedge)

# full profile strips by assembly step - parallel to profile curve
# NOTE: assumes all profile polyedges have all the same length
profile_strips = {}
profile_strips_split = {}
for i in range(max_step):
    strips = []
    strips_split = []
    for polyedge in profile_polyedges:
        u, v = polyedge[i], polyedge[i + 1]
        strip = mesh.collect_strip(u, v)
        strips.append(strip)

        # split polyedge in two sub-polyedges at the profile edge
        # NOTE: order strip to start from boundary
        idx = strip.index((u, v))
        side_a = strip[:idx]
        side_b = list(reversed(strip[idx + 1:]))
        strip_split = [side_a, side_b]
        strips_split.append(strip_split)

    profile_strips[i] = strips
    profile_strips_split[i] = strips_split

# ==========================================================================
# Network parts
# ==========================================================================

eqnetwork = fdm(network)

# ==========================================================================
# Optimization parameters
# ==========================================================================

if optimize:

    print()

    parameters = []

    for edge in network.edges():
        parameter = EdgeForceDensityParameter(edge, qmin, qmax)
        parameters.append(parameter)

    parameters_supports = []
    if add_supports_as_parameters:
        for node in network.nodes_anchors():
            x, y, z = network.node_coordinates(node)
            # skip spine supports
            if node in spine_nodes:
                continue
            for coordinate, parameter in zip((x, y), (NodeAnchorXParameter, NodeAnchorYParameter)):
                parameter = parameter(node, coordinate - ctol, coordinate + ctol)
                parameters_supports.append(parameter)

# ==========================================================================
# Goals
# ==========================================================================

    # target position of singularity node in spine
    goals_singularity_xyz = []
    if add_singularity_xyz_goal:
        x, y, z = network.node_coordinates(snode)
        goal = NodePointGoal(snode, target=[x, y, cross_height], weight=weight_node_target_goal)
        goals_singularity_xyz.append(goal)
        print('{} NodeSingularityXYZGoal'.format(len(goals_singularity_xyz)))

    goals_spine_xy =[]
    if add_spine_xy_goal:
        for polyedge in spine_polyedges:
            for node in polyedge:
                if node == snode or node in supports:
                    continue
                x, y, z = network.node_coordinates(node)
                for goal, xy in zip((NodeXCoordinateGoal, NodeYCoordinateGoal), (x, y)):
                    goal = goal(node, xy, weight_spine_xy_goal)
                    goals_spine_xy.append(goal)
        print('{} Spine NodeXCoordinateGoal and NodeYCoordinateGoal'.format(len(goals_spine_xy)))

    # preserve horizontal projection - other nodes
    goals_projection = []
    if add_horizontal_projection_goal:

        for node in network.nodes():
            if node not in supports and node not in spine_nodes and node != snode:
                x, y, z = network.node_coordinates(node)
                for goal, xy in zip((NodeXCoordinateGoal, NodeYCoordinateGoal), (x, y)):
                    goal = goal(node, xy, weight_projection_goal)
                    goals_projection.append(goal)
        print('{} NodeXCoordinateGoal and NodeYCoordinateGoal'.format(len(goals_projection)))

    # spline planarity
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

    goals_profile_direction = []
    profile_lines = []
    if add_edge_direction_profile_goal:
        for polyedge in profile_polyedges:
            n = len(polyedge)
            for i, edge in enumerate(pairwise(polyedge)):
                angle = s_start + (s_end - s_start) * (i / (n - 1)) ** s_exp
                # print(degrees(angle))
                vector0 = mesh.edge_vector(*edge)
                ortho = cross_vectors(vector0, [0.0, 0.0, 1.0])
                vector = rotate_points([vector0], pi / 2 - angle, axis=ortho, origin=[0.0, 0.0, 0.0])[0]
                goal = EdgeDirectionGoal(edge, target=vector, weight=weight_edge_direction_profile_goal)
                goals_profile_direction.append(goal)

                # for viz
                start = mesh.vertex_coordinates(edge[0])
                end = add_vectors(start, scale_vector(normalize_vector(vector), target_edge_length))
                line = Line(start, end)
                profile_lines.append(line)

        print('{} EdgeProfileDirectionGoal'.format(len(goals_profile_direction)))

    # length of profile curves edges
    goals_profile_length = []
    if add_edge_length_profile_goal:
        for polyedge in profile_polyedges:

            for i, edge in enumerate(pairwise(polyedge)):
                factor = 1.0
                if i == 0:
                    factor = 2.0
                goal = EdgeLengthGoal(edge, target=course_width * factor, weight=weight_edge_length_profile_goal)

                goals_profile_length.append(goal)

        print('{} EdgeProfileLengthGoal'.format(len(goals_profile_length)))

    # fixed target length of strip edges
    goals_length_strips = []
    if add_edge_length_strips_goal:
        for i, strips_split in profile_strips_split.items():
            for strip in strips_split:
                for side in strip:
                    # NOTE: Include supported or not?
                    indices = list(range(1, len(side)))
                    for j in indices:
                        edge = side[j]
                        goal = EdgeLengthGoal(edge, target=course_width, weight=weight_edge_length_strips_goal)
                        goals_length_strips.append(goal)
        print('{} EdgeStripsLengthGoal'.format(len(goals_length_strips)))

    # equalize length of strip edges
    goals_length_equal_strips = []
    if add_edge_length_equal_strips_goal:
        for step, strips in profile_strips.items():
            # if step == 0:  # NOTE: skips strips connected to the spine
                # continue
            for strip in strips:
                edges = strip
                goal = EdgesLengthEqualGoal(edges, weight=weight_edge_length_equal_strips_goal)
                goals_length_equal_strips.append(goal)
        print('{} EdgeStripsEqualLengthGoal'.format(len(goals_length_equal_strips)))

    # equalize length of edges parallel to spine, per polyedge
    goals_length_equal_polyedges = []
    if add_edge_length_equal_polyedges_goal:
        for pkey, polyedge in mesh.polyedges(True):
            ptype = pkey2type[pkey]
            if ptype != "span":
                continue
            edges = []
            for u, v in pairwise(polyedge):
                if not mesh.has_edge((u, v)):
                    u, v = v, u
                edges.append((u, v))
            goal = EdgesLengthEqualGoal(edges, weight=weight_edge_length_equal_polyedges_goal)
            goals_length_equal_polyedges.append(goal)
        print('{} EdgePolyedgesEqualLengthGoal'.format(len(goals_length_equal_polyedges)))

    # control the normal at the nodes of the edges parallel to the spine
    goals_normal = []
    nodes_normal = set()
    if add_node_tangent_goal:
        n = max_step - 1
        for i in range(1, max_step):  # NOTE: skip spine
            if i != 2:
                continue

            angle = t_start + (t_end - t_start) * ((i - 1) / (n - 1)) ** t_exp
            print(i, degrees(angle))
            for pkey, polyedge in mesh.polyedges(True):
                step = pkey2step[pkey]
                if step != i:
                    continue
                for node in polyedge:
                    if node in supports or node in profile_nodes:
                        continue
                    nodes_normal.add(node)
                    goal = NodeTangentAngleGoal(node, vector=[0.0, 0.0, 1.0], target=angle, weight=weight_node_tangent_goal)
                    goals_normal.append(goal)

        print('{} NodeTangentAngleGoal'.format(len(goals_normal)))

    # constrain strip edges to a plane
    goals_plane = []
    if add_edge_plane_goal:
        for step in range(max_step):
            back = 1
            if step == 0:
                back = 0

            for strips_a, strips_b in zip(profile_strips_split[step - back], profile_strips_split[step]):
                for strip_a, strip_b in zip(strips_a, strips_b):
                    for edge_a, edge_b in zip(strip_a, strip_b):
                        u, v = edge_b
                        if u in supports and v in supports:
                            continue
                        vector = network.edge_vector(*edge_a)
                        origin = network.node_coordinates(u)
                        normal = cross_vectors(vector, [0.0, 0.0, 1.0])
                        plane = (origin, normal)
                        goal = NodePlaneGoal(v, plane, weight=weight_edge_plane_goal)
                        goals_plane.append(goal)

        print('{} NodesStripsPlaneGoal'.format(len(goals_plane)))

# ==========================================================================
# Loss function
# ==========================================================================

    loss = Loss(
                SquaredError(goals=goals_singularity_xyz, name='NodeXYZSingularityGoal', alpha=1.0),
                SquaredError(goals=goals_spine_xy, name='NodeSpineXYGoal', alpha=1.0),
                SquaredError(goals=goals_projection, name='NodeXYProjectionGoal', alpha=1.0),
                SquaredError(goals=goals_spine_planarity, name='EdgeSpinePlanarityGoal', alpha=1.0),
                SquaredError(goals=goals_profile_direction, name='EdgeProfileDirectionGoal', alpha=1.0),
                SquaredError(goals=goals_profile_length, name='EdgeProfileLengthGoal', alpha=1.0),
                SquaredError(goals=goals_length_strips, name='EdgeLengthStripsGoal', alpha=1.0),
                PredictionError(goals=goals_length_equal_strips, name='EdgesLengthEqualStripsGoal', alpha=1.0),
                PredictionError(goals=goals_length_equal_polyedges, name='EdgesLengthEqualPolyedgesGoal', alpha=1.0),
                SquaredError(goals=goals_normal, name='NodeTangentAngleGoal', alpha=1.0),
                SquaredError(goals=goals_plane, name='NodePlaneGoal', alpha=1.0),
                )

# ==========================================================================
# Constraints
# ==========================================================================

    constraints = []
    if add_constraints:
        for polyedge in spine_polyedges:
            for node in polyedge:
                if node == snode:
                    continue
                constraint = NodeZCoordinateConstraint(node, bound_low=0.0, bound_up=cross_height)
                constraints.append(constraint)

# ==========================================================================
# Solve constrained form-finding problem
# ==========================================================================

    optimizer = opt()
    recorder = None
    if record:
        recorder = OptimizationRecorder(optimizer)

    network = constrained_fdm(network,
                              optimizer=optimizer,
                              parameters=parameters + parameters_supports,
                              constraints=constraints,
                              loss=loss,
                              maxiter=maxiter,
                              tol=tol,
                              callback=recorder
                              )

    cnetwork1 = network.copy()

    if record:
        plotter = LossPlotter(loss, network, dpi=150, figsize=(8, 4))
        plotter.plot(recorder.history)
        plotter.show()

# ==========================================================================
# Export
# ==========================================================================

if export:
    filepath = os.path.join(DATA, "tripod_network_3d.json")
    network.to_json(filepath)
    print("\nExported JSON file!")

# ==========================================================================
# Visualization
# ==========================================================================

if view:
    viewer = Viewer(width=1600, height=900, show_grid=False)

    viewer.view.color = (0.1, 0.1, 0.1, 1)  # change background to black

    # update mesh
    for vkey in mesh.vertices():
        xyz = network.node_coordinates(vkey)
        mesh.vertex_attributes(vkey, names="xyz", values=xyz)

    viewer.add(mesh, show_lines=False, show_points=False, opacity=0.5)

    # color strip edges according to sequence
    maxs = max_step
    cmap = ColorMap.from_mpl("viridis")
    for i in range(maxs):

        for strip in profile_strips[i]:
            for edge in strip:
                line = Line(*network.edge_coordinates(*edge))
                viewer.add(line, linecolor=cmap(i, maxval=maxs), linewidth=5.)

            for polyedge in zip(*strip):
                polyline = Polyline([network.node_coordinates(n) for n in polyedge])
                viewer.add(polyline)

    # viewer.add(eqnetwork, as_wireframe=True)
    viewer.add(network0, as_wireframe=True, show_points=False)

    if optimize:
        # profile lines
        for line in profile_lines:
            viewer.add(line)

        # NOTE: delete supported edges
        print("\nDeleting supported edges for viz and results...")
        edges = list(network.edges())
        for u, v in edges:
            if u in supports and v in supports:
                try:
                    network.delete_edge(u, v)
                except KeyError:
                    u, v = v, u
                    network.delete_edge(u, v)

        network = fdm(network)  # NOTE: recompute reaction forces

    # add solid network to display
    viewer.add(network,
               show_edges=False,
               edgewidth=(0.003, 0.02),
               edgecolor="force",
               reactionscale=1.,
               loadscale=1.)

    # node tangent arrows
    if view_node_tangents:
        angles_mesh = []
        tangent_angles_mesh = []
        arrows = []
        tangent_arrows = []
        vkeys = []

        for vkey in mesh.vertices():
        # for vkey in nodes_normal:
            vkeys.append(vkey)

            xyz = mesh.vertex_coordinates(vkey)

            if vkey in supports or vkey in spine_nodes:
                continue

            normal = mesh.vertex_normal(vkey)
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
            color = cmap(tangent_angle, minval=min_angle, maxval=max_angle)
            # viewer.add(arrow)
            viewer.add(tarrow, facecolor=color, show_edges=False, opacity=0.8)
            # print(f"node: {vkey}\tangle: {angle:.2f}\ttangent angle: {tangent_angle:.2f}\ttangent angle 2: {90-angle:.2f}")

#  ==========================================================================
# Results
# ==========================================================================

if results:
    print()
    # new meshes to compute areas
    mesh_ff_xyz = mesh.copy()  # mesh that follows the form found network
    mesh_ff_xy = mesh.copy()  # mesh that follows the projection of the form found network
    for vkey in mesh.vertices():
        for coord, new_coord in zip('xyz', network.node_coordinates(vkey)):
            mesh_ff_xyz.vertex[vkey][coord] = new_coord
            if not coord == 'z':
                mesh_ff_xy.vertex[vkey][coord] = new_coord

    # edge force densities
    sorted_fds = sorted([network.edge_forcedensity(edge) for edge in network.edges()])
    print('Minimum force density of {} kN/m and maximum force density of {} kN/m'.format(round(sorted_fds[0], 2), round(sorted_fds[-1], 2)))
    # edge lengths
    sorted_lengths = sorted([network.edge_length(*edge) for edge in network.edges()])
    print('Minimum length of {} m and maximum length of {} m'.format(round(sorted_lengths[0], 2), round(sorted_lengths[-1], 2)))

    # vault surface to build
    mesh_area = mesh_ff_xyz.area()
    brick_area = brick_length * brick_width # top/bottom surface - include mortar thickness as percentage?
    print('Surface area of {} m2, requiring about {} bricks'.format(round(mesh_area, 1), int(mesh_area / brick_area)))
    max_z = max([network.node[node]['z'] for node in network.nodes()])
    print('Maximum height to reach of {} m'.format(round(max_z, 2)))

    # space use - height, area and volume
    box = bounding_box([network.node_coordinates(node) for node in network.nodes()])
    dx, dy, dz = box[1][0] - box[0][0], box[3][1] - box[0][1], box[4][2] - box[0][2]
    print('Bounding box of {} m x {} m x {} m'.format(round(dx, 1), round(dy, 1), round(dz, 1)))
    apex = max([network.node[node]['z'] for node in network.nodes() if network.degree(node) == 6])
    print('Height of spine cross of {} m'.format(round(apex, 2)))
    mesh_area_xy = mesh_ff_xyz.area()
    print('Vault covered area {} m2'.format(round(mesh_area_xy, 1)))
    comfort_height = 2.0
    comfort_surface = sum([mesh_ff_xy.vertex_area(vkey) for vkey in mesh.vertices() if mesh_ff_xyz.vertex_coordinates(vkey)[2] > comfort_height])
    print('Vault covered area of {} m2 above {} m'.format(round(comfort_surface, 1), comfort_height))

    # structural design
    #forces on foundations
    reactions = []
    for pkey, ptype in pkey2type.items():
        if ptype == 'support':
            reactions.append(sum_vectors([network.node_reaction(node) for node in mesh.polyedge_vertices(pkey)]))
    reactions = [(length_vector_xy(xyz), xyz[2]) for xyz in reactions]
    print('Horizontal and vertical action forces on supports {} kN'.format(reactions))
    # forces
    sorted_forces = sorted([network.edge_force(edge) for edge in network.edges()])
    print('Minimum force of {} kN and maximum force of {} kN'.format(round(sorted_forces[0], 2), round(sorted_forces[-1], 2)))
    # stresses
    edge2stress = {}
    for u, v in network.edges():
        edge = (u, v)
        force = network.edge_force(edge)
        a = mesh.edge_midpoint(mesh.face_vertex_after(mesh.halfedge[u][v], v, 1), mesh.face_vertex_after(mesh.halfedge[u][v], v, 2)) if mesh.halfedge[u][v] is not None else mesh.edge_midpoint(u, v)
        b = mesh.edge_midpoint(mesh.face_vertex_after(mesh.halfedge[v][u], u, 1), mesh.face_vertex_after(mesh.halfedge[v][u], u, 2)) if mesh.halfedge[v][u] is not None else mesh.edge_midpoint(v, u)
        width = distance_point_point(a, b) / 2
        edge2stress[(u, v)] = force / (width * brick_thickness * brick_layers)
    sorted_sresses = sorted([stress for stress in edge2stress.values()])
    print('Minimum stress of {} MPa and maximum stress of {} MPa'.format(round(sorted_sresses[0] / 1000, 2), round(sorted_sresses[-1] / 1000, 2)))
    print('Minimum stress utilization of {} and maximum stress utilization of {} '.format(round(sorted_sresses[0] / 1000 / comp_strength, 2), round(sorted_sresses[-1] / 1000 / comp_strength, 2)))

    # vaulting
    brick_course_widths = [network.edge_length(*edge) for edge in edges_ortho]
    print('Brick course width of mean {} m and standard deviation {} m'.format(round(mean(brick_course_widths), 2), round(std(brick_course_widths), 2)))

if view:
    viewer.show()
