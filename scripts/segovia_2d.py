from collections import defaultdict

from math import pi, radians, degrees

from numpy import mean, std

from operator import itemgetter

from compas.colors import Color, ColorMap
from compas.geometry import Line, Polyline, distance_point_point, length_vector, length_vector_xy, sum_vectors, cross_vectors, rotate_points, bounding_box
from compas.geometry import add_vectors, scale_vector, normalize_vector

from compas_quad.datastructures import CoarseQuadMesh
from compas_quad.coloring import quad_mesh_polyedge_2_coloring

from fourf.topology import threefold_vault
from fourf.topology import quadmesh_densification
from fourf.support import support_shortest_boundary_polyedges
from fourf.support import polyedge_types
from fourf.sequence import quadmesh_polyedge_assembly_sequence
from fourf.utilities import mesh_to_fdnetwork
from fourf.view import view_f4

from jax_fdm.equilibrium import fdm, constrained_fdm
from jax_fdm.optimization import LBFGSB, SLSQP
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
from jax_fdm.visualization import Viewer

from compas.utilities import pairwise


### PROJECT ###

# system: masonry vault built using thin-tile vaulting of a herringbone pattern
# design: threefold arch like monkey saddle with creases
# objectives/constraints to integrate:
# - fit desired space -> target XY projection
# - planar parts of the spine for simple centering -> plane constraint
# - starting assembly without centering from spine -> succession of rising arches parallel to the spine
#   (directly using the edge slope objective or indirectly using the residual force objective)
# - regular width of the brick courses -> constant length of the edges in the transverse direction of the spine

# TO ADD:

# - constrain spine polyedges to be straight with plane goals
# - sequential form finding
# - add the X and Y coordinates of the supported nodes (except those of the spine) as parameters
# and add goals to guide them on radial lines
# - add loads as parameters to update them during form finding to match the self-weight

### INPUTS ###

r = 2.5  # circumcircle radius [m]
pos_angles = 0.0, 3 * pi / 4, 3 * pi / 2  # triangle parameterisation angle [radians]
wid_angles = pi / 6, pi / 6, pi / 6  # triangle to hexagon parameterisation angle [radians]
offset1, offset2 = 0.9, 0.95  # offset factor the unsupported and supported edges inward respectively [-]
target_edge_length = 0.20  # 0.25 [m]
support_raised_height = [0.0, 0.0, 0.0]  # raised height of each support [m]

brick_length, brick_width, brick_thickness = 0.24, 0.125, 0.04 # [m]
brick_layers = 0  # [-]
brick_density = 12.0  # [kN/m3]
comp_strength = 6.0  # [MPa]

dead_load = 0.0  # dead load [kN/m2]
pz = - (brick_density * brick_thickness * brick_layers) + dead_load  # vertical load (approximated self-weight + uniform dead load) [kN/m2]
q0 = -1.0  # initial force densities [kN/m]

opt = LBFGSB  # optimization solver
qmin, qmax = None, -1e-1  # bound on force densities [kN/m]
add_supports_as_parameters = True
ctol = 0.5  # bound on supports X and Y positions
maxiter = 1000  # maximum number of iterations
tol = 1e-6  # optimization tolerance

# aim for target positions
add_node_target_goal = True
cross_height = 0.0  # height of spine cross [m]
perimeter_height = 0.0  # height of perimetral arches [m]
weight_node_target_goal = 10.0

# keep spine planar
add_spine_planarity_goal = True
weight_spine_planarity_goal = 10.0

# keep spine planar
add_profile_planarity_goal = True
weight_profile_planarity_goal = 10.0

# edge length goal to obtain constant brick course widths
add_edge_length_profile_goal = True
weight_edge_length_profile_goal = 10.0

# equal edge length at the spine
add_edge_length_equal_spine_goal = False
weight_edge_length_equal_spine_goal = 1.0

# edge length goal to obtain constant brick course widths
add_edge_length_strips_goal = True
weight_edge_length_strips_goal = 0.1

# edge equalize length goal to obtain constant brick course widths
add_edge_length_equal_strips_goal = False
weight_edge_length_equal_strips_goal = 1.

# plane goal
add_edge_plane_goal = True
weight_edge_plane_goal = 0.1


# controls
optimize = True
add_constraints = False
view = True
results = False
export = False

### COARSE MESH ###

vertices, faces = threefold_vault(r, pos_angles, wid_angles, offset1=offset1, offset2=offset2)
mesh = CoarseQuadMesh.from_vertices_and_faces(vertices, faces)

### DENSE MESH ###

mesh = quadmesh_densification(mesh, target_edge_length)

### ELEMENT TYPES ###

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

vkey2step = {vkey: step for pkey, step in pkey2step.items() for vkey in mesh.polyedge_vertices(pkey) if step is not None and step >= 0}

# group objects per step
step2edges = {step: [] for step in range(min_step, max_step + 1)}
for u, v in edges_ortho:
    if u not in supports:
        if vkey2step[u] > vkey2step[v]:
            u, v = v, u
        step = max([vkey2step[u], vkey2step[v]])
        step2edges[step].append((u, v))

#### PLANAR SMOOTH ###
mesh.smooth_centroid(fixed=[vkey for vkey in mesh.vertices_on_boundary() if mesh.vertex_degree(vkey) == 2], kmax=5)
mesh.smooth_centroid(fixed=mesh.vertices_on_boundary(), kmax=100)

### DATA CONVERSION ###
network = mesh_to_fdnetwork(mesh, supports, pz, q0)

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

# profile nodes
pnodes = []
for pkey, polyedge in mesh.polyedges(True):
    if snode in polyedge:
        for node in polyedge:
            if mesh.is_vertex_on_boundary(node) and not network.is_node_support(node):
                pnodes.append(node)

# profile curves (from singularity to unsupported boundary)
profile_polyedges = []
for pkey, polyedge in mesh.polyedges(data=True):
    cdt1 = mesh.vertex_degree(polyedge[0]) == 6 and polyedge[-1] not in supports
    cdt2 = mesh.vertex_degree(polyedge[-1]) == 6 and polyedge[0] not in supports
    if cdt1 or cdt2:
        polyedge = list(reversed(polyedge)) if mesh.vertex_degree(polyedge[-1]) == 6 else polyedge
        profile_polyedges.append(polyedge)

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


# edges on polyedges parallel to spine
# pedges = {pkey: pedge for pkey, pedge in mesh.polyedges(True)}
# courses_edges = []
# for pkey, step in pkey2step.items():
#     if step and step != -1:
#         edges = list(pairwise(pedges[pkey]))
#         courses_edges.extend(edges)

### FDM  ###

eqnetwork = fdm(network)

if optimize:

    ### PARAMETERS ###
    parameters = []

    for edge in network.edges():
        parameter = EdgeForceDensityParameter(edge, qmin, qmax)
        parameters.append(parameter)

    parameters_supports = []
    if add_supports_as_parameters:
        for node in network.nodes_anchors():
            x, y, z = network.node_coordinates(node)
            for coordinate, parameter in zip((x, y), (NodeAnchorXParameter, NodeAnchorYParameter)):
                parameter = parameter(node, coordinate - ctol, coordinate + ctol)
                parameters_supports.append(parameter)

    ### GOALS ###

    # target position of spine central node
    goals_target = []
    if add_node_target_goal:
        for node in network.nodes():
            if network.degree(node) == 6:
                x, y, z = network.node_coordinates(node)
                goal = NodePointGoal(node, target=[x, y, cross_height], weight=weight_node_target_goal)
                goals_target.append(goal)
        print('{} NodePointGoal'.format(len(goals_target)))

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
                goals_spine_planarity.append(goal)
        print('{} ProfilePlanarityGoal'.format(len(goals_spine_planarity)))

    # length of profile curves edges
    course_width = 0.125
    goals_length_profile = []
    if add_edge_length_profile_goal:
        for polyedge in profile_polyedges:

            for i, edge in enumerate(pairwise(polyedge)):
                factor = 1.0
                if i == 0:
                    factor = 2.0
                goal = EdgeLengthGoal(edge, target=course_width * factor, weight=weight_edge_length_profile_goal)

                goals_length_profile.append(goal)

        print('{} EdgeProfileLengthGoal'.format(len(goals_length_profile)))

    goals_length_equal_spine = []
    if add_edge_length_equal_spine_goal:
        for polyedge in spine_polyedges:
            edges = list(pairwise(polyedge))
            goal = EdgesLengthEqualGoal(edges, weight=weight_edge_length_equal_spine_goal)
            goals_length_equal_spine.append(goal)
        print('{} EdgeSpineEqualLengthGoal'.format(len(goals_length_equal_spine)))

    goals_length_strips = []
    if add_edge_length_strips_goal:
        for i, strips_split in profile_strips_split.items():
            for strip in strips_split:
                for side in strip:
                    indices = list(range(len(side)))
                    # if i == 0:
                    #     continue
                        # indices = [1, -1]
                    # for j in (1, 2, 3, 4, 5):
                    for j in indices:
                        edge = side[-j]
                        goal = EdgeLengthGoal(edge, target=course_width, weight=weight_edge_length_strips_goal)
                        goals_length_strips.append(goal)
        print('{} EdgeStripsLengthGoal'.format(len(goals_length_strips)))

    goals_length_equal_strips = []
    if add_edge_length_equal_strips_goal:
        for step, strips in profile_strips.items():
            # if step != 3:
            #     continue
            for strip in strips:
                edges = strip[1:-1]
                goal = EdgesLengthEqualGoal(edges, weight=weight_edge_length_equal_strips_goal)
                goals_length_equal_strips.append(goal)
        print('{} EdgeStripsEqualLengthGoal'.format(len(goals_length_equal_strips)))


        # for strip in profile_strips[3]:
        #     edges = strip
        #     goal = EdgesLengthEqualGoal(edges, weight=weight_edge_length_equal_strips_goal)
        #     goals_length_equal_strips.append(goal)

        # for strip in profile_strips[0]:
        #     for u, v in (strip[2], strip[-2]):
        #         edge = (u, v) if network.has_edge(u, v) else (v, u)
        #         goal = EdgeLengthGoal(edge, target=course_width, weight=weight_edge_length_goal)
        #         goals_length.append(goal)

        print('{} EdgeStripsLengthGoal'.format(len(goals_length_equal_strips)))


    loss = Loss(
                SquaredError(goals=goals_target, name='NodePointGoal', alpha=1.0),
                SquaredError(goals=goals_spine_planarity, name='EdgeSpinePlanarityGoal', alpha=1.0),
                SquaredError(goals=goals_profile_planarity, name='EdgeProfilePlanarityGoal', alpha=1.0),
                SquaredError(goals=goals_length_profile, name='EdgeProfileLengthGoal', alpha=1.0),
                )

    ### CONSTRAINED FORM FINDING I ###
    network = constrained_fdm(network,
                              optimizer=opt(),
                              parameters=parameters + parameters_supports,
                              loss=loss,
                              maxiter=maxiter,
                              tol=tol
                              )

    # raise
    cnetwork1 = network.copy()

    ### CONSTRAINED FORM FINDING II ###

    if add_node_target_goal:
        for polyedge in spine_polyedges:
            for node in polyedge:
                if network.degree(node) == 6:
                    continue
                x, y, z = network.node_coordinates(node)
                goal = NodePointGoal(node, target=[x, y, z], weight=weight_node_target_goal)
                goals_target.append(goal)
        print('{} NodePointGoal'.format(len(goals_target)))

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

    loss = Loss(
                SquaredError(goals=goals_target, name='NodePointGoal', alpha=1.0),
                SquaredError(goals=goals_spine_planarity, name='EdgeSpinePlanarityGoal', alpha=1.0),
                SquaredError(goals=goals_profile_planarity, name='EdgeProfilePlanarityGoal', alpha=1.0),
                SquaredError(goals=goals_length_profile, name='EdgeProfileLengthGoal', alpha=1.0),
                SquaredError(goals=goals_length_strips, name='EdgeLengthStripsGoal', alpha=1.0),
                PredictionError(goals=goals_length_equal_strips, name='EdgesLengthEqualStripsGoal', alpha=1.0),
                SquaredError(goals=goals_plane, name='NodePlaneGoal', alpha=1.0),

                )

    network = constrained_fdm(network,
                              optimizer=opt(),
                              parameters=parameters + parameters_supports,
                              loss=loss,
                              maxiter=maxiter,
                              tol=tol
                              )

### VIEWER ###

if view:
    viewer = Viewer(width=1600, height=900, show_grid=False)

    # viewer.add(mesh)
    #
    # viewer.add(cnetwork1, as_wireframe=True, show_points=False)
    #
    #
    # for polyedge in spine_polyedges:
    #     for i, edge in enumerate(pairwise(polyedge)):
    #         line = Line(*mesh.edge_coordinates(*edge))
    #         viewer.add(line, linewidth=4.)

    maxs = max_step
    cmap = ColorMap.from_mpl("viridis")
    for i in range(maxs):

    # for polyedge in profile_polyedges:
    #     for i, edge in enumerate(pairwise(polyedge)):
    #         line = Line(*mesh.edge_coordinates(*edge))
    #         linecolor = cmap(i, maxval=maxs)
    #         viewer.add(line, linewidth=4., linecolor=linecolor)
        for strip in profile_strips[i]:
            for edge in strip:
                line = Line(*network.edge_coordinates(*edge))
                viewer.add(line, linecolor=cmap(i, maxval=maxs), linewidth=5.)

            for polyedge in zip(*strip):
                polyline = Polyline([network.node_coordinates(n) for n in polyedge])
                viewer.add(polyline)

    # viewer.add(eqnetwork, as_wireframe=True)

    # if optimize:
    #     viewer.add(network,
    #                edgewidth=(0.003, 0.02),
    #                edgecolor="force",
    #                show_loads=False,
    #                show_reactions=False,
    #                reactionscale=0.25,
    #                loadscale=0.5)

    viewer.show()


### RESULTS ###

if results:
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

if export:
    pass
