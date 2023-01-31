from math import pi

from numpy import mean, std, radians

from operator import itemgetter

from compas.geometry import distance_point_point, length_vector, length_vector_xy, sum_vectors, cross_vectors, rotate_points, bounding_box

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
from jax_fdm.parameters import EdgeForceDensityParameter
from jax_fdm.goals import NodePointGoal, EdgeLengthGoal, EdgeDirectionGoal, NodeTangentAngleGoal, NodeXCoordinateGoal, NodeYCoordinateGoal, NodeResidualForceGoal
from jax_fdm.constraints import EdgeLengthConstraint, EdgeForceConstraint
from jax_fdm.losses import SquaredError, Loss
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

r = 2.5 # circumcircle radius [m]
pos_angles = radians(90), radians(235), radians(305) # triangle parameterisation angle [radians]
wid_angles = radians(7.5), radians(7.5), radians(7.5) # triangle to hexagon parameterisation angle [radians]
offset1, offset2 = 0.85, 0.95 # offset factor the unsupported and supported edges inward respectively [-]
target_edge_length = 0.15 # [m]
support_raised_height = [0.0, 0.0, 0.0] # raised height of each support [m]

brick_length, brick_width, brick_thickness = 0.24, 0.12, 0.03 # [m]
brick_layers = 3 # [-]
brick_density = 12.0 # [kN/m3]
comp_strength = 6.0 # [MPa]

dead_load = -1.0 # dead load [kN/m2]
pz = - (brick_density * brick_thickness * brick_layers) + dead_load # vertical load (approximated self-weight + uniform dead load) [kN/m2]
q0 = -1.0 # initial force densities [kN/m]

opt = LBFGSB # optimization solver
qmin, qmax = None, -1e-1 # bound on force densities [kN/m]
maxiter = 1000 # maximum number of iterations
tol = 1e-3 # optimization tolerance

# aim for target positions
add_node_target_goal = True
cross_height = 2.5 # height of spine cross [m]
weight_node_target_goal = 1.0

# edge length goal to obtain constant brick course widths
add_edge_length_goal = False
factor_edge_length = 1.5 # multplicative factor of average length of planar transverse edge [-]
weight_edge_length_goal = 1.0

# vertex projection goal to cover the desire space
add_horizontal_projection_goal = True
weight_projection_goal = 1.0

# support node residual force to control the formation of creases and corrugations
add_node_residual_goal = False
rmin, rmax, rexp = 0.4, 3.0, 2.0 # minimum and maximum reaction forces [kN] and variation exponent [-]
weight_node_residual_goal = 1.0

# shape the profile of the polyedges running from the singularity to the unspupported boundary
# via a node tangent angle goal
add_node_tangent_goal = False
t_start, t_end, t_exp = radians(20), radians(40), 1.0 # minimum and maximum reaction forces [kN] and variation exponent [-]
weight_node_tangent_goal = 1.0
# via an edge slope angle goal
add_edge_slope_goal = True
s_start, s_end, s_exp = radians(70), radians(70), 1.0 # minimum and maximum reaction forces [kN] and variation exponent [-]
weight_edge_slope_goal = 1.0

view = True
export = False

### COARSE MESH ###

vertices, faces = threefold_vault(r, pos_angles, wid_angles, offset1=offset1, offset2=offset2)
mesh = CoarseQuadMesh.from_vertices_and_faces(vertices, faces)

### DENSE MESH ###

mesh = quadmesh_densification(mesh, target_edge_length)
print(mesh)

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

step2edges = {step: [] for step in range(min_step, max_step + 1)}
for u, v in edges_ortho:
    if u not in supports:
        if vkey2step[u] > vkey2step[v]:
            u, v = v, u
        step = max([vkey2step[u], vkey2step[v]])
        step2edges[step].append((u, v))

### PLANAR SMOOTH ###
mesh.smooth_centroid(fixed=[vkey for vkey in mesh.vertices_on_boundary() if mesh.vertex_degree(vkey) == 2], kmax=5)
mesh.smooth_centroid(fixed=mesh.vertices_on_boundary(), kmax=100)

### DATA CONVERSION ###

network = mesh_to_fdnetwork(mesh, supports, pz, q0)

### SUPPORT POSITIONING ###

i = 0
for pkey in mesh.polyedges():
    if pkey2type[pkey] == 'support':
        for vkey in mesh.polyedge_vertices(pkey):
            network.node[vkey]['z'] = support_raised_height[i]
        i += 1

### PARAMETERS ###

parameters = []

for edge in network.edges():
    parameter = EdgeForceDensityParameter(edge, qmin, qmax)
    parameters.append(parameter)

### GOALS ###

# target height of spine cross
goals_target = []
if add_node_target_goal:
    for node in network.nodes():
        if network.degree(node) == 6:
            x, y, z = network.node_coordinates(node)
            goal = NodePointGoal(node, target=[x, y, cross_height], weight=weight_node_target_goal)
            goals_target.append(goal)
    print('{} NodePointGoal'.format(len(goals_target)))

# constant brick course width
goals_length = []
if add_edge_length_goal:
    target_length = factor_edge_length * mean([mesh.edge_length(*edge) for edge in edges_ortho])
    for edge in network.edges():
        if edge2color[edge] == color_ortho: 
            goal = EdgeLengthGoal(edge, target=target_length, weight=weight_edge_length_goal)
            goals_length.append(goal)
    print('{} EdgeLengthGoal'.format(len(goals_length)))

# cover desired space
goals_projection = []
if add_horizontal_projection_goal:
    for node in network.nodes():
        if node not in supports:
            x, y, z = network.node_coordinates(node)
            for goal, xy in zip((NodeXCoordinateGoal, NodeYCoordinateGoal), (x, y)):
                goal = goal(node, xy)
                goals_projection.append(goal)
    print('{} NodeXCoordinateGoal and NodeYCoordinateGoal'.format(len(goals_projection)))

# form crease/corrugation through reaction forces
goals_residual = []
if add_node_residual_goal:
    for node in network.nodes():
        if node in supports:
            adim_step = (vkey2step[node] - min_step) / (max_step - min_step)
            goal_target_reaction = rmin + (rmax - rmin) * (1.0 - adim_step) ** rexp
            goal = NodeResidualForceGoal(node, target=goal_target_reaction, weight=weight_node_residual_goal)
            goals_residual.append(goal)
    print('{} NodeResidualForceGoal'.format(len(goals_residual)))

# shape profile curves from singularity to unsupported boundary
profile_polyedges = []
for pkey, polyedge in mesh.polyedges(data=True):
        cdt1 = mesh.vertex_degree(polyedge[0]) == 6 and polyedge[-1] not in supports
        cdt2 = mesh.vertex_degree(polyedge[-1]) == 6 and polyedge[0] not in supports
        if cdt1 or cdt2:
            polyedge = list(reversed(polyedge)) if mesh.vertex_degree(polyedge[-1]) == 6 else polyedge
            profile_polyedges.append(polyedge)

goals_normal = []
if add_node_tangent_goal:
    for polyedge in profile_polyedges:
        n = len(polyedge)
        for i, vkey in enumerate(polyedge):
            angle = t_start + (t_end - t_start) * (i / (n -1)) ** t_exp
            goal = NodeTangentAngleGoal(vkey, vector=[0.0, 0.0, 1.0], target=angle, weight=weight_node_tangent_goal)
            goals_normal.append(goal)
    print('{} NodeTangentAngleGoal'.format(len(goals_normal)))

goals_slope = []
if add_edge_slope_goal:
    for polyedge in profile_polyedges:
        n = len(polyedge)
        for i, edge in enumerate(pairwise(polyedge)):
            angle = s_start + (s_end - s_start) * (i / (n -1)) ** s_exp
            vector0 = mesh.edge_vector(*edge)
            ortho = cross_vectors(vector0, [0.0, 0.0, 1.0])
            vector = rotate_points([vector0], pi / 2 - angle, axis=ortho, origin=[0.0, 0.0, 0.0])[0]
            goal = EdgeDirectionGoal(edge, target=vector, weight=weight_edge_slope_goal)
            goals_slope.append(goal)
    print('{} EdgeDirectionGoal'.format(len(goals_slope)))
            
loss = Loss(
            SquaredError(goals=goals_target, name='NodePointGoal', alpha=1.0),
            SquaredError(goals=goals_length, name='EdgeLengthGoal', alpha=1.0),
            SquaredError(goals=goals_projection, name='NodeCoordinateGoal', alpha=1.0),
            SquaredError(goals=goals_residual, name='NodeResidualForceGoal', alpha=1.0),
            SquaredError(goals=goals_normal, name='NodeTangentAngleGoal', alpha=1.0),
            SquaredError(goals=goals_slope, name='EdgeAngleGoal', alpha=1.0),
            
            )

### FORM FINDING ###

network = constrained_fdm(network,
                             optimizer=opt(),
                             parameters=parameters,
                             loss=loss,
                             maxiter=maxiter,
                             tol=tol,
                             )

### RESULTS ###

# new meshes to compute areas
mesh_ff_xyz = mesh.copy() # mesh that follows the form found network
mesh_ff_xy = mesh.copy() # mesh that follows the projection of the form found network
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

### VIEWER ###

if view:
    viewer = Viewer(width=1600, height=900, show_grid=False)

    viewer.add(mesh)

    viewer.add(network,
               edgewidth=(0.003, 0.02),
               edgecolor="force",
               reactionscale=0.25,
               loadscale=0.5)

    viewer.show()

if export:
    pass
