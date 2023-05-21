import os
from math import fabs
from math import radians

from compas_quad.datastructures import QuadMesh
from compas.datastructures import network_find_cycles
from compas.datastructures import mesh_weld
from compas.geometry import normalize_vector
from compas.geometry import dot_vectors
from compas.datastructures import Mesh
from compas.datastructures import Network
from compas.datastructures import mesh_transformed
from compas.datastructures import network_transformed
from compas.geometry import Line
from compas.geometry import Rotation
from compas.geometry import Translation
from compas.geometry import Transformation

from compas.colors import Color
from compas.colors import ColorMap

from compas.utilities import geometric_key_xy
from compas.utilities import pairwise

from jax_fdm.datastructures import FDNetwork
from jax_fdm.visualization import Plotter
from jax_fdm.visualization import Viewer

from fourf import DATA
from fourf.support import polyedge_types
from fourf.sequence import quadmesh_polyedge_assembly_sequence

# ==========================================================================
# Import datastructures
# ==========================================================================

# angle bounds
l_start, l_end = 45, 15  # long spans, angle bounds and variation exponent [-]
s_start, s_end = 30, 15  # short spans, angle bounds and variation exponent [-]
max_step_sequential_short = 3
min_step = -1
max_step = 5

FILE_IN = os.path.abspath(os.path.join(DATA, "tripod_network_dual_spine_3d.json"))
network_spine = Network.from_json(FILE_IN)

FILE_IN = os.path.abspath(os.path.join(DATA, "tripod_network_dual_3d_full_step_3.json"))
network = FDNetwork.from_json(FILE_IN)

FILE_IN = os.path.abspath(os.path.join(DATA, "tripod_network_dual_3d.json"))
network_final = FDNetwork.from_json(FILE_IN)

# ==========================================================================
# Create mesh for visualization
# ==========================================================================

cycles = network_find_cycles(network)[1:]
vertices = {vkey: network.node_coordinates(vkey) for vkey in network.nodes()}
mesh = QuadMesh.from_vertices_and_faces(vertices, cycles)

# ==========================================================================
# Create 2D projection network for shadow effect
# ==========================================================================

_network_2d = network.copy()
_network_2d.nodes_attribute("z", 0.0)
cycles = network_find_cycles(network)[:1]
vertices = {vkey: _network_2d.node_coordinates(vkey) for vkey in _network_2d.nodes()}
mesh_2d = Mesh.from_vertices_and_faces(vertices, cycles)

network_spine_2d = network_spine.copy()
network_spine_2d.nodes_attribute("z", 0.0)
cycles = network_find_cycles(network_spine_2d)[:1]
vertices = {vkey: network_spine_2d.node_coordinates(vkey) for vkey in network_spine_2d.nodes()}
mesh_spine_2d = Mesh.from_vertices_and_faces(vertices, cycles)

# ==========================================================================
# Topological processing
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
# Profile
# ==========================================================================

# profile polyedges
profile_polyedges = []
for fkey in mesh.faces():
    if len(mesh.face_vertices(fkey)) == 6:
        for i, (u0, v0) in enumerate(mesh.face_halfedges(fkey)):
            if i != 0 and i % 2 != 0:
                continue
            polyedge = mesh.collect_polyedge(u0, v0, both_sides=True)
            if polyedge[-1] not in supported_vkeys:
                profile_polyedges.append(polyedge)

print("Num profile polyedges: ", len(profile_polyedges))

# profile edges
profile_edges = set([edge for polyedge in profile_polyedges for edge in pairwise(polyedge)])
print("Profile edges: ", profile_edges)

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
# min_step, max_step = int(min(steps)), int(max(steps))
# print(min_step, max_step)

edge2step = {}
for u, v in mesh.edges():
    edge2step[tuple(sorted((u, v)))] = max([vkey2step[u], vkey2step[v]])

step2edges = {step: [] for step in range(min_step, max_step + 1)}
for u, v in mesh.edges():
    step = max([vkey2step[u], vkey2step[v]])
    step2edges[step].append((u, v))

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
# Visualization
# ==========================================================================

# edge colors
cmap = ColorMap.from_mpl("magma")
edgecolor = {edge: Color.grey() for edge in mesh.edges()}

edgelabels = {}

for edge in profile_edges:
    u, v = edge
    edge_step = edge2step.get(edge, edge2step.get((v, u)))

    if edge_step in (0, -1):
        continue

    angle = l_start + (l_end - l_start) * ((edge_step - 1) / (max_step - 1))

    if edge in profile_edges_span_short or (v, u) in profile_edges_span_short:
        angle = s_start + (s_end - s_start) * ((edge_step - 1) / (max_step - 3))

    ratio = 1 - ((angle - l_end) / (l_start - l_end))

    edgelabels[(u, v)] = angle
    edgelabels[(v, u)] = angle

# edge widths
edgewidth = {}

target_step = 3
for u, v in mesh.edges():
    width = 0.001
    edge_step = edge2step.get((u, v), edge2step.get((v, u)))
    if edge_step == target_step:
        width = 1.
    edgewidth[(u, v)] = width

# face colors
# color = (181, 54, 122)
color = Color.pink().lightened(50)
facecolor = {fkey: Color.grey().lightened(90) for fkey in mesh.faces()}
for fkey in mesh.faces():
    if not mesh.is_face_on_boundary(fkey):
        continue
    if any(vkey in supports for vkey in mesh.face_vertices(fkey)):
        continue
    facecolor[fkey] = color

# plotter
plotter = Plotter(figsize=(16, 9), dpi=200)
viewer = Viewer(width=1600, height=900, show_grid=False)
projection = viewer.view.camera.viewworld()
T = Transformation(projection.tolist())

# plotter.add(mesh_transformed(mesh_2d, T),
#             show_vertices=False,
#             show_edges=False,
#             facecolor={fkey: Color.grey().lightened(70) for fkey in mesh_2d.faces()})

plotter.add(mesh_transformed(mesh_spine_2d, T),
            show_vertices=False,
            edgewidth=0.2,
            show_faces=False,
            edgecolor={edge: Color.grey() for edge in network_spine_2d.edges()}
            )

# artist = plotter.add(mesh_transformed(mesh, T),
#                      show_vertices=True,
#                      vertexsize=5,
#                      vertexcolor={vkey: Color.white() for vkey in mesh.vertices()},
#                      show_edges=False,
#                      edgewidth=edgewidth,
#                      edgecolor=edgecolor,
#                      show_faces=False,
#                      facecolor=facecolor,
#                      )

gkey_key = {geometric_key_xy(mesh.vertex_coordinates(vkey)): vkey for vkey in mesh.vertices()}
for vkey in mesh_2d.vertices_on_boundary():
    pb = mesh_2d.vertex_coordinates(vkey)
    pa = mesh.vertex_coordinates(gkey_key[geometric_key_xy(pb)])
    line = Line(pa, pb)
    plotter.add(line.transformed(T),
                linewidth=0.1,
                draw_as_segment=True,
                zorder=10000,
                )

# artist.draw_edgelabels(edgelabels)
plotter.zoom_extents(padding=-0.2)
plotter.save("innixar_goals_lines_3d.pdf")
plotter.show()
