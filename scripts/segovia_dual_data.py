from math import pi, radians

from compas_quad.datastructures import CoarseQuadMesh
from compas_quad.coloring import quad_mesh_polyedge_2_coloring

from fourf.topology import threefold_vault
from fourf.support import support_shortest_boundary_polyedges
from fourf.support import polyedge_types
from fourf.sequence import quadmesh_polyedge_assembly_sequence

from compas.utilities import pairwise


### INPUTS ###

r = 2.5 # circumcircle radius [m]

pos_angles = radians(90), radians(235), radians(305) # triangle parameterisation angle [radians]
wid_angles = radians(7.5), radians(7.5), radians(7.5) # triangle to hexagon parameterisation angle [radians]

offset1, offset2 = 0.85, 0.95 # offset factor the unsupported and supported edges inward respectively [-]

target_edge_length = 0.10 # [m]

view = True

### COARSE MESH ###

vertices, faces = threefold_vault(r, pos_angles, wid_angles, offset1=offset1, offset2=offset2)
mesh = CoarseQuadMesh.from_vertices_and_faces(vertices, faces)

### DENSE MESH ###

mesh.collect_strips()
mesh.set_strips_density_target(target_edge_length)
mesh.densification()
mesh = mesh.dense_mesh()
mesh = mesh.dual()
mesh.collect_polyedges()
print(mesh)

### SUPPORTS ###

supported_pkeys = set()
bdrypkey2length = {}
for pkey, polyedge in mesh.polyedges(data=True):
    if mesh.is_edge_on_boundary(polyedge[0], polyedge[1]):
        bdrypkey2length[pkey] = mesh.polyedge_length(polyedge)
avrg_length = sum(bdrypkey2length.values()) / len(bdrypkey2length)
supported_pkeys = set([pkey for pkey, length in bdrypkey2length.items() if length < avrg_length])
print('supported_pkeys', supported_pkeys)

supported_vkeys = set([vkey for pkey in supported_pkeys for vkey in mesh.polyedge_vertices(pkey)])

### SPINE ###

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

spine_strip_edges = []
for fkey in mesh.faces():
    if mesh.face_degree(fkey) == 6:
        for u, v in mesh.face_halfedges(fkey):
            strip_edges = mesh.collect_strip(v, u, both_sides=False)
            if strip_edges[-1][-1] in supported_vkeys:
                spine_strip_edges.append(strip_edges)

vault_profile_polyedges = []
for fkey in mesh.faces():
    if len(mesh.face_vertices(fkey)) == 6:
        for u0, v0 in mesh.face_halfedges(fkey):
            for u, v in ([u0, v0], [v0, u0]):
                polyedge = mesh.collect_polyedge(u, v, both_sides=False)
                if polyedge[-1] not in supported_vkeys:
                    vault_profile_polyedges.append(polyedge)
vault_profile_edges = set([edge for polyedge in vault_profile_polyedges for edge in pairwise(polyedge)])

### ASSEMBLY SEQUENCE ###


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

edges0 = [edge for pkey in mesh.polyedges() for edge in mesh.polyedge_edges(pkey) if pkey2color[pkey] == color0 and edge not in spine_strip_edges and edge not in vault_profile_edges]
edges1 = [edge for pkey in mesh.polyedges() for edge in mesh.polyedge_edges(pkey) if pkey2color[pkey] != color0 and edge not in spine_strip_edges and edge not in vault_profile_edges]


### VIEWER ###

if view:

    from compas.geometry import Line
    
    from compas_view2.app import App

    viewer = App(width=1600, height=900, show_grid=False)

    # viewer.add(mesh)

    # for pkey, ptype in pkey2type.items():
    #     if ptype == 'support':
    #         linecolor = (1.0, 0.0, 0.0)
    #     elif ptype == 'spine':
    #         linecolor = (1.0, 0.0, 1.0)
    #     elif ptype == 'span':
    #         linecolor = (0.0, 0.0, 1.0)
    #     else:
    #         linecolor = (0.0, 0.0, 0.0)
    #     for edge in pairwise(mesh.polyedge_vertices(pkey)):
    #         viewer.add(Line(*mesh.edge_coordinates(*edge)), linecolor=linecolor)

    for pkey in supported_pkeys:
        for edge in pairwise(mesh.polyedge_vertices(pkey)):
            viewer.add(Line(*mesh.edge_coordinates(*edge)), linecolor=(1.0, 0.0, 0.0))
    for strip in spine_strip_edges:
        for edge in strip:
            viewer.add(Line(*mesh.edge_coordinates(*edge)), linecolor=(1.0, 0.0, 1.0))
    for polyedge in vault_profile_polyedges:
        for edge in pairwise(polyedge):
            x = edge2step[tuple(sorted(edge))] / max_step
            viewer.add(Line(*mesh.edge_coordinates(*edge)), linecolor=(0.0, 0.2 + 0.8 * x, 0.2 + 0.8 * x))
    for edge in edges0:
        x = edge2step[tuple(sorted(edge))] / max_step
        viewer.add(Line(*mesh.edge_coordinates(*edge)), linecolor=(0.5 * x, 0.5 * x, 0.5 * x))
    for edge in edges1:
        x = edge2step[tuple(sorted(edge))] / max_step
        linecolor = (0.8 * x, 1.0, 0.8 * x)
        viewer.add(Line(*mesh.edge_coordinates(*edge)), linecolor=linecolor)

    viewer.show()
