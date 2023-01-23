from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from math import cos, sin

from compas_quad.datastructures import CoarseQuadMesh

from compas.geometry import midpoint_point_point, circle_from_points, add_vectors, subtract_vectors, scale_vector


__all__ = []


def threefold_vault(radius, support_angular_positions, support_angular_width, offset1=1.0, offset2=1.0):

    vertices = []
    for a, da in zip(support_angular_positions, support_angular_width):
        for angle in [a - da / 2, a + da / 2]:
            vertices.append([radius * cos(angle), radius * sin(angle), 0.0])
    vertices.append([0.0, 0.0, 0.0])

    k = len(vertices) - 1
    corners = vertices[:-1]
    for i in range(k):
        vertices.insert(2 * i + 1, midpoint_point_point(corners[i], corners[(i + 1) % k]))

    g, _, _ = circle_from_points(vertices[3], vertices[7], vertices[11])
    vertices[-1] = g

    n = len(vertices)
    faces = []
    for i in range(k):
        faces.append([n - 1, (2 * i + 1) % (n - 1), (2 * i + 2) % (n - 1), (2 * i + 3) % (n - 1)])

    for i in [3, 7, 11]:
        gv = subtract_vectors(vertices[i], g)
        vertices[i] = add_vectors(g, scale_vector(gv, offset1))

    for i in [1, 5, 9]:
        gv = subtract_vectors(vertices[i], g)
        vertices[i] = add_vectors(g, scale_vector(gv, offset2))

    return vertices, faces


def quadmesh_densification(quadmesh, target_edge_length):

    if type(quadmesh) is not CoarseQuadMesh:
        quadmesh = CoarseQuadMesh.from_vertices_and_faces(*quadmesh.to_vertices_and_faces())
    quadmesh.collect_strips()
    quadmesh.set_strips_density_target(target_edge_length)
    quadmesh.densification()
    quadmesh = quadmesh.dense_mesh()
    quadmesh.collect_polyedges()

    return quadmesh
