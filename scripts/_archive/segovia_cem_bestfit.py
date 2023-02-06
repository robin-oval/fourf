import os

from statistics import mean

from compas.datastructures import mesh_dual

from compas.geometry import add_vectors, scale_vector
from compas.geometry import Translation

from compas.utilities import pairwise

from compas_quad.datastructures import CoarseQuadMesh
from compas_quad.datastructures import QuadMesh

from compas_cem.diagrams import TopologyDiagram

from compas_cem.loads import NodeLoad

from compas_cem.equilibrium import static_equilibrium

from compas_cem.optimization import Optimizer

from compas_cem.optimization import TrailEdgeParameter
from compas_cem.optimization import DeviationEdgeParameter
from compas_cem.optimization import OriginNodeXParameter
from compas_cem.optimization import OriginNodeYParameter
from compas_cem.optimization import OriginNodeZParameter
from compas_cem.optimization import NodeLoadXParameter
from compas_cem.optimization import NodeLoadYParameter
from compas_cem.optimization import NodeLoadZParameter
from compas_cem.optimization import PlaneConstraint
from compas_cem.optimization import LineConstraint
from compas_cem.optimization import DeviationEdgeLengthConstraint
from compas_cem.optimization import PointConstraint

from compas_cem.viewers import Viewer
from compas_cem.plotters import Plotter

from fourf import DATA
from fourf.utilities import mesh_to_fdnetwork
from fourf.topology import threefold_vault
from fourf.topology import quadmesh_densification
from fourf.support import support_shortest_boundary_polyedges
from fourf.support import polyedge_types
from fourf.sequence import quadmesh_polyedge_assembly_sequence

# ------------------------------------------------------------------------------
# Controls
# ------------------------------------------------------------------------------

PLANAR = False
OPTIMIZE = True
OPTIMIZER = "LBFGS"
ITERS = 5000
EPS = 1e-6

EXPORT_JSON = False

PLOT = False
VIEW = True
SHOW_EDGETEXT = False

STRIPS_DENSITY = 4  # only even numbers (2, 4, 6, ...) for best results
SHIFT_TRAILS = False
TRAIL_LENGTH = 0.5   # starting abs length in all deviation edges
DEVIATION_FORCE = 0.6  # starting abs force in all deviation edges

brick_length, brick_width, brick_thickness = 0.24, 0.125, 0.04  # [m]
brick_layers = 4  # [-]
brick_density = 12.0  # [kN/m3]

dead_load = 1.0  # additional dead load [kN/m2]
pz = brick_density * brick_thickness * brick_layers + dead_load  # vertical area load (approximated self-weight + uniform dead load) [kN/m2]

# point constraint weights
w_start = 10.0
w_end = 5.0

ADD_DEVIATION_LENGTH_CONSTRAINT = True
w_length = 5.0

PARAMETRIZE_ORIGIN_NODES = True
ztol = 0.1

# ------------------------------------------------------------------------------
# Create a topology diagram
# ------------------------------------------------------------------------------

HERE = os.path.dirname(__file__)
FILE_IN = os.path.abspath(os.path.join(DATA, "bestfit_mesh_dual.json"))

mesh = QuadMesh.from_json(FILE_IN)
mesh.collect_polyedges()
mesh.collect_strips()
print('quad mesh:', mesh)

supports = []
boundary_vertices = mesh.vertices_on_boundary()[:-1]
for i in range(len(boundary_vertices)):
    u, v = boundary_vertices[0:2]
    polyedge = mesh.collect_polyedge(u, v)
    if i % 2 == 0:
        supports += polyedge
        for _ in range(len(polyedge)):
            del boundary_vertices[0]
    else:
        for _ in range(len(polyedge) - 2):
            del boundary_vertices[0]
    if len(boundary_vertices) == 0:
        break
print(len(supports), 'supports')

mean_length = mean([mesh.edge_length(*edge) for edge in mesh.edges()])

topology = TopologyDiagram.from_dualquadmesh(mesh,
                                             supports,
                                             trail_length=TRAIL_LENGTH,
                                             trail_state=-1,
                                             deviation_force=DEVIATION_FORCE,
                                             deviation_state=-1)
topology.build_trails()

if not PLANAR:
    for key in topology.nodes():
        area = mesh.vertex_area(key)
        node_load = area * pz
        topology.add_load(NodeLoad(key, [0.0, 0.0, -node_load]))

# ------------------------------------------------------------------------------
# Shift trail sequences to avoid having indirect deviation edges
# ------------------------------------------------------------------------------

if SHIFT_TRAILS:
    print("Shifting sequences, baby!")

    while topology.number_of_indirect_deviation_edges() > 0:
        for node_origin in topology.origin_nodes():

            for edge in topology.connected_edges(node_origin):

                if topology.is_indirect_deviation_edge(edge):
                    u, v = edge
                    node_other = u if node_origin != u else v
                    sequence = topology.node_sequence(node_origin)
                    sequence_other = topology.node_sequence(node_other)

                    if sequence_other > sequence:
                        topology.shift_trail(node_origin, sequence_other)


for trail in topology.trails():
    for u, v in pairwise(trail):
        vector = topology.edge_vector(u, v)
        origin = topology.node_coordinates(v)
        plane = (origin, vector)
        if not topology.has_edge(u, v):
            u, v = v, u
        topology.edge_attribute(key=(u, v), name="plane", value=plane)

# ------------------------------------------------------------------------------
# Compute a state of static equilibrium
# ------------------------------------------------------------------------------

form = static_equilibrium(topology, eta=1e-6, tmax=100)

# ------------------------------------------------------------------------------
# Optimization
# ------------------------------------------------------------------------------

if OPTIMIZE:
    # create optimizer
    opt = Optimizer()

    # parameters

    for edge in topology.deviation_edges():
        opt.add_parameter(DeviationEdgeParameter(edge, bound_low=None, bound_up=DEVIATION_FORCE-0.09))

    # parameters
    if PARAMETRIZE_ORIGIN_NODES:
        for node in topology.origin_nodes():
            x, y, z = topology.node_coordinates(node)
            opt.add_parameter(OriginNodeZParameter(node, ztol))

    # constraints
    for trail in topology.trails():
        n = len(trail) - 1
        for i, node in enumerate(trail[1:]):
            weight = w_start + (w_end - w_start) * (i / (n - 1))
            point = topology.node_coordinates(node)
            opt.add_constraint(PointConstraint(node, point=point, weight=weight))

    if ADD_DEVIATION_LENGTH_CONSTRAINT:
        for edge in topology.deviation_edges():
            length = topology.edge_length(*edge)
            opt.add_constraint(DeviationEdgeLengthConstraint(edge, length=length, weight=w_length))

    tmax = 1
    if topology.number_of_indirect_deviation_edges() > 0:
        tmax = 100

    # optimize
    form_opt = opt.solve(topology.copy(),
                         algorithm=OPTIMIZER,
                         iters=ITERS,
                         tmax=tmax,
                         eps=EPS,
                         verbose=True)

# ------------------------------------------------------------------------------
# Export to JSON
# ------------------------------------------------------------------------------

if EXPORT_JSON:
    datastructures = [mesh, form]
    names = ["shell_topology", "shell_form_found"]

    if OPTIMIZE:
        datastructures.append(form_opt)
        names.append("shell_form_opt")

    for ds, name in zip(datastructures, names):
        path = os.path.join(HERE, f"data/{name}.json")
        ds.to_json(path)
        print(f"Exported datastructure to {path}")

# ------------------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------------------

if PLOT or VIEW:
    shift_vector = [0.0, 0.0, 0.0]
    topology = topology.transformed(Translation.from_vector(scale_vector(shift_vector, 1.)))
    form = form.transformed(Translation.from_vector(scale_vector(shift_vector, 2.)))
    forms = [form]

    if OPTIMIZE:
        form_opt = form_opt.transformed(Translation.from_vector(scale_vector(shift_vector, 3.)))
        forms.append(form_opt)

# ------------------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------------------

if PLOT:

    ns = 12
    form_edgewidth = (0.75, 3.0)

    plotter = Plotter(figsize=(18, 9))

    plotter.add(topology,
                nodesize=ns,
                edgecolor="type",
                nodetext="key",
                nodecolor="type",
                show_nodetext=True)

    # for form in forms:
    #     plotter.add(form,
    #                 edgewidth=form_edgewidth,
    #                 nodesize=ns,
    #                 edgetext="force",
    #                 show_loads=False,
    #                 show_reactions=False,
    #                 show_edgetext=SHOW_EDGETEXT)

    plotter.zoom_extents(padding=0.0)
    plotter.show()

# ------------------------------------------------------------------------------
# Launch viewer
# ------------------------------------------------------------------------------

if VIEW:

    viewer = Viewer(width=1600, height=900, show_grid=False)

# ------------------------------------------------------------------------------
# Visualize starting mesh
# ------------------------------------------------------------------------------

    viewer.add(mesh)

# ------------------------------------------------------------------------------
# Visualize topology diagram
# ------------------------------------------------------------------------------

    viewer.add(topology,
               show_nodes=True,
               nodes=None,
               nodesize=15.0,
               show_edges=True,
               edges=None,
               edgewidth=0.01,
               show_loads=False,
               loadscale=1.0,
               show_edgetext=False,
               edgetext="key",
               show_nodetext=False,
               nodetext="key"
               )

# ------------------------------------------------------------------------------
# Visualize translated form diagram
# ------------------------------------------------------------------------------

    # viewer.add(form,
    #            show_nodes=False,
    #            nodes=None,
    #            nodesize=15.0,
    #            show_edges=True,
    #            edges=None,
    #            edgewidth=0.01,
    #            show_loads=True,
    #            loadscale=1.0,
    #            loadtol=1e-1,
    #            show_residuals=False,
    #            residualscale=1.0,
    #            residualtol=1.5,
    #            show_edgetext=SHOW_EDGETEXT,
    #            edgetext="force",
    #            show_nodetext=False,
    #            nodetext="xyz"
    #            )

# ------------------------------------------------------------------------------
# Visualize translated constrained form diagram
# ------------------------------------------------------------------------------

    if OPTIMIZE:
        # form_opt = form_opt.transformed(Translation.from_vector([9.0, -3.0, 0.0]))
        viewer.add(form_opt,
                   show_nodes=False,
                   nodes=None,
                   nodesize=15.0,
                   show_edges=True,
                   edges=None,
                   edgewidth=(0.01, 0.04),
                   show_loads=True,
                   loadscale=1.0,
                   loadtol=1e-1,
                   show_residuals=True,
                   residualscale=1.0,
                   residualtol=1.5,
                   show_edgetext=SHOW_EDGETEXT,
                   edgetext="force",
                   show_nodetext=False,
                   nodetext="xyz"
                   )

# ------------------------------------------------------------------------------
# Show scene
# -------------------------------------------------------------------------------

    viewer.show()
