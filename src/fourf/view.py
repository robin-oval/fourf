from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from compas_view2.app import App

from compas.geometry import Line

from compas.utilities import pairwise


__all__ = []


def view_f4(quadmesh, network, pkey2type, pkey2step):

    viewer = App(width=1600, height=900, show_grid=False)

    max_step = max([step for step in pkey2step.values() if step is not None])

    ptype2color = {
        'support': [1.0, 0.0, 0.0],
        'spine': [1.0, 0.0, 1.0],
        'span': [0.0, 0.0, 1.0],
        'cantilever': [0.0, 0.0, 0.0],
    }

    edge2color = {}
    for pkey, polyedge in quadmesh.polyedges(data=True):
        ptype = pkey2type[pkey]
        color = ptype2color[ptype]
        if ptype == 'span':
            step = pkey2step[pkey]
            lamba = step / max_step
            color = [1 - lamba, 0, lamba]
        for u, v in pairwise(polyedge):
            edge2color[(u, v)] = color

    for edge, color in edge2color.items():
        viewer.add(Line(*quadmesh.edge_coordinates(*edge)), linecolor=color)
    
    for edge in network.edges():
        viewer.add(Line(*network.edge_coordinates(*edge)))

    # viewer.add(quadmesh)
    # viewer.add(network)
    
    viewer.show()