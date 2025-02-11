import numpy as np
import torch
import polyscope as ps
import trimesh
from context_voronoi import ContextVoronoi
from vectoradammodified import VectorAdamModified
from util import compute_loss
from time import sleep

context = ContextVoronoi(radius=1.0, lr=1.0, betas=(0.9, 0.999), eps=1e-8, steps=20, num_observations=400)

context.initialize()

for i in range(1000):
    # generate the voronoi diagram
    context.generate_voronoi_diagram()
    context.perform_step()

    # print(context.average_distance_from_origin())

    context.normalize_points()
    context.update_centroids_display()

    # print(context.average_distance_from_origin())

    ps.frame_tick()
    sleep(0.2)


    # TODO TODAY FIGURE OUT WHY THE PARAMETERS ARE NOT UPDATING


ps.show()