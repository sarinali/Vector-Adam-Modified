import numpy as np
import torch
import polyscope as ps
import trimesh
from context_voronoi import ContextVoronoi
from vectoradammodified import VectorAdamModified
from util import compute_loss
from time import sleep

context = ContextVoronoi(radius=1.0, lr=0.1, betas=(0.9, 0.999), eps=1e-8, steps=20, num_observations=150, write_to_file=False, verbose=True)

context.initialize()

for i in range(100):
    # generate the voronoi diagram
    context.perform_step()

    # print(context.average_distance_from_origin())

    context.normalize_points()
    context.update_centroids_display()

    # print(context.average_distance_from_origin())

    ps.frame_tick()
    sleep(0.2)

# context.log_data_to_file()
# context.show_loss_visualization_single_line(override_path="logs/2025-02-10-22-19-16.json")
ps.show()