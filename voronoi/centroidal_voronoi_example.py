import numpy as np
import polyscope as ps
import trimesh
from context_voronoi import ContextVoronoi
from vectoradammodified import VectorAdamModified


context = ContextVoronoi(radius=1.0, lr=1.0, betas=(0.9, 0.999), eps=1e-8, steps=20, num_observations=400)

# generate the centroidal points
context.generate_centroidal_points()

# initialize the optimizer
context.initialize()

context.register_points()
context.register_anchor_points()

for i in range(context.steps):
    # generate the voronoi diagram
    context.generate_voronoi_diagram()
    loss = context.compute_loss()


ps.show()