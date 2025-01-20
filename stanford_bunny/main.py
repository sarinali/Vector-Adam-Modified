import numpy as np
import polyscope as ps
import trimesh
import torch

from time import sleep
from util import *
from vectoradammodified import VectorAdamModified

ps.set_ground_plane_mode("shadow_only")
ps.init()

mesh = trimesh.load("assets/stanford-bunny.obj")
mesh.apply_scale(5)

# [[-0.359545  0.90386  -0.279685]] # anchor point 1 (on the ear)
# [[-0.303015  0.176095  0.221145]] # anchor point 2 (on the foot)
# [[0.300525 0.304485 0.0358  ]] # starting point 1 (on the butt)

lr = 1.5
betas = (0.9, 0.999)
eps = 1e-8

# Define steps constant
steps = 30

starting_point = torch.from_numpy(create_pointer(0.300525, 0.304485, 0.0358))

# Create the initial pointer n
v = starting_point.clone().numpy()
vector_list = np.array(v)
vector_list = vector_list.reshape(-1, 3)

position_list = np.array(v)
position_list = position_list.reshape(-1, 3)
v = torch.from_numpy(v).to(torch.float32)
v.requires_grad_()

# visualize starting point
v_np = starting_point.clone().numpy().reshape(1,3)
# v_np = starting_point.clone().numpy()

# create vector lists
v_np_list = starting_point.clone().numpy().reshape(1,3)

# initialize the two points
a = torch.tensor([-0.359545, 0.90386,  -0.279685], dtype=torch.float32)
b = torch.tensor([-0.303015,  0.176095,  0.221145], dtype=torch.float32)

# the tensors as polyscope points
a_np = a.clone().numpy().reshape(1, 3)
b_np = b.clone().numpy().reshape(1, 3)

a_cloud = ps.register_point_cloud("a", a_np, radius=0.02, color=(0,0,0))
b_cloud = ps.register_point_cloud("b", b_np, radius=0.02, color=(0,0,0))
v_cloud_list = ps.register_point_cloud("v", v_np, radius=0.02, color=(0.48,0.99,0))

vadam = VectorAdamModified([{'params': v, 'axis': -1}], lr=lr, betas=betas, eps=eps)

# momentum_list = []

vertices = mesh.vertices
faces = mesh.faces
ps.register_surface_mesh("my_sphere", vertices, faces, transparency=0.2, material="ceramic")

for i in range(steps):
    vadam.zero_grad()
    vbf = v.detach().cpu().clone()
    loss1 = sphere_energy(a, b, v)
    loss1.backward()
    vadam.step_modified(v, project=True, project_momentum=False, rotate_momentum=True)
    vaf = v.detach().cpu().clone()
    adam_step = vaf - vbf

    vadam.transport_momentum(vbf, vaf)
    # momentum_list.append(vadam.get_momentum())

    # print(v.detach().shape)
    normalized_v, _, _ = mesh.nearest.on_surface(v.detach().reshape(1,3).numpy())  # Ensure v.detach() is passed directly as a tensor

    with torch.no_grad():
        normalized_v = torch.tensor(normalized_v.squeeze(0), dtype=v.dtype, device=v.device)
        vector_list = np.concatenate((vector_list, adam_step.clone().numpy().reshape(1,3)), axis=0)
        position_list = np.concatenate((position_list, v.detach().numpy().reshape(1, 3)), axis=0)

    # Update the existing point cloud
    # v_cloud_list.update_point_positions(position_list)
    v_cloud_list = ps.register_point_cloud("Points", position_list.reshape(-1, 3), color=(0.48,0.99,0))
    v_cloud_list.add_vector_quantity("Gradient Vector", np.array(vector_list), vectortype='ambient', enabled=True, color=(0.48,0.99,0))

    ps.frame_tick()
    sleep(0.5)
ps.show()