from util import *
from time import sleep
from vectoradammodified import *
from vectoradam import * 

import trimesh
import polyscope as ps

ps.init()

lr = 1.5
betas = (0.9, 0.999)
eps = 1e-8
radius = 1

# Define steps constant
steps = 10

scale_x = 2.0  # Scale along the x-axis
scale_y = 1.0  # Scale along the y-axis
scale_z = 1.0  # Scale along the z-axis
starting_point = torch.from_numpy(create_pointer(0, 0, radius*scale_z))

v = starting_point.clone().numpy()
v = torch.from_numpy(v).to(torch.float32)
v.requires_grad_()

v_np = starting_point.clone().numpy().reshape(1,3)

a = torch.tensor([radius*scale_x, 0.0, 0.0], dtype=torch.float32)
b = torch.tensor([0.0, radius*scale_y, 0.0], dtype=torch.float32)

a_np = a.clone().numpy().reshape(1, 3)
b_np = b.clone().numpy().reshape(1, 3)

sphere = trimesh.primitives.Sphere(radius=radius)
ellipsoid_faces = sphere.faces
ellipsoid_vertices = sphere.vertices @ np.diag([scale_x, scale_y, scale_z])

a_cloud = ps.register_point_cloud("a", a_np, radius=0.02, color=(0,0,0))
b_cloud = ps.register_point_cloud("b", b_np, radius=0.02, color=(0,0,0))
v_cloud = ps.register_point_cloud("v", v_np, radius=0.02, color=(0.48,0.99,0))

vadam = VectorAdamModified([{'params': v, 'axis': -1}], lr=lr, betas=betas, eps=eps)

# Set up direction
ps.set_up_dir("y_up")

# ps.set_view_from_json(json_string)
ps.set_ground_plane_mode("shadow_only")

with open('elipse_view.json', 'r') as file:
    json_string = file.read()

# Create a Polyscope mesh
ps.register_surface_mesh("my_ellipsoid", ellipsoid_vertices, ellipsoid_faces, transparency=0.2, material="ceramic")

rotated_momentum = []

# ps.show()

# my_str = ps.get_view_as_json()
# print(my_str)
for i in range(steps):
    vadam.zero_grad()
    vbf = v.detach().cpu().clone()
    loss1 = sphere_energy(a, b, v)
    loss1.backward()
    vadam.step_modified(v, project=True, project_momentum=False, rotate_momentum=True)
    vaf = v.detach().cpu().clone()
    adam_step = vaf - vbf

    vadam.approximate_parallel_transport(vbf, vaf)

    vector_list = np.array(adam_step.clone().numpy().reshape(1,3))
    # momentum_vector = np.array(vadam.get_momentum().clone().numpy().reshape(1,3))
    momentum_vector = np.array([])

    normalized_v = normalize_to_ellipsoid(v.clone(), scale_x, scale_y, scale_z)

    with torch.no_grad():
        v.copy_(normalized_v)

        position_list = np.array(v.detach().numpy().reshape(1,3))

    # Update the points
    new_v_np = normalized_v.detach().numpy().reshape(1,3)

    v_cloud.update_point_positions(new_v_np)

    # v_cloud_list = ps.register_point_cloud("Points", position_list.reshape(-1,3), color=(0.48,0.99,0))
    # v_cloud_list.add_vector_quantity("Gradient Vector", np.array(momentum_vector), vectortype='ambient', enabled=True, color=(0.48,0.99,0))

    ps.frame_tick()
    sleep(0.5)
ps.show()