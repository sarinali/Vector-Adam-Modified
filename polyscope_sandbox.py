from util import *
from time import sleep
from vectoradammodified import *
from vectoradam import * 

import json
import trimesh
import polyscope as ps
import argparse

parser = argparse.ArgumentParser(description='The short demo of the VectorAdam Modified algorithm for simple distance between points.')

ps.init()

parser.add_argument('arg2', type=int, help='Iterations')
parser.add_argument('--learning_rate', type=float, default=1.5, help='Learning rate (default: 1.5)')
parser.add_argument('-f', '--flag', action='store_true', help='A boolean flag for showing vectors')

# Parse the arguments
# args = parser.parse_args()

# Access the arguments
# print(args.arg1, file=sys.stderr)
# print(args.arg2, file=sys.stderr)
# print(args.learning_rate, file=sys.stderr)
# print(args.arg2, file=sys.stderr)

# Set optimizer hyperparameters
# lr = args.learning_rate
lr = 1.5
betas = (0.9, 0.999)
eps = 1e-8
radius = 1


# Define steps constant
steps = 30

# Define vector constant
# show_vector_flag = args.flag

starting_point = torch.from_numpy(create_pointer(0, 0, radius))
# starting_point = torch.from_numpy(create_pointer(0.707, 0.707, 0))
print(f"The starting point is {starting_point}.")

# Create the initial pointer n
v = starting_point.clone().numpy()
v = torch.from_numpy(v).to(torch.float32)
v.requires_grad_()

# create initial pointer n, unprojected version
v_unprojected = starting_point.clone().numpy()
v_unprojected = torch.from_numpy(v_unprojected).to(torch.float32)
v_unprojected.requires_grad_()

# visualize both points
v_np = starting_point.clone().numpy().reshape(1,3)
v_unprojected_np = starting_point.clone().numpy().reshape(1,3)

# create vector lists
v_np_list = starting_point.clone().numpy().reshape(1,3)

# initialize the two points
a = torch.tensor([radius, 0.0, 0.0], dtype=torch.float32)
b = torch.tensor([0.0, radius, 0.0], dtype=torch.float32)

# the tensors as polyscope points
a_np = a.clone().numpy().reshape(1, 3)
b_np = b.clone().numpy().reshape(1, 3)

a_cloud = ps.register_point_cloud("a", a_np, radius=0.02, color=(0,0,0))
b_cloud = ps.register_point_cloud("b", b_np, radius=0.02, color=(0,0,0))
v_cloud = ps.register_point_cloud("v", v_np, radius=0.02, color=(0.48,0.99,0))
v_unprojected_cloud = ps.register_point_cloud("v_unprojected", v_unprojected_np, radius=0.02, color=(1.0,0,0))

print(f"The two points a and b respectively are {a} and {b}")

vadam = VectorAdamModified([{'params': v, 'axis': -1}], lr=lr, betas=betas, eps=eps)
vadam_unnormalized = VectorAdam([{'params': v_unprojected, 'axis': -1}], lr=lr, betas=betas, eps=eps)

loss_list = []
loss_list_unnormalized = []

vector_list = np.array([v.detach().numpy().reshape(3)])
vector_list = vector_list.reshape(-1, 3)

vector_list = np.array([0, 0, 0])
vector_list = vector_list.reshape(-1, 3)

# vector_list_unprojected = np.array([v_unprojected.detach().numpy().reshape(3)])
# vector_list_unprojected = vector_list_unprojected.reshape(-1,3)

vector_list_unprojected = np.array([0, 0, 0])
vector_list_unprojected = vector_list_unprojected.reshape(-1,3)

position_list = np.array(v.detach().numpy().reshape(1,3))
position_list_unprojected = np.array(v_unprojected.detach().numpy().reshape(1,3))

# Generate a sphere mesh
sphere = trimesh.primitives.Sphere(radius=radius)

# Convert to Polyscope format
vertices = sphere.vertices
faces = sphere.faces

# Set up direction
ps.set_up_dir("y_up")

# Set camera to look at the point (1, 1, 1) from a diagonal bird's eye view toward the origin
with open('format.json', 'r') as file:
    json_string = file.read()

ps.set_view_from_json(json_string)
ps.set_ground_plane_mode("shadow_only")

# Create a Polyscope mesh
ps.register_surface_mesh("my_sphere", vertices, faces, transparency=0.2, material="ceramic")

for i in range(steps):
    vadam.zero_grad()
    vbf = v.detach().cpu().clone()
    loss1 = sphere_energy(a, b, v)
    loss1.backward()
    vadam.step_modified(v)
    vaf = v.detach().cpu().clone()
    adam_step = vaf - vbf

    # append to vector lists and loss lists
    vector_list = np.concatenate((vector_list, adam_step.clone().numpy().reshape(1,3)), axis=0)
    loss_list.append(loss1.item())

    vadam_unnormalized.zero_grad()
    vbf_unnormalized = v_unprojected.detach().cpu().clone()
    loss2 = sphere_energy(a, b, v_unprojected)
    loss2.backward()
    vadam_unnormalized.step()
    vaf_unnormalized = v_unprojected.detach().cpu().clone()
    adam_step_unnormalized = vaf_unnormalized - vbf_unnormalized

    # append to vector lists and loss lists (unprojected)
    vector_list_unprojected = np.concatenate((vector_list_unprojected, adam_step_unnormalized.clone().numpy().reshape(1,3)), axis=0)
    loss_list_unnormalized.append(loss2.item())

    normalized_v = normalize_tensor(v.clone(), radius)
    normalized_unprojected_v = normalize_tensor(v_unprojected.clone(), radius)

    with torch.no_grad():
        v.copy_(normalized_v)
        v_unprojected.copy_(normalized_unprojected_v)

        # for multiple vectors to show up
        position_list = np.concatenate((position_list, v.detach().numpy().reshape(1,3)), axis=0)
        position_list_unprojected = np.concatenate((position_list_unprojected, v_unprojected.detach().numpy().reshape(1,3)), axis=0)

    # Update the points
    new_v_np = normalized_v.detach().numpy().reshape(1,3)
    new_unprojected_v = normalized_unprojected_v.detach().numpy().reshape(1,3)

    # put on sphere
    v_cloud.update_point_positions(new_v_np)
    v_unprojected_cloud.update_point_positions(new_unprojected_v)

    v_cloud_list = ps.register_point_cloud("Points", position_list.reshape(-1,3), color=(0.48,0.99,0))
    v_cloud_list.add_vector_quantity("Gradient Vector", np.array(vector_list), vectortype='ambient', enabled=True, color=(0.48,0.99,0))

    v_unprojected_cloud_list = ps.register_point_cloud("Points (unprojected)", position_list_unprojected.reshape(-1,3), color=(1.0,0,0))
    v_unprojected_cloud_list.add_vector_quantity("Gradient Vector (unprojected)", np.array(vector_list_unprojected), color=(1.0,0,0), vectortype='ambient', enabled=True)

    ps.frame_tick()
    sleep(0.5)

ps.show()

# Disable interactive mode and show the final plot
print(f"The final position of projected VectorAdam is {v}, the intended position is {project_point_to_sphere(find_closest_point(a, b), radius)}")
print(f"The final position of the unnormalized and unproject point is {v_unprojected}, the intended position is {project_point_to_sphere(find_closest_point(a, b), radius)}")

# print_list(loss_list, radius)
# print_list(loss_list_unnormalized, radius)
