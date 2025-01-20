import matplotlib.pyplot as plt
import torch
from util import *
from vectoradam import * 
from vectoradammodified import *
import matplotlib.gridspec as gridspec

import argparse

parser = argparse.ArgumentParser(description='The short demo of the VectorAdam Modified algorithm for simple distance between points.')

parser.add_argument('arg1', type=float, help='Radius')
parser.add_argument('arg2', type=int, help='Iterations')
parser.add_argument('--learning_rate', type=float, default=1.5, help='Learning rate (default: 1.5)')
parser.add_argument('-f', '--flag', action='store_true', help='A boolean flag for showing vectors')

# Parse the arguments
args = parser.parse_args()

# Access the arguments
if args.learning_rate:
    print(args.learning_rate, file=sys.stderr)
else:
    print(args.arg1, file=sys.stderr)
# print(args.arg2, file=sys.stderr)
# print(args.learning_rate, file=sys.stderr)
print(args.arg2, file=sys.stderr)

# Set optimizer hyperparameters
lr = args.learning_rate
betas = (0.9, 0.999)
eps = 1e-8

# Define radius constant
radius = args.arg1

# Define vector constant
show_vector_flag = args.flag

# starting_point = random_point_on_sphere(radius)
starting_point = torch.from_numpy(create_pointer(0, 0, radius))
# print(f"The starting point is {starting_point}.")

# Create the initial pointer n
v = starting_point.clone().numpy()
v = torch.from_numpy(v).to(torch.float32)
v.requires_grad_()

# create initial pointer n, unnormalized version
v_unprojected = starting_point.clone().numpy()
v_unprojected = torch.from_numpy(v_unprojected).to(torch.float32)
v_unprojected.requires_grad_()

# initialize the two points
a = torch.tensor([radius, 0.0, 0.0], dtype=torch.float32)
# a = random_point_on_sphere(radius)
b = torch.tensor([0.0, radius, 0.0], dtype=torch.float32)
# b = random_point_on_sphere(radius)

# print(f"The two points     a and b respectively are {a} and {b}")

# Initialize VectorAdam optimizer
vadam = VectorAdamModified([{'params': v, 'axis': -1}], lr=lr, betas=betas, eps=eps, r=radius)
vadam_unnormalized = VectorAdamModified([{'params': v_unprojected, 'axis': -1}], lr=lr, betas=betas, eps=eps, r=radius)

# Enable interactive mode
plt.ion()
fig = plt.figure(figsize=(21, 7))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

# ax_sphere = fig.add_subplot(gs[0], projection='3d')
ax_loss = fig.add_subplot(gs[0])
ax_loss_unnormalized = fig.add_subplot(gs[1])

# Plot the initial sphere
# plot_sphere(ax_sphere, radius, a, b, v, v_unprojected)
# plt.draw()

loss_list = []
loss_list_unnormalized = []

vector_list = [v.detach().cpu().clone()]
vector_list_unprojected = [v_unprojected.detach().cpu().clone()]

position_list = [v.detach().cpu().clone()]
position_list_unprojected = [v_unprojected.detach().cpu().clone()]

for i in range(args.arg2):
    vadam.zero_grad()
    vbf = v.detach().cpu().clone()
    loss1 = sphere_energy(a, b, v)
    loss1.backward()
    vadam.step_modified(v, project=True, project_momentum=False)
    vaf = v.detach().cpu().clone()
    adam_step = vaf - vbf

    # rotate the momentum
    vadam.transport_momentum(vbf, vaf)

    # print(adam_step)
    vector_list.append(adam_step.clone())
    loss_list.append(loss1.item())

    vadam_unnormalized.zero_grad()
    vbf_unnormalized = v_unprojected.detach().cpu().clone()
    loss2 = sphere_energy(a, b, v_unprojected)
    loss2.backward()
    # project the gradient and the momentum
    vadam_unnormalized.step_modified(v_unprojected, project=True, project_momentum=True)
    vaf_unnormalized = v_unprojected.detach().cpu().clone()
    adam_step_unnormalized = vaf_unnormalized - vbf_unnormalized

    # print(adam_step_unnormalized)
    vector_list_unprojected.append(adam_step_unnormalized.clone())
    loss_list_unnormalized.append(loss2.item())

    normalized_v = normalize_tensor(v.clone(), radius)
    normalized_unprojected_v = normalize_tensor(v_unprojected.clone(), radius)

    with torch.no_grad():
        v.copy_(normalized_v)
        v_unprojected.copy_(normalized_unprojected_v)

    if show_vector_flag:
        position_list.append(v.detach().cpu().clone())
        position_list_unprojected.append(v_unprojected.detach().cpu().clone())

    # Clear the previous plots
    # ax_sphere.cla()
    # ax_loss.cla()
    # ax_loss_unnormalized.cla()

    # Plot the sphere and points
    # plot_sphere(ax_sphere, radius, a, b, v, v_unprojected)
    # if show_vector_flag:
    #     plot_vectors(ax_sphere, vector_list, position_list, 'g')
    #     plot_vectors(ax_sphere, vector_list_unprojected, position_list_unprojected, 'm')
    # ax_sphere.set_title(f'Iteration {i+1}')

    # Plot the loss
    # ax_loss.plot(loss_list, label='Loss')
    # ax_loss.set_xlabel('Iteration')
    # ax_loss.set_ylabel('Loss')
    # ax_loss.set_title('Loss Over Iterations (Gradient: Projected, Momentum: Rotated)')
    # ax_loss.legend()

    # # Plot the loss
    # ax_loss_unnormalized.plot(loss_list_unnormalized, label='Loss')
    # ax_loss_unnormalized.set_xlabel('Iteration')
    # ax_loss_unnormalized.set_ylabel('Loss')
    # ax_loss_unnormalized.set_title('Loss Over Iterations (Gradient: Projected, Momentum: Projected)')
    # ax_loss_unnormalized.legend()

    # Pause to update the plot
    # plt.draw()
    # plt.pause(0.1)  # Pause for 0.5 seconds

# Disable interactive mode and show the final plot
# print(f"The final position of projected VectorAdam is {v}, the intended position is {project_point_to_sphere(find_closest_point(a, b), radius)}")
# print(f"The final position of the unnormalized and unproject point is {v_unprojected}, the intended position is {find_closest_point(a, b)}")

print_list(loss_list, radius)
print_list(loss_list_unnormalized, radius)

# plt.ioff()
# plt.show()

# plt.close()