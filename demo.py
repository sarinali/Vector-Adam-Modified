import matplotlib.pyplot as plt
import torch
from util import *
from vectoradam import * 

# Set optimizer hyperparameters
lr = 0.5
betas = (0.9, 0.999)
eps = 1e-8

# define radius constant
radius = 1

# create the initial pointer n
v = create_pointer(0, 0, 1)
v = torch.from_numpy(v).to(torch.float32)
v.requires_grad_()

# specify a and b to your liking, 2 points on a sphere.
a = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
b = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

# Initialize VectorAdam optimizers
vadam = VectorAdam([{'params': v, 'axis': -1}], lr=lr, betas=betas, eps=eps)

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


plot_sphere(ax, radius, a, b, v)
plt.draw()

loss_list = []

for i in range(100):
    vadam.zero_grad()
    vbf = v.detach().cpu().clone()
    loss1 = sphere_energy(a, b, v)
    loss1.backward()
    vadam.step()
    vaf = v.detach().cpu().clone()
    adam_step = vaf - vbf

    normalized_v = normalize_tensor(v, radius)
    loss_list.append(loss1.item())

    with torch.no_grad():
        v.copy_(normalized_v)

    # Clear the previous plot
    ax.cla()

    # Plot the sphere and points
    plot_sphere(ax, 1, a, b, v)

    # Add title to indicate the current iteration
    ax.set_title(f'Iteration {i+1}')

    # Pause to update the plot
    plt.draw()
    plt.pause(0.5)  # Pause for 0.5 seconds

# Disable interactive mode and show the final plot
plt.ioff()
plt.show()