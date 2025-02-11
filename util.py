import sys
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import torch 
import polyscope as ps
from mpl_toolkits.mplot3d import Axes3D
from typing import List
from spherical_geometry.polygon import SphericalPolygon
from scipy.spatial import SphericalVoronoi


def print_list(loss_list: List[float], radius: float):
    for i in range(len(loss_list)):
        ending = ","
        if i == len(loss_list)-1:
            ending = ""
        print(str(loss_list[i]/(radius*radius)) + ending, end="", file=sys.stderr)
    print("", file=sys.stderr)


def random_point_on_sphere(r: float):
    theta = np.random.uniform(0, 2 * np.pi)
    
    phi = np.arccos(np.random.uniform(-1, 1))
    
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    
    point = torch.tensor([x, y, z], dtype=torch.float32)
    return point

def find_closest_point(a: torch.Tensor, b: torch.Tensor):
    x = (a[0] + b[0])/2
    y = (a[1] + b[1])/2
    z = (a[2] + b[2])/2
    return torch.tensor([x, y, z], dtype=torch.float32)

def project_point_to_sphere(point: torch.Tensor, radius: float) -> torch.Tensor:
    norm = torch.norm(point)
    normalized_point = point / norm
    projected_point = normalized_point * radius
    return projected_point


def normalize_tensor(tensor: torch.Tensor, radius: float):
    magnitude = tensor.norm()
    normalization_constant = radius / magnitude
    normalized_tensor = normalization_constant * tensor
    return normalized_tensor

def normalize_to_ellipsoid(tensor: torch.Tensor, scale_x: float, scale_y: float, scale_z: float) -> torch.Tensor:
    # Scale the input point to the ellipsoid's parameter space
    scaled_tensor = tensor / torch.tensor([scale_x, scale_y, scale_z], device=tensor.device)
    
    # Compute the magnitude of the scaled vector
    magnitude = scaled_tensor.norm()
    
    # Normalize the scaled tensor
    normalized_scaled_tensor = scaled_tensor / magnitude
    
    # Rescale back to the ellipsoid's dimensions
    normalized_tensor = normalized_scaled_tensor * torch.tensor([scale_x, scale_y, scale_z], device=tensor.device)
    
    return normalized_tensor

def create_pointer(x: float, y: float, z: float):
    return np.array([x, y, z])

def plot_vectors(ax, vector_list: List[torch.Tensor], position_list: List[torch.Tensor], color: str):
    if len(vector_list) <= 1:
        return
    
    for i in range(1, len(vector_list)):
        prev_vec = position_list[i-1]
        new_vec = vector_list[i].detach().cpu().numpy()
        ax.quiver(prev_vec[0], prev_vec[1], prev_vec[2], new_vec[0], new_vec[1], new_vec[2], color=color, normalize=False)
        
    print(vector_list)
    print(position_list)

def plot_sphere(ax, radius: float, a: torch.Tensor, b: torch.Tensor, n: torch.Tensor, not_normalized: torch.Tensor):
    # Generate spherical coordinates
    phi = np.linspace(0, np.pi, 50)  # Azimuthal angle
    theta = np.linspace(0, 2 * np.pi, 50)  # Polar angle

    phi, theta = np.meshgrid(phi, theta)

    # Convert spherical coordinates to Cartesian coordinates
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    # Plot the surface without edges
    ax.plot_surface(x, y, z, color='b', alpha=0.6, edgecolor='none')

    # Convert PyTorch tensors to NumPy arrays for plotting
    a_np = a.detach().cpu().numpy()
    b_np = b.detach().cpu().numpy()
    n_np = n.detach().cpu().numpy()
    nn_np = not_normalized.detach().cpu().numpy()

    # Plot the points a, b, and n
    ax.scatter(a_np[0], a_np[1], a_np[2], color='r', s=100, label='Point A')
    ax.scatter(b_np[0], b_np[1], b_np[2], color='g', s=100, label='Point B')
    ax.scatter(n_np[0], n_np[1], n_np[2], color='y', s=100, label='Point N (normalized)')
    ax.scatter(nn_np[0], nn_np[1], nn_np[2], color='k', s=100, label='Point N (not normalized)')

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set equal aspect ratio for all axes
    ax.set_box_aspect([1, 1, 1])

    # Set the limits
    ax.set_xlim([-radius*2, radius*2])
    ax.set_ylim([-radius*2, radius*2])
    ax.set_zlim([-radius*2, radius*2])

    # Add a legend
    ax.legend()

def sphere_energy(a, b, n):
    dist_a = torch.norm(n - a, p=2) ** 2
    dist_b = torch.norm(n - b, p=2) ** 2
    energy = 0.5 * (dist_a + dist_b)
    return energy

def laplacian_uniform_2d(v, l):
    V = v.shape[0]
    L = l.shape[0]
    
    #neighbor indices 
    ii = l[:,[1,0]].flatten()
    jj = l[:,[0,1]].flatten()
    adj = torch.stack([torch.cat([ii,jj]), torch.cat([jj,ii])], dim=0).unique(dim=1)
    adj_values = torch.ones(adj.shape[1], device='cuda', dtype=torch.float)    
    diag_idx = adj[0]
    idx = torch.cat((adj, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
    values = torch.cat((-adj_values, adj_values))
    L = torch.sparse_coo_tensor(idx, values, (V, V)).coalesce()
    return L

def plot_mesh2d(v, l, y_lim=None, x_lim=None, return_ax=False, showfig=False, filename=None):
    #with sns.axes_style('dark'):
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(5,5)
    ax.set_aspect('equal', adjustable='box')

    vtx = v[l, :]
    x = vtx[:, :, 0].reshape((-1, 1))
    y = vtx[:, :, 1].reshape((-1, 1))
    ax.plot(x, y, linewidth=4, color='#3b3d3f')
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if x_lim is not None:
        ax.set_xlim(x_lim)
    ax.axis("off")
    if showfig:
        plt.show()
    if filename is not None:
        plt.savefig(filename)
    if return_ax:
        return fig, ax
    else:
        plt.close()

def create_circle(n_points=20, radius=5, noise_level=1e-1):
    '''
    @output:
    vertices [np,2] point coordinates 
    lines [np-1,2] per-segment point id
    '''
    angles = np.linspace(2*np.pi - 2*np.pi/n_points, 0, n_points) # need to clockwise to match the gptoolbox output vertices order 
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    segment_id = [[i%n_points,(i+1)%n_points] for i in range(n_points)]
    vertices = np.stack([x,y], axis=1)
    lines = np.stack(segment_id, axis=0)

    vertices = vertices + np.random.normal(scale=noise_level,size=vertices.shape)
    return vertices, lines

def plotMesh2D(v_in=None, l_in=None, vn_in=None, ln_in=None, rv_in=None,
               v_tgt=None, l_tgt=None, vn_tgt=None, ln_tgt=None, rv_tgt=None, sdf_error = None,   
               nr=None, gradient=None, sdf=None, showfig=False, savefig=False, figname="image.png"):
    '''
    See https://towardsdatascience.com/the-many-ways-to-call-axes-in-matplotlib-2667a7b06e06#:~:text=Rarely%2C%20as%20for%20figure%20with,can%20find%20an%20example%20here) 
    to understand more.
    @input
    - data: list of list of [vertices, lines] data in which 
        - vertices: numpy array of shape [nv, 3] of ng groups of vertices to be visualized with different color 
        - lines: [ng, nl, 2]
    '''
    #>>> open a figure
    n_rows = 1
    n_cols = 0
    if v_in is not None: n_cols+=1 
    if v_tgt is not None: n_cols+=1 
    fig = plt.figure()
    fig.set_size_inches(20, 10.5)
    ax = fig.add_subplot(n_rows, n_cols, 1)
    canvas = FigureCanvas(fig)

    #>>> plot input mesh 
    #> get axes
    ax.set_aspect('equal', adjustable='box')
    
    #> get data
    v = v_in
    l = l_in
    
    #> set axes range
    rg = v.max() - v.min()
    ax.set_xlim(v.min() - rg/4, v.max() + rg/4)
    ax.set_ylim(v.min() - rg/4, v.max() + rg/4)
    vtx = v[l,:]
    x = vtx[:,:,0].reshape((-1,1))
    y = vtx[:,:,1].reshape((-1,1))
    ax.plot(x, y, linewidth=1, zorder=0)

    line_centers = np.mean(v[l,:],axis=1)
    if sdf_error is not None:
        for i in range(line_centers.shape[0]):
            ax.annotate("{:.2f}".format(sdf_error[i]), line_centers[i])
            ax.annotate(i, line_centers[i]-np.array([0.3,0]),color='r')

    #> visualize normals
    if vn_in is not None:
        ax.quiver(v[:,0],v[:,1],vn_in[:,0],vn_in[:,1])
    if ln_in is not None:
        line_centers = np.mean(v[l,:],axis=1)
        ax.quiver(line_centers[:,0],line_centers[:,1],ln_in[:,0],ln_in[:,1])

    #>>> plot rays on input mesh
    if rv_in is not None:
        # print(rv_in.shape)
        rv_in = rv_in.reshape(-1,2) #[ray0p0,ray0p1,ray1p0,ray1p1,...]
        # print(rv_in.shape)
        rl_in = np.array([[i*2,i*2+1] for i in range(rv_in.shape[0]//2)])
        for i in range(rl_in.shape[0]//nr):
            v = rv_in
            l = rl_in[i*nr:(i+1)*nr,]
            vtx = v[l,:]
            x = vtx[:,:,0].reshape((-1,1))
            y = vtx[:,:,1].reshape((-1,1))
            ax.plot(x,y,linewidth=0.5,color='orange',zorder=1)

    #> visualize gradients
    if gradient is not None:
        gradient = - gradient
        ax.quiver(v_in[:,0], v_in[:,1], gradient[:,0], gradient[:,1], 
                  angles='xy', 
                  scale_units='xy', 
                  scale=0.5,zorder=2)

    #>>> plot target mesh
    if v_tgt is not None and l_tgt is not None:
        #> get axes
        ax = fig.add_subplot(n_rows, n_cols, 2)
        ax.set_aspect('equal', adjustable='box')
    
        #> get data
        v = v_tgt
        l = l_tgt
    
        #> set axes range
        rg = v.max() - v.min()
        ax.set_xlim(v.min() - rg/4, v.max() + rg/4)
        ax.set_ylim(v.min() - rg/4, v.max() + rg/4)
        vtx = v[l,:]
        x = vtx[:,:,0].reshape((-1,1))
        y = vtx[:,:,1].reshape((-1,1))
        ax.plot(x,y,linewidth=1)

        #>>> plot rays on target mesh 
        if rv_tgt is not None:
            #> get data
            rv_tgt = rv_tgt.reshape(-1,2) #[ray0p0,ray0p1,ray1p0,ray1p1,...]
            rl_tgt = np.array([[i*2,i*2+1] for i in range(rv_tgt.shape[0]//2)]) 
            for i in range(rv_tgt.shape[0]//nr):
                v = rv_tgt
                l = rl_tgt[i*nr:(i+1)*nr,]
                vtx = v[l,:]
                x = vtx[:,:,0].reshape((-1,1))
                y = vtx[:,:,1].reshape((-1,1))
                ax.plot(x,y,linewidth=0.5,color='orange')

    if showfig:
        plt.show()
    if savefig:
        plt.savefig(figname)
    
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(int(height), int(width), 3)
    image = np.transpose(image, (2,0,1))
    plt.close()
    return image

def point_in_spherical_hull(p: torch.Tensor, pts: List[np.int64], sv: SphericalVoronoi) -> bool:
    """
    Check if point p lies within the spherical convex hull of points pts using spherical-geometry.
    
    Args:
        p: Point to check (torch.Tensor with dtype=torch.float64)
        pts: List of numpy int64 indices
        
    Returns:
        bool: True if p lies within the hull, False otherwise
    """    
    if len(pts) < 3:
        return False
    
    # Convert Cartesian coordinates to lon/lat
    def cart_to_lonlat(xyz: torch.Tensor) -> tuple[float, float]:
        lon = np.arctan2(xyz[1], xyz[0])
        lat = np.arcsin(xyz[2])
        return lon, lat
    
    # Convert hull points to lon/lat pairs
    hull_points = [cart_to_lonlat(pt) for pt in pts]
    
    # Create spherical polygon
    polygon = SphericalPolygon.from_lonlat(*zip(*hull_points))
    
    # Check if point is inside polygon
    return polygon.contains_point(p)

def compute_loss(centroidal_points: torch.Tensor, anchor_points: torch.Tensor):
    # Compute pairwise distances between centroids and anchors using PyTorch
    distances = torch.norm(centroidal_points[:, None, :] - anchor_points[None, :, :], dim=2)
    # Find the minimum distance for each anchor point
    min_distances, _ = torch.min(distances, dim=0)
    # Compute the loss as the sum of squared minimum distances
    loss = torch.sum(min_distances ** 2)
    return loss