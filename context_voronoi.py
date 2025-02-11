from collections import defaultdict
import torch
import trimesh
import polyscope as ps
import numpy as np
from vectoradammodified import VectorAdamModified
from context import Context
from uniform_sampling.uniform_sampler import UniformSampler
from scipy.spatial import SphericalVoronoi
from util import *

class ContextVoronoi(Context):
    def __init__(self, 
                 radius: float, 
                 lr: float, 
                 betas: tuple, 
                 eps: float,
                 steps: int,
                 num_observations: int, 
                 mesh_path: str = None,
                 mesh_scale: float = 1.0,
                 params: torch.Tensor = None,
                 verbose=False,
                 write_to_file=False,
                 ):
        super().__init__(radius, lr, betas, eps, steps, params, verbose=verbose, write_to_file=write_to_file)
        self.num_observations = num_observations
        self.uniform_sampler = UniformSampler(self.num_observations)
        self.anchor_sampler = UniformSampler(self.num_anchor_points)

        if mesh_path is None:
            self.mesh = trimesh.primitives.Sphere(self.radius)
        else:
            self.mesh = trimesh.load(mesh_path)
            self.mesh.apply_scale(mesh_scale)

        vertices = self.mesh.vertices
        faces = self.mesh.faces
        
        ps.register_surface_mesh("mesh", vertices, faces, transparency=0.2, material="ceramic")

    def generate_centroidal_points(self):
        self.centroidal_points = self.uniform_sampler.generate_uniform_points()
        self.centroidal_points = torch.stack(self.centroidal_points)

    def register_points(self):
        # Create a copy of the centroidal points for visualization
        self.centroids_display = ps.register_point_cloud("centroids", self.centroidal_points, color=(0, 0, 0))

    def update_centroids_display(self):
        centroidal_points_np = self.centroidal_points.detach().cpu().numpy()
        self.centroids_display.update_point_positions(centroidal_points_np)
        
    # register some anchor points on the mesh
    def register_anchor_points(self):
        self.anchor_points = self.anchor_sampler.generate_uniform_points()
        self.anchor_points = torch.stack(self.anchor_points)
        # ps.register_point_cloud("anchor_points", self.anchor_points, color=(0, 0, 0), transparency=0.5)

    def initialize_optimizer(self):
        self.vadam = VectorAdamModified([{'params': self.centroidal_points, 'axis': -1}], lr=self.lr, betas=self.betas, eps=self.eps)
    
    def initialize(self):
        self.generate_centroidal_points()
        self.register_points()
        ps.set_ground_plane_mode("shadow_only")
        # after registering the points on the mesh mark as requires grad
        self.centroidal_points.requires_grad_()
        self.register_anchor_points()
        self.initialize_optimizer()

    def average_distance_from_origin(self):
        distances = torch.norm(self.centroidal_points, dim=1)  # Compute distances from the origin
        # Print distances that exceed self.radius
        for distance in distances:
            if distance > self.radius:
                print(f"Distance {distance.item()} exceeds radius {self.radius}")
        average_distance = torch.mean(distances)  # Calculate the average distance
        return average_distance.item()  # Return as a Python float

    def perform_step(self):
        loss = compute_loss(self.centroidal_points, self.anchor_points)
        if self.verbose:
            print(f"Current loss: {loss}")
        if self.write_to_file:
            self.logging_data["loss"].append(loss.item())
        loss.backward()
        self.vadam.step_modified(self.centroidal_points, project=False)

        self.register_anchor_points()
    
    def compute_loss(self):
        anchor_points_tensor = torch.tensor(self.anchor_points)
        distances = torch.norm(self.centroidal_points[:, None, :] - anchor_points_tensor[None, :, :], dim=2)
        min_distances, _ = torch.min(distances, dim=0)
        loss = torch.sum(min_distances ** 2)

        return loss
    
    def normalize_points(self):
        # Normalize the centroidal points and scale to self.radius
        with torch.no_grad():  # Prevent tracking gradients during normalization
            normalized_centroidal_points = self.centroidal_points / torch.norm(self.centroidal_points, dim=1, keepdim=True) * self.radius
            self.centroidal_points.copy_(normalized_centroidal_points)  # Use copy_ to maintain the original tensor's gradient tracking
    