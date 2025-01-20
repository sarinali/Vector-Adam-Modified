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
                 params: torch.Tensor = None):
        super().__init__(radius, lr, betas, eps, steps, params)
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
        self.params = self.centroidal_points

    def register_points(self):
        points_tensor = torch.stack(self.centroidal_points)
        ps.register_point_cloud("centroids", points_tensor, color=(0, 0, 0))

    # register some anchor points on the mesh
    def register_anchor_points(self):
        self.anchor_points = self.anchor_sampler.generate_uniform_points()
        anchor_points_tensor = torch.stack(self.anchor_points)
        ps.register_point_cloud("anchor_points", anchor_points_tensor, color=(0, 0, 0), transparency=0.5)

    
    def generate_voronoi_diagram(self):
        self.center = np.array([0, 0, 0])
        self.sv = SphericalVoronoi(self.centroidal_points, self.radius, self.center)

    # def compute_loss(self):
    #     # go through all the centroid points 
    #     loss = 0.0
    #     region_mapping = self.get_region_mapping()
    #     for i in range(len(self.centroidal_points)):
    #         centroid = self.centroidal_points[i]
    #         for anchor in region_mapping[i]:
    #             loss += (centroid - anchor).norm() ** 2
    #     return loss
    
    def compute_loss(self):
        # Convert centroidal and anchor points to PyTorch tensors
        centroidal_points_tensor = torch.tensor(self.centroidal_points, requires_grad=True)
        anchor_points_tensor = torch.tensor(self.anchor_points)

        # Compute pairwise distances between centroids and anchors using PyTorch
        distances = torch.norm(centroidal_points_tensor[:, None, :] - anchor_points_tensor[None, :, :], dim=2)

        # Find the minimum distance for each anchor point
        min_distances, _ = torch.min(distances, dim=0)

        # Compute the loss as the sum of squared minimum distances
        loss = torch.sum(min_distances ** 2)

        return loss
            
    # index in self.sv -> list of anchor points that fall in the region
    def get_region_mapping(self):
        region_mapping = defaultdict(list)
        for anchor in self.anchor_points:
            for i in range(len(self.sv.regions)):
                region = self.sv.regions[i]
                point_region = [self.sv.vertices[r] for r in region]
                if point_in_spherical_hull(anchor, point_region, self.sv):
                    region_mapping[i].append(anchor)
        return region_mapping
            
    