import numpy as np
import torch
from typing import List
class UniformSampler:
    def __init__(self, observations: int):
        self.observations = observations

    def generate_point_on_sphere(self) -> np.ndarray:
        x1, x2, x3 = np.random.randn(3)

        norm = np.sqrt(x1**2+x2**2+x3**2)
        while norm < 0.001:
            norm = np.sqrt(x1**2+x2**2+x3**2)

        unit_vec = np.array([x1, x2, x3]) / norm
        return torch.from_numpy(unit_vec)
    
    def generate_uniform_points(self) -> List[np.ndarray]:
        points = []
        for _ in range(self.observations):
            points.append(self.generate_point_on_sphere())
        return points