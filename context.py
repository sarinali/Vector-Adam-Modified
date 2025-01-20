import polyscope as ps
import torch
from vectoradammodified import VectorAdamModified

class Context:
    def __init__(self, radius: float, 
                lr: float, 
                betas: tuple, 
                eps: float,
                steps: int,
                params: torch.Tensor = None,
                num_anchor_points: int = 1000
                ):
        self.radius = radius
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.steps = steps
        self.params = params
        self.num_anchor_points = num_anchor_points
        ps.init()

    def initialize(self):
        self.optimizer = VectorAdamModified([{'params': self.params, 'axis': -1}], lr=self.lr, betas=self.betas, eps=self.eps)