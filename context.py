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
                num_anchor_points: int = 6000 #stress test this later, experiment with learning rate, plot loss, use the modified version
                ):
        self.radius = radius
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.steps = steps
        self.params = params
        self.num_anchor_points = num_anchor_points
        ps.init()