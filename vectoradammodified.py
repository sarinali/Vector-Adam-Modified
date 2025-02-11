import torch
from numpy.typing import NDArray
import numpy as np
from typing import Union
from scipy.spatial.transform import Rotation

class VectorAdamModified(torch.optim.Optimizer):
    momentum = None
    rotation_matrix = None
    def __init__(self, params, lr=0.1, betas=(0.9, 0.999), eps=1e-8, axis=-1, r=1):
        defaults = dict(lr=lr, betas=betas, eps=eps, axis=axis)
        self.radius = r
        super(VectorAdamModified, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(VectorAdamModified, self).__setstate__(state)

    def get_rotated_vector(self, vector: NDArray[Union[np.float64, np.int_]]) -> NDArray[np.float64]:
        rotated_vector = np.dot(self.rotation_matrix, vector)
        return rotated_vector

    def calculate_rotation_matrix(self, start_point: NDArray[Union[np.float64, np.int_]], end_point: NDArray[Union[np.float64, np.int_]]):
        rotation, _ = Rotation.align_vectors(np.reshape(start_point, (1, 3)), np.reshape(end_point, (1, 3)))
        self.rotation_matrix = rotation.as_matrix()

    def calculate_and_return_rotation_matrix(self, start_point: NDArray[Union[np.float64, np.int_]], end_point: NDArray[Union[np.float64, np.int_]]):
        rotation, _ = Rotation.align_vectors(np.reshape(start_point, (1, 3)), np.reshape(end_point, (1, 3)))
        self.rotation_matrix = rotation.as_matrix()
        return self.rotation_matrix

    def transport_momentum(self, before: torch.Tensor, after: torch.Tensor):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                self.calculate_rotation_matrix(before.numpy(), after.numpy())
                g1_np = state["g1"].cpu().numpy()
                rotated_g1_np = self.get_rotated_vector(g1_np)

                rotated_g1_tensor = torch.from_numpy(rotated_g1_np).to(state["g1"].device)
                state["g1"] = rotated_g1_tensor
                self.momentum = rotated_g1_np

    def approximate_parallel_transport(self, before: torch.Tensor, after: torch.Tensor, steps: int = 50):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                
                # Extract the vector to be transported
                g1_np = state["g1"].cpu().numpy()
                
                # Generate intermediate points along the path
                path = np.linspace(before.cpu().numpy(), after.cpu().numpy(), steps)
                
                # Initialize the transported vector
                transported_vector = g1_np
                
                for i in range(steps - 1):
                    current_point = path[i]
                    next_point = path[i + 1]
                    
                    # Compute the rotation matrix between the tangent spaces at the two points
                    rotation_matrix= self.calculate_and_return_rotation_matrix(current_point, next_point)

                    # Rotate the vector
                    transported_vector = np.dot(rotation_matrix, transported_vector)
                
                # Convert back to a tensor and update the state
                transported_vector_tensor = torch.from_numpy(transported_vector).to(state["g1"].device)
                state["g1"] = transported_vector_tensor

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            b1, b2 = group['betas']
            eps = group['eps']
            axis = group['axis']
            for p in group["params"]:
                state = self.state[p]
                # Lazy initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["g1"] = torch.zeros_like(p.data)
                    state["g2"] = torch.zeros_like(p.data)

                g1 = state["g1"]
                g2 = state["g2"]

                state["step"] += 1
                grad = p.grad.data

                g1.mul_(b1).add_(grad, alpha=1-b1)
                self.momentum = g1
                if axis is not None:
                    dim = grad.shape[axis]
                    grad_norm = torch.norm(grad, dim=axis).unsqueeze(axis).repeat_interleave(dim, dim=axis)
                    grad_sq = grad_norm * grad_norm
                    g2.mul_(b2).add_(grad_sq, alpha=1-b2)
                else:
                    g2.mul_(b2).add_(grad.square(), alpha=1-b2)

                m1 = g1 / (1-(b1**state["step"]))
                m2 = g2 / (1-(b2**state["step"]))
                gr = m1 / (eps + m2.sqrt())
                p.data.sub_(gr, alpha=lr)

    @torch.no_grad()
    def step_modified(self, cur_point: torch.Tensor, project=True, project_momentum=True, rotate_momentum=False):
        for group in self.param_groups:
            lr = group['lr']
            b1, b2 = group['betas']
            eps = group['eps']
            axis = group['axis']
            for p in group["params"]:
                state = self.state[p]
                # Lazy initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["g1"] = torch.zeros_like(p.data)
                    state["g2"] = torch.zeros_like(p.data)

                g1 = state["g1"]
                g2 = state["g2"]

                state["step"] += 1
                grad = p.grad.data

                # v is gr and n is cur_point. find the projection of v onto the tangent plane of n
                if project:
                    dot_numerator = torch.dot(cur_point, grad)
                    dot_denominator = torch.dot(cur_point, cur_point)
                    proj_v_on_n = (dot_numerator / dot_denominator) * cur_point

                    grad = grad - proj_v_on_n

                    if project_momentum:
                        # project m1 onto grad
                        g1.mul_(b1).add_(grad, alpha=1-b1)
                        dot_numerator = torch.dot(g1.float(), grad.float())
                        dot_denominator = torch.dot(grad.float(), grad.float())
                        # g1 = (dot_numerator / dot_denominator) * g1
                        g1 = (dot_numerator / dot_denominator) * grad
                    else:
                        g1.mul_(b1).add_(grad, alpha=1-b1)
                else:
                    g1.mul_(b1).add_(grad, alpha=1-b1)

                if project_momentum:
                    self.momentum = g1

                # norm_g1 = torch.norm(g1)
                # unit_tensor = g1 / norm_g1
                # g1 = self.radius * unit_tensor
                # self.momentum = g1
                
                if axis is not None:
                    dim = grad.shape[axis]
                    grad_norm = torch.norm(grad, dim=axis).unsqueeze(axis).repeat_interleave(dim, dim=axis)
                    grad_sq = grad_norm * grad_norm
                    g2.mul_(b2).add_(grad_sq, alpha=1-b2)
                else:
                    g2.mul_(b2).add_(grad.square(), alpha=1-b2)

                m1 = g1 / (1-(b1**state["step"]))
            
                m2 = g2 / (1-(b2**state["step"]))
                gr = m1 / (eps + m2.sqrt())

                p.data.sub_(gr, alpha=lr)

    def get_momentum(self) -> torch.Tensor:
        return self.momentum