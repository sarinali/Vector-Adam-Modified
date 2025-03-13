import polyscope as ps
import torch
from vectoradammodified import VectorAdamModified
from graph_service import LossVisualizer
import json
from datetime import datetime
import os  # Import os module to handle directory operations

class Context:
    def __init__(self, radius: float, 
                lr: float, 
                betas: tuple, 
                eps: float,
                steps: int,
                params: torch.Tensor = None,
                num_anchor_points: int = 5000, #stress test this later, experiment with learning rate, plot loss, use the modified version
                verbose=False,
                write_to_file=False,
                logging_path="logs"
                ):
        self.radius = radius
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.steps = steps
        self.params = params
        self.num_anchor_points = num_anchor_points
        self.verbose = verbose
        self.write_to_file = write_to_file
        self.logging_path = logging_path
        ps.init()
        
        if self.verbose:
            print(f"Initialized Context with parameters:\n"
                  f"Radius: {self.radius}, Learning Rate: {self.lr}, Betas: {self.betas}, "
                  f"Epsilon: {self.eps}, Steps: {self.steps}, Number of Anchor Points: {self.num_anchor_points}")
        
        self.initialize_logging_data()  # Initialize logging data

    def initialize_logging_data(self):
        self.logging_data = {} 
        self.logging_data["num_anchor"] = self.num_anchor_points  
        self.logging_data["radius"] = self.radius
        self.logging_data["learning_rate"] = self.lr
        self.logging_data["betas"] = self.betas
        self.logging_data["epsilon"] = self.eps
        self.logging_data["num_steps"] = self.steps
        self.logging_data["loss"] = []

    def log_data_to_file(self):
        # Ensure the logging directory exists
        os.makedirs(self.logging_path, exist_ok=True)  # Create the directory if it doesn't exist
        
        filename = f"{self.logging_path}/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.json"  # Use logging_path as parent folder
        self.saved_file_path = filename

        with open(filename, 'w') as f:
            json.dump(self.logging_data, f, indent=4)
        
        if self.verbose:
            print(f"Logged data to {filename}")

    def show_loss_visualization_single_line(self, override_path=None):
        if override_path is not None:
            LossVisualizer(override_path).plot_loss()
        else:
            LossVisualizer(self.saved_file_path).plot_loss()