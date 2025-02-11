import matplotlib.pyplot as plt
import json

class LossVisualizer:
    def __init__(self, log_file):
        self.log_file = log_file
        self.data = self.load_data()

    def load_data(self):
        with open(self.log_file, 'r') as f:
            return json.load(f)

    def plot_loss(self):
        loss = self.data["loss"]
        num_steps = self.data["num_steps"]
        radius = self.data["radius"]
        learning_rate = self.data["learning_rate"]
        betas = self.data["betas"]

        plt.figure(figsize=(10, 6))
        plt.plot(loss, label='Loss', marker='o')
        plt.title('Loss over Steps')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.xticks(range(num_steps))
        plt.grid(True)

        # Adding a small label with parameters
        plt.text(0.5, 0.95, f'Radius: {radius}, Learning Rate: {learning_rate}, Betas: {betas}', 
                 horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, 
                 fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

        plt.legend()
        plt.show()
