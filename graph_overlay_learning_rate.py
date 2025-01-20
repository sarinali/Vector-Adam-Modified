from typing import List
import matplotlib.pyplot as plt
import mplcursors

class SphereSimulation:
    def __init__(self, learning_rate: float, projected: List[float], unprojected: List[float]) -> None:
        self.learning_rate = learning_rate
        self.projected = projected
        self.unprojected = unprojected
        
simulations = []
num_values = 0

with open('logs/20250106_073718_loss_logs.txt', 'r') as file:
    # Read the lines
    lines = file.readlines()

cur_line = 0

for i in range(0, 11):
    # Extract the first and second lines
    rate = float(lines[cur_line].strip())
    num_values = int(lines[cur_line+1].strip())

    # Extract the third and fourth lines and convert them to lists of floats
    list1 = [float(value) for value in lines[cur_line+2].strip().split(',')]
    list2 = [float(value) for value in lines[cur_line+3].strip().split(',')]
    cur_line += 4
    simulations.append(
        SphereSimulation(
            rate, list1, list2
        )
    )

x = range(num_values)

projected_avg = []
for i in range(num_values):
    val_sum = 0.0
    for j in range(len(simulations)):
        val_sum += simulations[j].projected[i]
    projected_avg.append(val_sum/len(simulations))

unprojected_avg = []
for i in range(num_values):
    val_sum = 0.0
    for j in range(len(simulations)):
        val_sum += simulations[j].unprojected[i]
    unprojected_avg.append(val_sum/len(simulations))

for i in range(0, len(simulations), 2):
    plt.plot(x, simulations[i].projected, label=f"Learning Rate {simulations[i].learning_rate} (Gradient: Projected, Momentum: Rotated)", alpha=0.2, color="blue")
    plt.plot(x, simulations[i].unprojected, label=f"Learning Rate {simulations[i].learning_rate} (Gradient: Projected, Momentum: Projected)", alpha=0.2, color="green")

plt.plot(x, projected_avg, label="Average Loss (Gradient: Projected, Momentum: Rotated)", color="blue")
plt.plot(x, unprojected_avg, label="Average Loss (Gradient: Projected, Momentum: Projected)", color="green")

plt.xlabel("Step Number")
plt.ylabel("Loss")

plt.title("Loss over different learning rates")
# plt.legend()

cursor = mplcursors.cursor(hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(sel.artist.get_label()))

plt.show()
