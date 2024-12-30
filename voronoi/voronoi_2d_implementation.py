import numpy as np
import matplotlib.pyplot as plt

# Define sites
sites = np.array([[2, 3], [6, 5], [8, 1]])

# Define the bounding box
bounding_box = [0, 10, 0, 10]  # [x_min, x_max, y_min, y_max]

# Function to compute the boundary line equation between two points
def boundary_line(p1, p2):
    # Calculate midpoint
    mid_x = (p1[0] + p2[0]) / 2
    mid_y = (p1[1] + p2[1]) / 2
    # Calculate perpendicular slope
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dy == 0:  # Horizontal line
        slope = None  # Infinite slope for vertical boundary
        intercept = mid_x
    elif dx == 0:  # Vertical line
        slope = 0  # Horizontal boundary
        intercept = mid_y
    else:
        slope = -dx / dy
        intercept = mid_y - slope * mid_x
    return slope, intercept

# Function to clip a line segment to the bounding box
def clip_line_segment(slope, intercept, bbox):
    x_min, x_max, y_min, y_max = bbox
    points = []
    
    # Handle vertical lines
    if slope is None:
        x = intercept
        if x_min <= x <= x_max:
            points = [(x, y_min), (x, y_max)]
    # Handle horizontal lines
    elif slope == 0:
        y = intercept
        if y_min <= y <= y_max:
            points = [(x_min, y), (x_max, y)]
    else:
        # Compute intersections with the bounding box
        y_at_x_min = slope * x_min + intercept
        y_at_x_max = slope * x_max + intercept
        x_at_y_min = (y_min - intercept) / slope
        x_at_y_max = (y_max - intercept) / slope
        
        # Add valid points within the bounding box
        if y_min <= y_at_x_min <= y_max:
            points.append((x_min, y_at_x_min))
        if y_min <= y_at_x_max <= y_max:
            points.append((x_max, y_at_x_max))
        if x_min <= x_at_y_min <= x_max:
            points.append((x_at_y_min, y_min))
        if x_min <= x_at_y_max <= x_max:
            points.append((x_at_y_max, y_max))
    
    # Return at most two points (start and end of the segment)
    return points[:2]

# Compute boundaries
boundaries = []
for i in range(len(sites)):
    for j in range(i + 1, len(sites)):
        slope, intercept = boundary_line(sites[i], sites[j])
        clipped_segment = clip_line_segment(slope, intercept, bounding_box)
        if len(clipped_segment) == 2:  # Only store valid line segments
            boundaries.append(clipped_segment)

# Plot
plt.figure(figsize=(8, 8))
# Plot the bounding box
plt.plot([0, 0, 10, 10, 0], [0, 10, 10, 0, 0], 'b-', label="Bounding Box")

# Plot the boundaries
for segment in boundaries:
    x_vals, y_vals = zip(*segment)
    plt.plot(x_vals, y_vals, 'k-', label='Boundary Line')

# Plot the sites
plt.scatter(sites[:, 0], sites[:, 1], color='red', label='Sites', zorder=5)
plt.legend()
plt.title('Voronoi Diagram (Clipped Boundaries)')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.show()
