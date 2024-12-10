import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def minimal_distance_moving_points(p1, v1, p2, v2):
    p1, v1, p2, v2 = map(np.array, (p1, v1, p2, v2))
    dp = p1 - p2
    dv = v1 - v2
    
    a = np.dot(dv, dv)
    b = 2 * np.dot(dp, dv)
    c = np.dot(dp, dp)
    
    if a == 0:
        t_min = 0
    else:
        t_min = -b / (2 * a)
    
    point1 = p1 + t_min * v1
    point2 = p2 + t_min * v2
    minimal_distance = np.linalg.norm(point1 - point2)
    
    return minimal_distance, t_min, point1, point2

# Parameters for the two moving points
p1 = [0, 0, 0]  # Initial position of the first point
v1 = [1, 0, 0]  # Velocity of the first point
p2 = [0, 1, 0]  # Initial position of the second point
v2 = [0, -1, 1]  # Velocity of the second point

distance, time_min, position1, position2 = minimal_distance_moving_points(p1, v1, p2, v2)

# Generate time values for the animation
t_values = np.linspace(0, max(2 * time_min, 2), 200)

# Compute positions of points over time
positions1 = np.array([np.array(p1) + t * np.array(v1) for t in t_values])
positions2 = np.array([np.array(p2) + t * np.array(v2) for t in t_values])

# Create the 3D animation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1, 10)
ax.set_ylim(-5, 5)
ax.set_zlim(-1, 5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Motion of Two Points')

# Plot trajectories
ax.plot(positions1[:, 0], positions1[:, 1], positions1[:, 2], 'b-', label='Trajectory of Point 1')
ax.plot(positions2[:, 0], positions2[:, 1], positions2[:, 2], 'r-', label='Trajectory of Point 2')

# Initial plot elements
point1_marker, = ax.plot([], [], [], 'bo', label='Point 1')
point2_marker, = ax.plot([], [], [], 'ro', label='Point 2')
distance_line, = ax.plot([], [], [], 'g--', label='Distance')
ax.legend()

def update(frame):
    # Update positions
    pos1 = positions1[frame]
    pos2 = positions2[frame]
    
    # Update point positions
    point1_marker.set_data([pos1[0]], [pos1[1]])
    point1_marker.set_3d_properties([pos1[2]])
    
    point2_marker.set_data([pos2[0]], [pos2[1]])
    point2_marker.set_3d_properties([pos2[2]])
    
    # Update line connecting the points
    distance_line.set_data([pos1[0], pos2[0]], [pos1[1], pos2[1]])
    distance_line.set_3d_properties([pos1[2], pos2[2]])
    
    return point1_marker, point2_marker, distance_line

ani = FuncAnimation(fig, update, frames=len(t_values), interval=50, blit=False)

# Display the animation
plt.show()
