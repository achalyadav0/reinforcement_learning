import numpy as np
import matplotlib.pyplot as plt

# Grid size
rows, cols = 20, 20

# Create random grid
obstacle_prob = 0.3
grid = np.random.choice([0, 1], size=(rows, cols),
                        p=[1 - obstacle_prob, obstacle_prob])

# Define start and goal
start = (0, 0)
goal = (19, 19)

grid[start] = 2
grid[goal] = 3

# Create color map
# 0 = Free (white)
# 1 = Obstacle (black)
# 2 = Start (green)
# 3 = Goal (red)

cmap = plt.cm.colors.ListedColormap(
    ['white', 'black', 'green', 'red']
)

# Plot
plt.figure(figsize=(7, 7))
plt.imshow(grid, cmap=cmap)

plt.title("2D Grid Map")
plt.xticks(range(cols))
plt.yticks(range(rows))
plt.grid(False)

plt.show()