from neighborhood_funcs import *
from pattern_generation import *

grid_base = generate_grid(50, 50, 4, 2)

gridA = grid_base

gridB = rotate(grid_base, 90)
gridB = translate(gridB, 5, 0)

fig, ax = plt.subplots(1, 1)
ax.scatter(gridA[:, 0], gridA[:, 1], c='blue', edgecolors='black')
ax.scatter(gridB[:, 0], gridB[:, 1], c='red', edgecolors='black')

#gridR, real = mod_procrustes(gridA,gridB)
ax.scatter(gridR[:, 0], gridR[:, 1], c='green', edgecolors='black')

ax.set_aspect('equal')
plt.show()