from neighborhood_funcs import *
from pattern_generation import *

grid_base = generate_grid(50, 50, 4, 2)

gridA = grid_base

gridB = rotate(grid_base, -123)
gridB = translate(gridB, 11, 6)
gridB *= 1.2

fig, ax = plt.subplots(1, 1)
ax.scatter(gridA[:, 0], gridA[:, 1], c='blue', edgecolors='black')
ax.scatter(gridB[:, 0], gridB[:, 1], c='red', edgecolors='black')

mtx1, mtx2, disp, R, trans, scale = mod_procrustes(gridA, gridB)

gridR = procrustes_transformation(gridB, R, trans, scale)

ax.scatter(gridR[:, 0], gridR[:, 1], c='none', edgecolors='green')

ax.set_aspect('equal')
plt.show()