from neighborhood_funcs import *
from pattern_generation import *

grid_base = generate_grid(50, 50, 4, 2)

gridA = grid_base

gridB = rotate(grid_base, 3)
gridB = translate(gridB, 1, 0)
gridB *= 1.02

gridB = np.array([[pt[0] + 1 * math.sin(10 * pt[1]),
                 pt[1]] for i, pt in enumerate(gridB)])

np.random.shuffle(gridB)
keep = int(len(gridB) * 0.9)
ptc = []
new_d = []
for i in range(keep):
    ptc.append(gridB[i])
gridB = np.array(ptc)

fig, ax = plt.subplots(1, 1)
ax.scatter(gridA[:, 0], gridA[:, 1], c='yellow', edgecolors='black')
ax.scatter(gridB[:, 0], gridB[:, 1], c='red', edgecolors='black')

#mtx1, mtx2, disp, R, trans, scale = mod_procrustes(gridA, gridB)
#gridR = procrustes_transformation(gridB, R, trans, scale)
gridR, R, trans, scale = iterative_procrustes(gridA,gridB)

ax.scatter(gridR[:, 0], gridR[:, 1], c='none', edgecolors='green', linewidth=1.5)

ax.set_aspect('equal')
plt.show()