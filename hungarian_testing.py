import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

np.random.seed(100)

points1 = np.array([(x, y) for x in np.linspace(-1,1,5) for y in np.linspace(-1,1,5)])
N = points1.shape[0] + 5
points2 = 2*np.random.rand(N,2)-1

C = cdist(points1, points2)

_, assigment = linear_sum_assignment(C)

plt.plot(points1[:,0], points1[:,1],'bo', markersize = 10)
plt.plot(points2[:,0], points2[:,1],'rs',  markersize = 7)
for p in range(N):
    try:
        plt.plot([points1[p,0], points2[assigment[p],0]], [points1[p,1], points2[assigment[p],1]], 'k')
    except IndexError:
        pass
plt.xlim(-1.1,1.1)
plt.ylim(-1.1,1.1)
plt.axes().set_aspect('equal')
plt.show()
