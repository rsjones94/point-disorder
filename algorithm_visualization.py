import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy

from neighborhood_funcs import *
from pattern_generation import *
from icp import icp

np.random.seed(0)
poi_1 = 370


outloc = r'C:\Users\rsjon_000\Documents\point-disorder\point_disorder_paper\figures\neighborhood_generation.png'

params = {'neighbor_search_dist': 45,
          'ka': 6,
          'coop': 6,
          'punishment': 1,
          'punish_out_of_hull': False,
          'euc': False,
          'reorientation': 10e-3}

outloc2 = f'C:\\Users\\rsjon_000\\Documents\\point-disorder\\point_disorder_paper\\figures\\' \
          f'scoring_func.png'

neighb_num = 43

distance_metric1 = 'euclidean'
distance_metric2 = lambda p1, p2: score_distance_p1p2(p1, p2, params['ka'], params['coop'])
d_mecs = [distance_metric1, distance_metric2]
plot_positions = [(0,1),(1,0)]
exes = np.linspace(0,15,500)
whys = [distance_metric2([0,0],[0,x]) for x in exes]

grid_a = generate_grid(100,100,10,5)
grid_b = translate(grid_a, 4, 6)

total = np.append(grid_a, grid_b, 0)
total = peturb_constant(total, 0.75)

for i in range(10):
    total = np.append(total,np.random.rand(1,2)*100, 0)

neighbors = compose_neighborhoods(total, params['neighbor_search_dist'])

poi_2 = neighbors[poi_1]['neighbors'][neighb_num]

fig, ax = plt.subplots(2, 2, figsize=(8, 8))
scores = []

ax[0][0].scatter(total[:,0], total[:,1], color='gold', edgecolors='black')
ax[0][0].scatter(total[poi_1,0], total[poi_1,1], color='red', edgecolors='black')
ax[0][0].scatter(total[poi_2,0], total[poi_2,1], color='blue', edgecolors='black')

c1 = plt.Circle((total[poi_1,0], total[poi_1,1]), params['neighbor_search_dist'], color='red', linewidth=2, fill=False)
ax[0][0].add_patch(c1)
c1 = plt.Circle((total[poi_2,0], total[poi_2,1]), params['neighbor_search_dist'], color='blue', linewidth=2, fill=False)
ax[0][0].add_patch(c1)
ax[0][0].set_aspect('equal')

for (pax,pay), distance_metric in zip(plot_positions, d_mecs):
    set_1 = neighbors[poi_1]['coords']
    set_2 = neighbors[poi_2]['coords']
    ax[pax][pay].scatter(set_1[:,0], set_1[:,1], color='red', edgecolors='black')
    ax[pax][pay].scatter(set_2[:,0], set_2[:,1], color='blue', edgecolors='black')

    swapped = False
    s1, s2 = set_1, set_2
    if len(set_1) > len(set_2):
        s1, s2 = set_2, set_1
        swapped = True


    C = cdist(s1, s2, metric=distance_metric)
    _, assignment = linear_sum_assignment(C)

    #  some points in s2 may not be matched. We need to exclude them from the convex hull
    assigned_coords = [s2[i] for i in assignment]
    pared_s2 = assigned_coords.copy()
    unpaired_cords = [s for i, s in enumerate(s2) if i not in assignment]
    assigned_coords.extend(s1)  # we know all of s1 is matched because its length is always < s2
    assigned_coords = np.array(assigned_coords)
    try:
        hull = scipy.spatial.ConvexHull(assigned_coords)
        unpaired_cords_in_hull = [s for s in unpaired_cords if in_hull(s, hull.points)]
    except scipy.spatial.qhull.QhullError:
        hull = None
        unpaired_cords_in_hull = unpaired_cords

    n_smaller = len(s1)
    n_bigger = len(s2)
    n_unpaired = len(unpaired_cords)
    n_unpaired_in_hull = len(unpaired_cords_in_hull)
    deviations = [distance(p, s2[assignment[i]]) for i, p in enumerate(s1)]
    if True:
        scored_vals = [distance_metric2(p, s2[assignment[i]]) for i, p in enumerate(s1)]
    else:
        scored_vals = deviations

    scores.append(np.mean(scored_vals))

    for simplex in hull.simplices:
        ax[pax][pay].plot(assigned_coords[simplex, 0], assigned_coords[simplex, 1], 'k-', color='green', lw=2)

    for p in range(min([len(s1), len(s2)])):
        ax[pax][pay].plot([s1[p, 0], s2[assignment[p], 0]], [s1[p, 1], s2[assignment[p], 1]], 'k')

    ax[pax][pay].set_aspect('equal')

ax[0][0].set_title('Neighborhoods')
ax[0][1].set_title('Euclidean registration\n'
                   f'Score: {round(np.mean(scores[0]),3)}')
ax[1][0].set_title('Alternative registration\n'
                   f'Score: {round(np.mean(scores[1]),3)}')

"""
ax[1][1].scatter(set_1[:, 0], set_1[:, 1], color='red', edgecolors='black')
set_2, R_cum, trans_cum, scale_cum = iterative_procrustes(set_1, set_2, distance_metric1)
ax[1][1].scatter(set_2[:, 0], set_2[:, 1], color='blue', edgecolors='black')

pax = 1
pay = 1
distance_metric = distance_metric2
ax[pax][pay].scatter(set_1[:,0], set_1[:,1], color='red', edgecolors='black')
ax[pax][pay].scatter(set_2[:,0], set_2[:,1], color='blue', edgecolors='black')

swapped = False
s1, s2 = set_1, set_2
if len(set_1) > len(set_2):
    s1, s2 = set_2, set_1
    swapped = True


C = cdist(s1, s2, metric=distance_metric)
_, assignment = linear_sum_assignment(C)

#  some points in s2 may not be matched. We need to exclude them from the convex hull
assigned_coords = [s2[i] for i in assignment]
pared_s2 = assigned_coords.copy()
unpaired_cords = [s for i, s in enumerate(s2) if i not in assignment]
assigned_coords.extend(s1)  # we know all of s1 is matched because its length is always < s2
assigned_coords = np.array(assigned_coords)
try:
    hull = scipy.spatial.ConvexHull(assigned_coords)
    unpaired_cords_in_hull = [s for s in unpaired_cords if in_hull(s, hull.points)]
except scipy.spatial.qhull.QhullError:
    hull = None
    unpaired_cords_in_hull = unpaired_cords

n_smaller = len(s1)
n_bigger = len(s2)
n_unpaired = len(unpaired_cords)
n_unpaired_in_hull = len(unpaired_cords_in_hull)
deviations = [distance(p, s2[assignment[i]]) for i, p in enumerate(s1)]
if distance_metric is not 'euclidean':
    scored_vals = [distance_metric(p, s2[assignment[i]]) for i, p in enumerate(s1)]
else:
    scored_vals = deviations

for simplex in hull.simplices:
    ax[pax][pay].plot(assigned_coords[simplex, 0], assigned_coords[simplex, 1], 'k-', color='green', lw=2)

for p in range(min([len(s1), len(s2)])):
    ax[pax][pay].plot([s1[p, 0], s2[assignment[p], 0]], [s1[p, 1], s2[assignment[p], 1]], 'k')
"""

## ICP REALIGNMENT
t, distances, iterations = icp(set_2, set_1, distance_metric=distance_metric2)
C = np.ones((len(set_2), 3))
C[:, 0:2] = np.copy(set_2)
set_2_readj = np.dot(t, C.T).T
## END ICP REALIGNMENT

ax[1][1].scatter(set_1[:,0], set_1[:,1], color='red', edgecolors='black')
ax[1][1].scatter(set_2_readj[:,0], set_2_readj[:,1], color='blue', edgecolors='black')
ax[1][1].set_aspect('equal')

# register the realigned points
set_2 = set_2_readj[:,0:2]
pax = 1
pay = 1
ax[pax][pay].scatter(set_1[:, 0], set_1[:, 1], color='red', edgecolors='black')
ax[pax][pay].scatter(set_2[:, 0], set_2[:, 1], color='blue', edgecolors='black')

swapped = False
s1, s2 = set_1, set_2
if len(set_1) > len(set_2):
    s1, s2 = set_2, set_1
    swapped = True

C = cdist(s1, s2, metric=distance_metric2)
_, assignment = linear_sum_assignment(C)

#  some points in s2 may not be matched. We need to exclude them from the convex hull
assigned_coords = [s2[i] for i in assignment]
pared_s2 = assigned_coords.copy()
unpaired_cords = [s for i, s in enumerate(s2) if i not in assignment]
assigned_coords.extend(s1)  # we know all of s1 is matched because its length is always < s2
assigned_coords = np.array(assigned_coords)
try:
    hull = scipy.spatial.ConvexHull(assigned_coords)
    unpaired_cords_in_hull = [s for s in unpaired_cords if in_hull(s, hull.points)]
except scipy.spatial.qhull.QhullError:
    hull = None
    unpaired_cords_in_hull = unpaired_cords

n_smaller = len(s1)
n_bigger = len(s2)
n_unpaired = len(unpaired_cords)
n_unpaired_in_hull = len(unpaired_cords_in_hull)
deviations = [distance(p, s2[assignment[i]]) for i, p in enumerate(s1)]
if distance_metric is not 'euclidean':
    scored_vals = [distance_metric(p, s2[assignment[i]]) for i, p in enumerate(s1)]
else:
    scored_vals = deviations

for simplex in hull.simplices:
    ax[pax][pay].plot(assigned_coords[simplex, 0], assigned_coords[simplex, 1], 'k-', color='green', lw=2)

for p in range(min([len(s1), len(s2)])):
    ax[pax][pay].plot([s1[p, 0], s2[assignment[p], 0]], [s1[p, 1], s2[assignment[p], 1]], 'k')

ax[1][1].set_title(f'ICP realignment + alternative registration\n'
                   f'Score: {round(np.mean(scored_vals),3)}')
ax[pax][pay].set_aspect('equal')

plt.tight_layout()
plt.close()
fig.savefig(outloc)


fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.plot(exes,whys,color='dodgerblue')
ax.set_aspect(15)
ax.set_title(f'Scoring Function: Sigmoidal (Km = {params["ka"]}, n = {params["coop"]})')
ax.set(xlabel='Deviation', ylabel='Score')
ax.plot([0,max(exes)], [1,1], '--', color='dodgerblue')
ax.plot([0,params['ka'],params['ka']], [0.5,0.5,0], '--', color='green')
plt.close()
fig.savefig(outloc2)

