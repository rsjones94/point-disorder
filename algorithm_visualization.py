import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
import matplotlib.patheffects as PathEffects

from neighborhood_funcs import *
from pattern_generation import *
from icp import icp, realign_points

np.random.seed(0)
poi_1 = 44
neighb_num = -3
neighb_num2 = -5

write = True

params = {'neighbor_search_dist': 15,
          'ka': 2,
          'coop': 2,
          'punishment': 1,
          'punish_out_of_hull': False,
          'euc': False,
          'reorientation': None}

outloc = r'C:\Users\rsjon_000\Documents\point-disorder\point_disorder_paper\figures\algorithm_demonstration.png'

distance_metric1 = 'euclidean'
distance_metric2 = lambda p1, p2: score_distance_p1p2(p1, p2, params['ka'], params['coop'])
d_mecs = [distance_metric1, distance_metric2]
plot_positions = [(0,1),(1,0)]
exes = np.linspace(0,15,500)
whys = [distance_metric2([0,0],[0,x]) for x in exes]

grid_a = generate_grid(30,30,4,3)
grid_a = translate(grid_a, 30, 30)
total = peturb_constant(grid_a, 0.55, 0)

neighbs = get_neighbors(total, params['neighbor_search_dist'])

fig, ax = plt.subplots(3, 2, figsize=(8,12))

## subfig 1
ax[0][0].scatter(total[:,0], total[:,1], color='gray', edgecolors='black')
ax[0][0].scatter(total[poi_1,0], total[poi_1,1], color='red', edgecolors='black')

ax[0][0].set_aspect('equal')
ax[0][0].set(xlabel='Absolute x', ylabel='Absolute y')
ax[0][0].set_title(f'A: Select parent point')

## subfig 2

c1 = plt.Circle((total[poi_1,0], total[poi_1,1]), params['neighbor_search_dist'], color='red', linewidth=2, fill=False)
ax[0][1].add_patch(c1)

ax[0][1].scatter(total[:,0], total[:,1], color='gray', edgecolors='black')

n1 = neighbs[poi_1]
for n in n1:
    ax[0][1].scatter(total[n, 0], total[n, 1], color='salmon', edgecolors='black')


ax[0][1].scatter(total[poi_1,0], total[poi_1,1], color='red', edgecolors='black')

ax[0][1].set_aspect('equal')
ax[0][1].set(xlabel='Absolute x', ylabel='Absolute y')
ax[0][1].set_title(f'B: Calculate parent neighborhood')

## subfig 3

neighb_index = neighbs[poi_1][neighb_num]

c1 = plt.Circle((total[poi_1,0], total[poi_1,1]), params['neighbor_search_dist'], color='red', linewidth=2, fill=False)
ax[1][0].add_patch(c1)
c2 = plt.Circle((total[neighb_index,0], total[neighb_index,1]), params['neighbor_search_dist'], color='blue', linewidth=2, fill=False)
ax[1][0].add_patch(c2)

ax[1][0].scatter(total[:,0], total[:,1], color='gray', edgecolors='black')

n1 = neighbs[poi_1]
for n in n1:
    ax[1][0].scatter(total[n, 0], total[n, 1], color='salmon', edgecolors='black')
n2 = neighbs[neighb_index]
for n in n2:
    ax[1][0].scatter(total[n, 0], total[n, 1], color='cornflowerblue', edgecolors='black')
common = list(set(n1).intersection(n2))
for n in common:
    ax[1][0].scatter(total[n, 0], total[n, 1], color='mediumpurple', edgecolors='black')

ax[1][0].scatter(total[poi_1,0], total[poi_1,1], color='red', edgecolors='black')
ax[1][0].scatter(total[neighb_index,0], total[neighb_index,1], color='blue', edgecolors='black')

ax[1][0].set_aspect('equal')
ax[1][0].set(xlabel='Absolute x', ylabel='Absolute y')
ax[1][0].set_title(f'C: Select neighbor and \ncalculate neighbor neighborhood')

## subfig 4

neighbors = compose_neighborhoods(total, params['neighbor_search_dist'])
poi_2 = neighb_index

set_1 = neighbors[poi_1]['coords']
set_2 = neighbors[poi_2]['coords']
ax[1][1].scatter(set_1[:,0], set_1[:,1], color='salmon', edgecolors='black')
ax[1][1].scatter(set_2[:,0], set_2[:,1], color='cornflowerblue', edgecolors='black')


ax[1][1].set_aspect('equal')
ax[1][1].set(xlabel='Relative x', ylabel='Relative y')
ax[1][1].set_title(f'D: Calculate position of neighborhoods \nrelative to parents and superimpose')

## subfig 5

neighbors = compose_neighborhoods(total, params['neighbor_search_dist'])
poi_2 = neighb_index

set_1 = neighbors[poi_1]['coords']
set_2 = neighbors[poi_2]['coords']
ax[2][0].scatter(set_1[:,0], set_1[:,1], color='salmon', edgecolors='black')
ax[2][0].scatter(set_2[:,0], set_2[:,1], color='cornflowerblue', edgecolors='black')

swapped = False
s1, s2 = set_1, set_2
if len(set_1) > len(set_2):
    s1, s2 = set_2, set_1
    swapped = True

distance_metric = lambda p1, p2: score_distance_p1p2(p1, p2, params['ka'], params['coop'])

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

for simplex in hull.simplices:
    ax[2][0].plot(assigned_coords[simplex, 0], assigned_coords[simplex, 1], 'k-', color='green', lw=2)

scores = []
dees = []
for p in range(min([len(s1), len(s2)])):

    exes, whys = [s1[p, 0], s2[assignment[p], 0]], [s1[p, 1], s2[assignment[p], 1]]
    pt1 = exes[0], whys[0]
    pt2 = exes[1], whys[1]

    d = distance(pt1, pt2)
    dees.append(d)
    score = score_distance(d, params['ka'], params['coop'])
    scores.append(score)
    ax[2][0].plot(exes, whys, 'k')
    txt = ax[2][0].text(pt2[0]+.1, pt2[1]+.1, round(score,2))
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])

ax[2][0].set_aspect('equal')
ax[2][0].set(xlabel='Relative x', ylabel='Relative y')
ax[2][0].set_title(f'E: Assign correspondence and score \n(Mean: {round(np.mean(scores),2)})')

## subfig 6

neighb_index2 = neighbs[poi_1][neighb_num2]

c1 = plt.Circle((total[poi_1,0], total[poi_1,1]), params['neighbor_search_dist'], color='red', linewidth=2, fill=False)
ax[2][1].add_patch(c1)
c2 = plt.Circle((total[neighb_index2,0], total[neighb_index2,1]), params['neighbor_search_dist'], color='gold', linewidth=2, fill=False)
ax[2][1].add_patch(c2)

ax[2][1].scatter(total[:,0], total[:,1], color='gray', edgecolors='black')

n1 = neighbs[poi_1]
for n in n1:
    ax[2][1].scatter(total[n, 0], total[n, 1], color='salmon', edgecolors='black')
n2 = neighbs[neighb_index2]
for n in n2:
    ax[2][1].scatter(total[n, 0], total[n, 1], color='navajowhite', edgecolors='black')
common = list(set(n1).intersection(n2))
for n in common:
    ax[2][1].scatter(total[n, 0], total[n, 1], color='peru', edgecolors='black')

ax[2][1].scatter(total[poi_1,0], total[poi_1,1], color='red', edgecolors='black')
ax[2][1].scatter(total[neighb_index,0], total[neighb_index,1], color='blue', edgecolors='black')
ax[2][1].scatter(total[neighb_index2,0], total[neighb_index2,1], color='gold', edgecolors='black')

pt = total[neighb_index,0], total[neighb_index,1]
txt = ax[2][1].text(pt[0], pt[1], round(np.mean(scores), 2), ha='right')
txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])

ax[2][1].set_aspect('equal')
ax[2][1].set(xlabel='Absolute x', ylabel='Absolute y')
ax[2][1].set_title(f'F: Store score and repeat \nfor other neighbors')


plt.tight_layout()
if write:
    plt.close()
    fig.savefig(outloc)