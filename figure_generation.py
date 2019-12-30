from skimage import io
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib.cm import ScalarMappable
from numpy import linspace

from tree_extraction import extract_crowns_from_dhm
from neighborhood_funcs import *
from pattern_generation import *

pet_radius = 40
pet_gradient = 0.05

square_grid = generate_grid(100, 100, 5, 5)
large_square_grid = generate_grid(150, 150, 5, 5)
rect_grid = generate_grid(100, 100, 6, 3)
stamp = np.array([np.array([1, 0]),
                  np.array([0, 2]),
                  np.array([0, 1]),
                  np.array([0, -1]),
                  np.array([0, 0]),
                  np.array([1, 2])]
                 ) * 2
grids = []
names = []
params = []


working = stamp_pattern(stamp, 8, 8, 100, 100, flip=False, rotate_by=2)
working = peturb_gradational(working, pet_gradient, pet_radius)
grids.append(working)
names.append('stamp_rot')
params.append({'neighbor_search_dist': 20,
               'ka': 3,
               'coop': 5,
               'punishment': 1,
               'punish_out_of_hull': False,
               'euc': False,
               'reorientation': 10e-3})

working = stamp_pattern(stamp, 8, 8, 100, 100, flip=False, rotate_by=0)
working = peturb_gradational(working, pet_gradient, pet_radius)
grids.append(working)
names.append('stamp')
params.append({'neighbor_search_dist': 20,
               'ka': 3,
               'coop': 5,
               'punishment': 1,
               'punish_out_of_hull': False,
               'euc': False,
               'reorientation': None})

working = square_grid
working = peturb_gradational(working, pet_gradient, pet_radius)
grids.append(working)
names.append('square_grid')
params.append({'neighbor_search_dist': 15,
               'ka': 3,
               'coop': 5,
               'punishment': 1,
               'punish_out_of_hull': False,
               'euc': False,
               'reorientation': 10e-3})

working = rect_grid
working = peturb_gradational(working, pet_gradient, pet_radius)
grids.append(working)
names.append('rect_grid')
params.append({'neighbor_search_dist': 15,
               'ka': 3,
               'coop': 5,
               'punishment': 1,
               'punish_out_of_hull': False,
               'euc': False,
               'reorientation': 10e-3})

working = square_grid
working = decimate_grid(working, 0.85)
working = peturb_gradational(working, pet_gradient, pet_radius)
grids.append(working)
names.append('square_grid_decimated')
params.append({'neighbor_search_dist': 15,
               'ka': 3,
               'coop': 5,
               'punishment': 1,
               'punish_out_of_hull': False,
               'euc': False,
               'reorientation': 10e-3})

working = concentric_circles(12, 0.5, 8)
working = peturb_gradational(working, pet_gradient, pet_radius)
grids.append(working)
names.append('circles')
params.append({'neighbor_search_dist': 12,
               'ka': 3,
               'coop': 5,
               'punishment': 1,
               'punish_out_of_hull': False,
               'euc': False,
               'reorientation': 10e-3})

working = rect_grid
working = wavify_grid(working, 5, 75)
working = peturb_gradational(working, pet_gradient, pet_radius)
grids.append(working)
names.append('wavy_grid')
params.append({'neighbor_search_dist': 15,
               'ka': 3,
               'coop': 5,
               'punishment': 1,
               'punish_out_of_hull': False,
               'euc': False,
               'reorientation': 10e-3})

working = rect_grid
working = wavify_grid(working, 5, 75)
working = rotate(working, 45)
working = wavify_grid(working, 10, 75)
working = peturb_gradational(working, pet_gradient, pet_radius)
grids.append(working)
names.append('wavy_grid_complex')
params.append({'neighbor_search_dist': 15,
               'ka': 3,
               'coop': 5,
               'punishment': 1,
               'punish_out_of_hull': False,
               'euc': False,
               'reorientation': 10e-3})

working = large_square_grid
working2 = rotate(large_square_grid, 45)
working = np.append(working, working2, 0)
working = peturb_gradational(working, pet_gradient, pet_radius)
grids.append(working)
names.append('overlapping_grid_rot')
params.append({'neighbor_search_dist': 15,
               'ka': 3,
               'coop': 5,
               'punishment': 1,
               'punish_out_of_hull': False,
               'euc': False,
               'reorientation': 10e-3})

working = rect_grid
working2 = translate(rect_grid, 3, 3)
working = np.append(working, working2, 0)
working = peturb_gradational(working, pet_gradient, pet_radius)
grids.append(working)
names.append('overlapping_grid_alternating')
params.append({'neighbor_search_dist': 15,
               'ka': 3,
               'coop': 5,
               'punishment': 1,
               'punish_out_of_hull': False,
               'euc': False,
               'reorientation': 10e-3})

working = rect_grid
working2 = translate(rect_grid, 6, 2)
working3 = translate(rect_grid, 6, 4)
working = np.append(working, working2, 0)
working = np.append(working, working3, 0)
working = peturb_gradational(working, pet_gradient, pet_radius)
grids.append(working)
names.append('overlapping_grid_triple_trans')
params.append({'neighbor_search_dist': 15,
               'ka': 3,
               'coop': 5,
               'punishment': 1,
               'punish_out_of_hull': False,
               'euc': False,
               'reorientation': 10e-3})

working = large_square_grid
working2 = rotate(large_square_grid, 45)
working2 = translate(working2, 2.5, 2.5)
working = np.append(working, working2, 0)
working = peturb_gradational(working, pet_gradient, pet_radius)
grids.append(working)
names.append('overlapping_grid_trans_rot')
params.append({'neighbor_search_dist': 15,
               'ka': 3,
               'coop': 5,
               'punishment': 1,
               'punish_out_of_hull': False,
               'euc': False,
               'reorientation': 10e-3})


for pts, name, pars in zip(grids, names, params):
    neighbor_search_dist = pars['neighbor_search_dist']
    ka = pars['ka']
    coop = pars['coop']
    punishment = pars['punishment']
    punish_out_of_hull = pars['punish_out_of_hull']
    euc = pars['euc']
    reorientation = pars['reorientation']

    scores, neighborhoods, scatter_key, score_key = point_disorder_index(pts[:, 0:2],
                                                                         neighbor_search_dist,
                                                                         ka=ka,
                                                                         coop=coop,
                                                                         punishment=punishment,
                                                                         punish_out_of_hull=punish_out_of_hull,
                                                                         euclidean=euc,
                                                                         reorient_tol=None)
    print('Drawing')
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    color_map = cm.get_cmap('RdYlGn_r')
    im1 = ax[0].scatter(pts[:, 0], pts[:, 1], c=scores, cmap=color_map, vmin=0, vmax=1, edgecolors='black')
    c1 = plt.Circle((0, 0), pet_radius, color='red', linewidth=2, fill=False)
    c2 = plt.Circle((0, 0), neighbor_search_dist, color='blue', linewidth=2, fill=False)
    ax[0].add_patch(c1)
    ax[0].add_patch(c2)
    # for i,(x,y) in enumerate(pts):
    #    ax.annotate(i, (x, y))
    ax[0].set_title(f'r={neighbor_search_dist}, ka={ka}, coop={coop}\n'
                    f'punishment={punishment}, punish_out_of_hull={punish_out_of_hull}\n'
                    f'euc={euc}, reorientation=None')
    ax[0].set_aspect('equal')

    scores, neighborhoods, scatter_key, score_key = point_disorder_index(pts[:, 0:2],
                                                                         neighbor_search_dist,
                                                                         ka=ka,
                                                                         coop=coop,
                                                                         punishment=punishment,
                                                                         punish_out_of_hull=punish_out_of_hull,
                                                                         euclidean=euc,
                                                                         reorient_tol=reorientation)
    im2 = ax[1].scatter(pts[:, 0], pts[:, 1], c=scores, cmap=color_map, vmin=0, vmax=1, edgecolors='black')
    c1 = plt.Circle((0, 0), pet_radius, color='red', linewidth=2, fill=False)
    c2 = plt.Circle((0, 0), neighbor_search_dist, color='blue', linewidth=2, fill=False)
    ax[1].add_patch(c1)
    ax[1].add_patch(c2)
    # for i,(x,y) in enumerate(pts):
    #    ax.annotate(i, (x, y))
    ax[1].set_title(f'r={neighbor_search_dist}, ka={ka}, coop={coop}\n'
                    f'punishment={punishment}, punish_out_of_hull={punish_out_of_hull}\n'
                    f'euc={euc}, reorientation={reorientation}')
    ax[1].set_aspect('equal')

    plt.tight_layout()

    norm = plt.Normalize(0, 1)
    sm = ScalarMappable(norm=norm, cmap=color_map)
    cbar = fig.colorbar(sm, ax=ax[:])
    cbar.ax.set_title('IoD')

    plt.close()
    fig.savefig(f'C:\\Users\\rsjon_000\\Documents\\point-disorder\\point_disorder_paper\\figures\\{name}.png')
