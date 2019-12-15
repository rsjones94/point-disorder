from skimage import io
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from tree_extraction import extract_crowns_from_dhm
from neighborhood_funcs import *
from comparison import BidirectionalDict
from pattern_generation import *

"""
Thoughts:
should allow some kind of scaling/rotation/translation/mirroring (affine), maybe even warping (non-affine)
    BUT be aware that allowing unlimited transformation would probably result in cheating by
    optimization heuristics
^ if above is paired with NOT penalizing unpaired points, then excess densification could cheat the measure
Perhaps penalize unapaired points IFF they are within the convex hull of the paired points (which would
discourage densification but reduce edge penalization)

ISSUE: Hungarian method minimizes total distance, NOT my scoring scheme
    e.g., sometimes it is obvious that an overlapping pattern exists,
    but is offset in a way s.t. the Hungarian matching
    produces a strange matching scheme

What if we extended the idea of Haralick textures to vectors? We could build
a "GLCM" but for a point cloud instead of rasters. "directionality" to disorder
"""
neighbor_search_dist = 20

use_dhm = False
im_xlim = (2500, 4000)
im_ylim = (500, 2000)
im_path = r'F:\entropy_veg\lidar\las_products\USGS_LPC_TN_27County_blk2_2015_2276581SE_LAS_2017\USGS_LPC_TN_27County_blk2_2015_2276581SE_LAS_2017_dhm.tif'

###############
fig, ax = plt.subplots(1, 1)
if use_dhm:
    image_gray = io.imread(im_path)
    image_gray[image_gray > 500] = 0  # remove weird tall anomalies
    image_gray[image_gray < 3] = 0  # remove anything under 3m (noise)

    sub_image_gray = image_gray[im_xlim[0]:im_xlim[1], im_ylim[0]:im_ylim[1]]

    print('Extracting points')
    pts = extract_crowns_from_dhm(sub_image_gray)
    ax.imshow(sub_image_gray, cmap='gray')
else:
    grid_base = generate_grid(80, 80, 3, 3)

    # gridA = rotate(grid_base, 45)
    gridA = grid_base

    gridB = rotate(grid_base, 90)
    gridB = translate(gridB, 5, 0)

    # pts = np.append(gridA, gridB, 0)
    pts = gridA
    intensity = [(x**2 + y**2)**0.5 / 40 for x, y in pts]
    pts = np.array([[pt[0] + np.random.normal(pt[0], intensity[i]),
                     pt[1] + np.random.normal(pt[1], intensity[i])] for i,pt in enumerate(pts)])
    #pts = rotate(pts, 45)
    # pts = np.random.normal(pts,intensity)

scores, neighborhoods, scatter_key, score_key = point_disorder_index(pts[:, 0:2],
                                                                     neighbor_search_dist,
                                                                     ka=2,
                                                                     coop=5,
                                                                     punishment=1,
                                                                     punish_out_of_hull=False)

print('Drawing')
color_map = cm.get_cmap('plasma')
if use_dhm:
    for i, tree in enumerate(pts):
        x, y, r = tree
        raw_col = scores[i]
        col = color_map(scores[i])
        if np.isnan(raw_col):
            col = 'dodgerblue'
        c = plt.Circle((x, y), r, color=col, vmin=0, vmax=1, linewidth=1, fill=True)
        ax.add_patch(c)
else:
    ax.scatter(pts[:, 0], pts[:, 1], c=scores, cmap=color_map, vmin=0, vmax=1, edgecolors='black')
ax.set_aspect('equal')
plt.show()
# fig.savefig('F:\entropy_veg\scored.png')

"""
ptn = 167
for neighbor in neighborhoods[ptn]['neighbors']:
    compare_scatters(neighborhoods[ptn]['coords'],neighborhoods[neighbor]['coords'],'True')
"""

d = [(x**2 + y**2)**0.5 for x, y in pts]
fig, ax = plt.subplots(1, 2)
ax[0].scatter(pts[:, 0], pts[:, 1], c=scores, cmap=color_map, vmin=0, vmax=1, edgecolors='black')
ax[0].set_aspect('equal')
ax[1].scatter(d, scores)
plt.show()

