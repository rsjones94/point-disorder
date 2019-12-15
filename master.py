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

use_dhm = False

if use_dhm:
    im_xlim = (2000, 5000)
    im_ylim = (0, 3000)
    im_path = r'F:\entropy_veg\lidar\las_products\USGS_LPC_TN_27County_blk2_2015_2276581SE_LAS_2017\USGS_LPC_TN_27County_blk2_2015_2276581SE_LAS_2017_dhm.tif'

    neighbor_search_dist = 100
    ka = 10
    coop = 4
    punishment = 1
    punish_out_of_hull = False
else:
    neighbor_search_dist = 10
    ka = 2
    coop = 5
    punishment = 1
    punish_out_of_hull = False

###############

exes = np.linspace(0,100,1100)
whys = [score_distance(x,ka,coop) for x in exes]
plt.figure()
plt.plot(exes,whys)
plt.show()

"""
response = input("How's the response curve look?")
if response not in ['good', 'Good', True, 'True', 1, 'fine', 'Fine']:
    raise Exception
"""

if use_dhm:
    image_gray = io.imread(im_path)
    image_gray[image_gray > 500] = 0  # remove weird tall anomalies
    image_gray[image_gray < 3] = 0  # remove anything under 3m (noise)

    sub_image_gray = image_gray[im_xlim[0]:im_xlim[1], im_ylim[0]:im_ylim[1]]

    print('Extracting points')
    pts = extract_crowns_from_dhm(sub_image_gray)
    d = [(x**2 + y**2)**0.5 for x, y in pts]
else:
    grid_base = generate_grid(80, 80, 6, 1.5)

    # gridA = rotate(grid_base, 45)
    gridA = grid_base

    gridB = rotate(grid_base, 90)
    gridB = translate(gridB, 5, 0)

    # pts = np.append(gridA, gridB, 0)
    pts = gridA
    pts = np.array([[pt[0] + 1*math.sin(10*pt[1]),
                    pt[1]] for i,pt in enumerate(pts)])

    d = [(x**2 + y**2)**0.5 for x, y in pts]
    intensity = [di / 40 for di in d]

    """
    pts = np.array([[np.random.normal(pt[0], intensity[i]),
                     np.random.normal(pt[1], intensity[i])] for i,pt in enumerate(pts)])
    """

    pt_copy = []
    for i,(pt,di) in enumerate(zip(pts,d)):
        if di > 50:
            adder = [np.random.normal(pt[0], 2),
                     np.random.normal(pt[1], 2)]
        else:
            adder = pt
        pt_copy.append(adder)
    pts = np.array(pt_copy)


    #pts = rotate(pts, 45)
    # pts = np.random.normal(pts,intensity)
    """
    np.random.shuffle(pts)
    keep = int(len(pts)*0.9)
    ptc = []
    new_d = []
    for i in range(keep):
        ptc.append(pts[i])
        new_d.append(d[i])
    pts = np.array(ptc)
    d = new_d
    """



scores, neighborhoods, scatter_key, score_key = point_disorder_index(pts[:, 0:2],
                                                                     neighbor_search_dist,
                                                                     ka=ka,
                                                                     coop=coop,
                                                                     punishment=punishment,
                                                                     punish_out_of_hull=punish_out_of_hull)

print('Drawing')
fig, ax = plt.subplots(1, 1)
color_map = cm.get_cmap('RdYlGn_r')
if use_dhm:
    ax.imshow(sub_image_gray, cmap='gray')
    for i, tree in enumerate(pts):
        x, y, r = tree
        raw_col = scores[i]
        col = color_map(scores[i])
        if np.isnan(raw_col):
            col = 'dodgerblue'
        c = plt.Circle((x, y), 5, color=col, linewidth=1, fill=True)
        ax.add_patch(c)
        #ax.annotate(i, (x, y))
else:
    ax.scatter(pts[:, 0], pts[:, 1], c=scores, cmap=color_map, vmin=0, vmax=1, edgecolors='black')
ax.set_aspect('equal')
plt.show()
fig.savefig('F:\entropy_veg\scored_synthetic_wall_wavelines.png')

"""
ptn = 1319
for neighbor in neighborhoods[ptn]['neighbors']:
    compare_scatters(neighborhoods[ptn]['coords'],neighborhoods[neighbor]['coords'],'True')
    
if not use_dhm:
    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(pts[:, 0], pts[:, 1], c=scores, cmap=color_map, vmin=0, vmax=1, edgecolors='black')
    ax[0].set_aspect('equal')
    ax[1].scatter(d, scores)
    plt.show()
"""
