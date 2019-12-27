from skimage import io
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from tree_extraction import extract_crowns_from_dhm
from neighborhood_funcs import *
from pattern_generation import *
from comparison import BidirectionalDict

"""
What if we extended the idea of Haralick textures to vectors? We could build
a "GLCM" but for a point cloud instead of rasters. "directionality" to disorder
"""

use_dhm = False

if use_dhm:
    im_xlim = (2000, 5000)
    im_ylim = (0, 3000)
    im_path = r'F:\entropy_veg\lidar\las_products\USGS_LPC_TN_27County_blk2_2015_2276581SE_LAS_2017\USGS_LPC_TN_27County_blk2_2015_2276581SE_LAS_2017_dhm.tif'

    neighbor_search_dist = 50
    ka = 8
    coop = 4
    punishment = 1
    punish_out_of_hull = False
    euc = True
    reorientation = None
else:
    neighbor_search_dist = 15
    ka = 3
    coop = 5
    punishment = 1
    punish_out_of_hull = False
    euc = False
    reorientation = 10e-3

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

    sub_image_gray = image_gray
    #sub_image_gray = image_gray[im_xlim[0]:im_xlim[1], im_ylim[0]:im_ylim[1]]

    print('Extracting points')
    pts = extract_crowns_from_dhm(sub_image_gray)
else:
    grid_base = generate_grid(100, 100, 6, 3)

    # gridA = rotate(grid_base, 45)
    gridA = grid_base

    gridB = rotate(grid_base, 30)
    #gridB = translate(gridB, 5, 0)

    #pts = np.append(gridA, gridB, 0)
    pts = gridA
    pts = np.array([[pt[0] + 15*math.sin(0.05*pt[1]),
                    pt[1]] for i,pt in enumerate(pts)])
    pts = rotate(pts, 90)
    pts = np.array([[pt[0] + 15*math.sin(0.05*pt[1]),
                    pt[1]] for i,pt in enumerate(pts)])

    d = [(x**2 + y**2)**0.5 for x, y in pts]
    intensity = [di / 40 for di in d]

    """
    pts = np.array([[np.random.normal(pt[0], intensity[i]),
                     np.random.normal(pt[1], intensity[i])] for i,pt in enumerate(pts)])
    """


    #pts = peturb_constant(pts, 5, 70)
    pts = peturb_gradational(pts, 0.05, 50)


    #pts = rotate(pts, 45)
    # pts = np.random.normal(pts,intensity)

    #pts = decimate_grid(pts, 0.7)




scores, neighborhoods, scatter_key, score_key = point_disorder_index(pts[:, 0:2],
                                                                     neighbor_search_dist,
                                                                     ka=ka,
                                                                     coop=coop,
                                                                     punishment=punishment,
                                                                     punish_out_of_hull=punish_out_of_hull,
                                                                     euclidean=euc,
                                                                     reorient_tol=reorientation)

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
    #for i,(x,y) in enumerate(pts):
    #    ax.annotate(i, (x, y))
ax.set_title(f'r={neighbor_search_dist}, ka={ka}, coop={coop}\n'
         f'punishment={punishment}, punish_out_of_hull={punish_out_of_hull}\n'
         f'euc={euc}, reorientation={reorientation}')
ax.set_aspect('equal')
plt.show()
#fig.savefig('F:\entropy_veg\scored_w_metric+pr.png')

"""
ptn = 346
distance_metric = lambda p1, p2: score_distance_p1p2(p1, p2, ka, coop)
for neighbor in neighborhoods[ptn]['neighbors']:
    compare_scatters(neighborhoods[ptn]['coords'],neighborhoods[neighbor]['coords'], plot=True, distance_metric=distance_metric, reorient_tol=0.01)
    
if not use_dhm:
    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(pts[:, 0], pts[:, 1], c=scores, cmap=color_map, vmin=0, vmax=1, edgecolors='black')
    ax[0].set_aspect('equal')
    ax[1].scatter(d, scores)
    plt.show()
"""