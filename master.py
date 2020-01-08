import itertools
import pandas as pd

from skimage import io
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.path import Path
from sklearn import metrics

from tree_extraction import extract_crowns_from_dhm
from neighborhood_funcs import *
from pattern_generation import *
from comparison import BidirectionalDict

"""
What if we extended the idea of Haralick textures to vectors? We could build
a "GLCM" but for a point cloud instead of rasters. "directionality" to disorder
"""

use_dhm = True

trees_1 = True
trees_2 = False

save = False
sensitivity = True

if use_dhm:
    if trees_1:
        outname = r'C:\Users\rsjon_000\Documents\point-disorder\point_disorder_paper\figures\trees_1.png'
        im_xlim = (1500, 5000)
        im_ylim = (0, 3000)
        im_path = r'F:\entropy_veg\lidar\las_products\USGS_LPC_TN_27County_blk2_2015_2276581SE_LAS_2017\USGS_LPC_TN_27County_blk2_2015_2276581SE_LAS_2017_dhm.tif'

        neighbor_search_dist = 70
        ka = 6
        coop = 5
        punishment = 1
        punish_out_of_hull = False
        euc = False
        reorientation = None

        plot_planted = True

        max_sigma = 30
        min_sigma = 6
        num_sigma = 10
        threshold = 2
        overlap = .05

        thresh = 0.8

        aoi = np.array([
            np.array([
                np.array([373, 1950]),
                np.array([236, 2090]),
                np.array([223, 2280]),
                np.array([308, 2400]),
                np.array([390, 2418]),
                np.array([602, 2234]),
                np.array([675, 2010]),
                np.array([806, 1871]),
                np.array([655, 1713])
            ]),
            np.array([
                np.array([828, 1557]),
                np.array([1123, 1886]),
                np.array([1056, 1986]),
                np.array([1636, 2497]),
                np.array([2050, 2490]),
                np.array([1571, 2071]),
                np.array([1189, 1709]),
                np.array([1087, 1705]),
                np.array([1073, 1572]),
                np.array([955, 1459])
            ])
        ]
        )

    elif trees_2:
        outname = r'C:\Users\rsjon_000\Documents\point-disorder\point_disorder_paper\figures\trees_2.png'
        im_xlim = (0, 3200)
        im_ylim = (4250, 7000)
        im_path = r'F:\entropy_veg\lidar\las_products\USGS_LPC_TN_27County_blk2_2015_2276581SE_LAS_2017\USGS_LPC_TN_27County_blk2_2015_2276581SE_LAS_2017_dhm.tif'

        neighbor_search_dist = 80
        ka = 15
        coop = 10
        punishment = 1
        punish_out_of_hull = False
        euc = False
        reorientation = None
        max_sigma = 50
        min_sigma = 10
        num_sigma = 10
        threshold = 5
        overlap = 0.5

        plot_planted = False

        thresh = 0.5
    else:
        raise Exception('No parameters')
else:
    neighbor_search_dist = 15
    ka = 3
    coop = 5
    punishment = 1
    punish_out_of_hull = False
    euc = False
    reorientation = 10e-3

###############

"""
exes = np.linspace(0,100,1100)
whys = [score_distance(x,ka,coop) for x in exes]
plt.figure()
plt.plot(exes,whys)
plt.show()
"""

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
    sub_image_gray = image_gray[im_xlim[0]:im_xlim[1], im_ylim[0]:im_ylim[1]]

    print('Extracting points')
    pts = extract_crowns_from_dhm(sub_image_gray,
                                  max_sigma=max_sigma,
                                  min_sigma=min_sigma,
                                  num_sigma=num_sigma,
                                  threshold=threshold,
                                  overlap=overlap)
else:
    grid_base = generate_grid(100, 100, 6, 3)

    # gridA = rotate(grid_base, 45)
    gridA = grid_base

    gridB = rotate(grid_base, 30)
    # gridB = translate(gridB, 5, 0)

    # pts = np.append(gridA, gridB, 0)
    pts = gridA
    pts = np.array([[pt[0] + 15 * math.sin(0.05 * pt[1]),
                     pt[1]] for i, pt in enumerate(pts)])
    pts = rotate(pts, 90)
    pts = np.array([[pt[0] + 15 * math.sin(0.05 * pt[1]),
                     pt[1]] for i, pt in enumerate(pts)])

    d = [(x ** 2 + y ** 2) ** 0.5 for x, y in pts]
    intensity = [di / 40 for di in d]

    """
    pts = np.array([[np.random.normal(pt[0], intensity[i]),
                     np.random.normal(pt[1], intensity[i])] for i,pt in enumerate(pts)])
    """

    # pts = peturb_constant(pts, 5, 70)
    pts = peturb_gradational(pts, 0.05, 50)

    # pts = rotate(pts, 45)
    # pts = np.random.normal(pts,intensity)

    # pts = decimate_grid(pts, 0.7)

if sensitivity:
    pts_out = r'C:\Users\rsjon_000\Documents\point-disorder\point_disorder_paper\figures\pts_as_fxn_of_radius.png'
    sens_out = r'C:\Users\rsjon_000\Documents\point-disorder\point_disorder_paper\figures\sensitivity.csv'

    radii = np.arange(30, 160, 10)
    kas = np.arange(1, 22, 2)

    r_k = [(a, b) for a in radii for b in kas]
    kappas = []

    is_in = []
    for shape in aoi:
        p = Path(shape)
        is_in.append(p.contains_points(pts[:,0:2]))
    is_in = np.vstack(is_in)
    is_in = np.array([any(is_in[:,i]) for i,_ in enumerate(is_in[0])])

    kappa_df = pd.DataFrame(index=radii, columns=kas)
    num_neighbors = pd.Series(index=radii)
    num_points = pd.Series(index=radii)

    for i,(r,k) in enumerate(r_k):
        print(f'\n\nSENSITIVITY: {i+1} of {len(r_k)}\n\n')

        scores, neighborhoods, scatter_key, score_key = point_disorder_index(pts[:, 0:2],
                                                                             r,
                                                                             ka=k,
                                                                             coop=coop,
                                                                             punishment=punishment,
                                                                             punish_out_of_hull=punish_out_of_hull,
                                                                             euclidean=euc,
                                                                             reorient_tol=reorientation)

        scores = np.array(scores)
        under_thresh = np.array([score <= thresh for score in scores])
        mask = ~np.isnan(scores)

        good_scores = scores[mask]
        good_thresh = under_thresh[mask]
        good_is_in = is_in[mask]

        kappa = metrics.cohen_kappa_score(good_is_in, good_thresh)
        kappas.append(kappa)

        kappa_df[k][r] = kappa

        neighbs = np.mean([len(pt['neighbors']) for pt in neighborhoods])
        num_neighbors[r] = neighbs

        num_points[r] = len(good_scores)

    fig, ax = plt.subplots(2, 1, figsize=(8,12))
    ax[0].plot(radii, num_points, marker='o')
    ax[0].set_xlim([0,max(radii)+5])
    ax[1].plot(radii, num_neighbors, marker='o')
    ax[1].set_xlim([0,max(radii)+5])

    ax[0].set_title(f'Number of points with at least one neighbor')
    ax[1].set_title(f'Average number of neighbors')

    ax[0].set_xlabel('Neighborhood radius (m)')
    ax[1].set_xlabel('Neighborhood radius (m)')

    ax[0].set_ylabel('Points')
    ax[1].set_ylabel('Neighbors')

    plt.tight_layout()
    fig.savefig(pts_out)

    kappa_df.to_csv(sens_out)

else:
    scores, neighborhoods, scatter_key, score_key = point_disorder_index(pts[:, 0:2],
                                                                         neighbor_search_dist,
                                                                         ka=ka,
                                                                         coop=coop,
                                                                         punishment=punishment,
                                                                         punish_out_of_hull=punish_out_of_hull,
                                                                         euclidean=euc,
                                                                         reorient_tol=reorientation)

    print('Drawing')
    color_map = cm.get_cmap('RdYlGn_r')
    norm = plt.Normalize(0, 1)
    sm = ScalarMappable(norm=norm, cmap=color_map)
    if use_dhm:
        fig, ax = plt.subplots(1, 2, figsize=(12, 8))

        ax[0].imshow(sub_image_gray, cmap='gray')
        ax[1].imshow(sub_image_gray, cmap='gray')
        ax[1].set_title(f'r={neighbor_search_dist}, ka={ka}, coop={coop}\n'
                        f'punishment={punishment}, punish_out_of_hull={punish_out_of_hull}\n'
                        f'euc={euc}, reorientation={reorientation}')
        ax[0].set_title(f'Threshold = {thresh}')
        # plt.tight_layout()

        line1 = Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="red")
        line2 = Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="green")
        line3 = Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="dodgerblue")
        lines = [line1, line2, line3]
        line_names = ['Above threshold ("disordered")', 'Below threshold ("ordered")', 'Insufficient neighborhood']
        if plot_planted:
            lines.append(Line2D(range(1), range(1), color="purple"))
            line_names.append('Planted area')

        ax[0].legend(lines, line_names, numpoints=1, loc=1, prop={'size': 5})

        for i, tree in enumerate(pts):
            x, y, r = tree
            raw_col = scores[i]
            col = color_map(scores[i])
            if raw_col > thresh:
                threshcol = 'red'
            else:
                threshcol = 'green'

            if np.isnan(raw_col):
                col = 'dodgerblue'
                threshcol = 'dodgerblue'
            c = plt.Circle((x, y), radius=5, color=col, linewidth=1, fill=True)
            ax[1].add_patch(c)
            c = plt.Circle((x, y), radius=5, color=threshcol, linewidth=1, fill=True)
            ax[0].add_patch(c)
            # ax.annotate(i, (x, y))
        ax[0].set_aspect('equal')
        ax[1].set_aspect('equal')

        if plot_planted:
            for shape in aoi:
                shape = np.vstack([shape,shape[0]])
                ax[0].plot(shape[:,0], shape[:,1], color='purple', linewidth=2)
                ax[1].plot(shape[:,0], shape[:,1], color='purple', linewidth=2)

        cbar = fig.colorbar(sm, ax=ax, fraction=0.02)
        cbar.ax.set_title('IoD')

    else:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.scatter(pts[:, 0], pts[:, 1], c=scores, cmap=color_map, vmin=0, vmax=1, edgecolors='black')
        # for i,(x,y) in enumerate(pts):
        #    ax.annotate(i, (x, y))
        ax.set_title(f'r={neighbor_search_dist}, ka={ka}, coop={coop}\n'
                     f'punishment={punishment}, punish_out_of_hull={punish_out_of_hull}\n'
                     f'euc={euc}, reorientation={reorientation}')
        ax.set_aspect('equal')
        plt.tight_layout()

        cbar = fig.colorbar(sm, ax=ax)
        cbar.ax.set_title('IoD')

    plt.show()
    if save:
        fig.savefig(outname)
        plt.close()

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

    print('Done')
