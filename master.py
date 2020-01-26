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

trees_1 = False
trees_2 = False
trees_3 = True

save = True
sensitivity = False

if use_dhm:
    if trees_1:
        #outname = r'C:\Users\rj3h\Documents\programming_projects\point-disorder\point_disorder_paper\figures\trees_1.png'
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

        #outname = r'C:\Users\rj3h\Documents\programming_projects\point-disorder\point_disorder_paper\figures\trees_2.png'
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

    elif trees_3:
        #outname = r'C:\Users\rj3h\Documents\programming_projects\point-disorder\point_disorder_paper\figures\trees_3.png'
        outname = r'C:\Users\rsjon_000\Documents\point-disorder\point_disorder_paper\figures\trees_3.png'
        im_xlim = (2200, 5300)
        im_ylim = (1800, 3900)
        im_path = r'F:\entropy_veg\lidar\atwell\las_products\dhm.tif'

        punishment = 1
        punish_out_of_hull = False
        euc = False
        reorientation = None

        """
        max_sigma = 20
        min_sigma = 7
        num_sigma = 10
        threshold = 5 # min tree height
        overlap = 0.4
        thresh = 0.7 # disorder cutoff
        
        neighbor_search_dist = 40
        ka = 5
        coop = 3
        max_sigma = 20
        min_sigma = 7
        num_sigma = 10
        threshold = 5
        overlap = 0.6
        thresh = 0.75
        """

        neighbor_search_dist = 25
        ka = 2
        coop = 3

        max_sigma = 20
        min_sigma = 7
        num_sigma = 10
        threshold = 5
        overlap = 0.6
        thresh = 0.75

        plot_planted = True

        aoi = np.array([
            np.array([
                np.array([288, 327]),
                np.array([641, 140]),
                np.array([889, 121]),
                np.array([915, 308]),
                np.array([832, 296]),
                np.array([743, 226]),
                np.array([686, 200]),
                np.array([670, 251]),
                np.array([482, 340]),
                np.array([320, 330])
            ]),
            np.array([
                np.array([385, 463]),
                np.array([283, 1270]),
                np.array([235, 1643]),
                np.array([130, 1979]),
                np.array([55, 2515]),
                np.array([155, 2571]),
                np.array([690, 1885]),
                np.array([995, 926]),
                np.array([952, 689]),
                np.array([827, 826]),
                np.array([821, 939]),
                np.array([773, 1027]),
                np.array([724, 1001]),
                np.array([707, 866]),
                np.array([662, 870]),
                np.array([659, 1007]),
                np.array([671, 1431]),
                np.array([652, 1543]),
                np.array([472, 1854]),
                np.array([416, 1879]),
                np.array([310, 1836]),
                np.array([578, 1082]),
                np.array([628, 1051]),
                np.array([647, 789]),
                np.array([647, 353])
            ]),
            np.array([
                np.array([105, 2751]),
                np.array([167, 2770]),
                np.array([250, 2674]),
                np.array([401, 2662]),
                np.array([634, 2633]),
                np.array([786, 2409]),
                np.array([653, 2347]),
                np.array([834, 2149]),
                np.array([887, 2026]),
                np.array([1052, 1593]),
                np.array([1027, 1499]),
                np.array([1064, 1356]),
                np.array([1214, 889]),
                np.array([1263, 789]),
                np.array([1269, 627]),
                np.array([1045, 702]),
                np.array([1064, 876]),
                np.array([758, 1779]),
                np.array([696, 1948]),
                np.array([354, 2421])
            ]),
            np.array([
                np.array([1550, 1099]),
                np.array([1351, 1376]),
                np.array([1343, 1698]),
                np.array([1298, 1918]),
                np.array([1367, 1971]),
                np.array([1502, 1999]),
                np.array([1958, 1987]),
                np.array([1938, 1775]),
                np.array([1815, 1531]),
                np.array([1779, 1539]),
                np.array([1754, 1352]),
                np.array([1669, 1193])
            ]),
            np.array([
                np.array([1545, 854]),
                np.array([1429, 1063]),
                np.array([1582, 1075]),
                np.array([1758, 1162]),
                np.array([1612, 852])
            ]),
            np.array([
                np.array([1549, 586]),
                np.array([1786, 999]),
                np.array([1826, 1037]),
                np.array([1902, 1392]),
                np.array([1942, 1418]),
                np.array([1965, 1354]),
                np.array([1939, 967]),
                np.array([1923, 614]),
                np.array([1878, 459])
            ])
        ]
        )

        aoi = aoi * .3048
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

    if trees_3:
        pts = pts * .3048
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
    if trees_1:
        pts_out = r'C:\Users\rsjon_000\Documents\point-disorder\point_disorder_paper\figures\pts_as_fxn_of_radius1.png'
        sens_out = r'C:\Users\rsjon_000\Documents\point-disorder\point_disorder_paper\figures\sensitivity1.csv'
        radii = np.arange(30, 160, 10)
        kas = np.arange(1, 22, 2)
    elif trees_3:
        pts_out = r'C:\Users\rsjon_000\Documents\point-disorder\point_disorder_paper\figures\pts_as_fxn_of_radius3.png'
        sens_out = r'C:\Users\rsjon_000\Documents\point-disorder\point_disorder_paper\figures\sensitivity3.csv'
        radii = np.arange(5, 45, 5)
        kas = np.arange(0.25, 5, 0.25)
    else:
        raise Exception('Improper parameterization: must be trees_1 or trees_3')



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
        print(f'\n\nSENSITIVITY: {i+1} of {len(r_k)}\n'
              f'RADIUS: {r}, Km: {k}\n\n')

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

    fig, ax = plt.subplots(2, 1, figsize=(16,8))
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
        fig, ax = plt.subplots(1, 2, figsize=(32, 24))
        if trees_3:
            ax[0].imshow(sub_image_gray, cmap='gray', extent=(0, (im_ylim[1]-im_ylim[0])*.3048, (im_xlim[1]-im_xlim[0])*.3048, 0))
            ax[1].imshow(sub_image_gray, cmap='gray', extent=(0, (im_ylim[1]-im_ylim[0])*.3048, (im_xlim[1]-im_xlim[0])*.3048, 0))
        else:
            ax[0].imshow(sub_image_gray, cmap='gray')
            ax[1].imshow(sub_image_gray, cmap='gray')
        ax[1].set_title(f'r={neighbor_search_dist}, km={ka}, coop={coop}\n'
                        f'punishment={punishment}, punish_out_of_hull={punish_out_of_hull}\n'
                        f'euc={euc}, reorientation={reorientation}')
        ax[0].set_title(f'Threshold = {thresh}')
        # plt.tight_layout()

        line1 = Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="red", markeredgecolor='black')
        line2 = Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="green", markeredgecolor='black')
        line3 = Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="dodgerblue", markeredgecolor='black')
        lines = [line1, line2, line3]
        line_names = ['Above threshold ("disordered")', 'Below threshold ("ordered")', 'Insufficient neighborhood']
        if plot_planted:
            lines.append(Line2D(range(1), range(1), color="purple"))
            line_names.append('Planted area')

        ax[0].legend(lines, line_names, numpoints=1, loc=1, prop={'size': 12})

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
            rad = 5
            if trees_3:
                rad = rad * .3048
            c = plt.Circle((x, y), radius=rad, color=col, linewidth=1, fill=True, edgecolor='black')
            ax[1].add_patch(c)
            c = plt.Circle((x, y), radius=rad, color=threshcol, linewidth=1, fill=True, edgecolor='black')
            ax[0].add_patch(c)
            # ax.annotate(i, (x, y))
        ax[0].set_aspect('equal')
        ax[1].set_aspect('equal')

        if plot_planted:
            for shape in aoi:
                shape = np.vstack([shape,shape[0]])
                wid = 2
                if trees_3:
                    wid = 3
                ax[0].plot(shape[:,0], shape[:,1], color='purple', linewidth=wid)
                ax[1].plot(shape[:,0], shape[:,1], color='purple', linewidth=wid)

        cbar = fig.colorbar(sm, ax=ax, fraction=0.02, use_gridspec=True)
        cbar.ax.set_title('IoD')

    else:
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.scatter(pts[:, 0], pts[:, 1], c=scores, cmap=color_map, vmin=0, vmax=1, edgecolors='black')
        # for i,(x,y) in enumerate(pts):
        #    ax.annotate(i, (x, y))
        ax.set_title(f'r={neighbor_search_dist}, ka={ka}, coop={coop}\n'
                     f'punishment={punishment}, punish_out_of_hull={punish_out_of_hull}\n'
                     f'euc={euc}, reorientation={reorientation}')
        ax.set_aspect('equal')
        plt.tight_layout()

        cbar = fig.colorbar(sm, ax=ax, use_gridspec=True)
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
