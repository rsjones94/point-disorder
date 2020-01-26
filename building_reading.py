from osgeo import ogr
from skimage import io
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib.cm import ScalarMappable
from numpy import linspace
import geopandas as gpd
from sklearn import metrics
from matplotlib.lines import Line2D
import pylab as pl
import pandas as pd

from tree_extraction import extract_crowns_from_dhm
from neighborhood_funcs import *
from pattern_generation import *


#base_file = r"F:\entropy_veg\building_footprints\tn_footprint\koordinates_data\footprints_sylvan_hillswest_lockeland"
base_file = r"F:\entropy_veg\building_footprints\tn_footprint\koordinates_data\footprints_lockeland"
sensitivity = False
thresh = 0.7


file = base_file + '.shp'

df = gpd.read_file(file)
filt = df

# lockeland
#25, 7, 3 = .33
#23, 7, 3 = .36
#23, 8, 3 = .36
#21, 7, 3 = .38
#21, 7, 5 = .39


pars = {'neighbor_search_dist': 21,
               'ka': 6.5,
               'coop': 5,
               'punishment': 1,
               'punish_out_of_hull': False,
               'euc': False,
               'reorientation': None}

pts = np.array([np.array([x,y]) for x,y in zip(list(filt.geometry.centroid.x),list(filt.geometry.centroid.y))])

neighbor_search_dist = pars['neighbor_search_dist']
ka = pars['ka']
coop = pars['coop']
punishment = pars['punishment']
punish_out_of_hull = pars['punish_out_of_hull']
euc = pars['euc']
reorientation = pars['reorientation']

name = 'nashville'

if sensitivity:
    sens_out = r'C:\Users\rsjon_000\Documents\point-disorder\point_disorder_paper\figures\sensitivity_lockeland.csv'
    thresh_out = r'C:\Users\rsjon_000\Documents\point-disorder\point_disorder_paper\figures\thresh_lockeland.csv'
    radii = np.arange(15, 32, 2)
    kas = np.arange(3, 12, 0.5)

    r_k = [(a, b) for a in radii for b in kas]
    kappas = []

    kappa_df = pd.DataFrame(index=radii, columns=kas)
    thresh_df = pd.DataFrame(index=radii, columns=kas)

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


        filt['iod'] = scores

        maj = filt['DESCRIPT'] == 'Major'
        filt['is_major'] = maj

        filt['building_type'] = ['Major' if i else 'Other' for i in filt['is_major']]

        is_major_mean = filt[maj].iod.mean()
        other_mean = filt[~maj].iod.mean()

        #thresh = np.average([is_major_mean, other_mean], 0, [1, 3])
        classified_as_aux = filt.iod >= thresh

        kappa = metrics.cohen_kappa_score(classified_as_aux, ~maj)
        kappas.append(kappa)

        kappa_df[k][r] = kappa
        thresh_df[k][r] = thresh

    kappa_df.to_csv(sens_out)
    thresh_df.to_csv(thresh_out)

else:
    scores, neighborhoods, scatter_key, score_key = point_disorder_index(pts[:, 0:2],
                                                                     neighbor_search_dist,
                                                                     ka=ka,
                                                                     coop=coop,
                                                                     punishment=punishment,
                                                                     punish_out_of_hull=punish_out_of_hull,
                                                                     euclidean=euc,
                                                                     reorient_tol=reorientation)



    filt['iod'] = scores

    maj = filt['DESCRIPT'] == 'Major'
    filt['is_major'] = maj

    filt['building_type'] = ['Major' if i else 'Other' for i in filt['is_major']]

    ax = filt.boxplot('iod', 'building_type')
    plt.title(f'IoD of Building Types in Lockeland Springs, Nashville, TN\n'
              f'r={neighbor_search_dist} meters, km={ka} meters, n={coop} ')
    plt.suptitle('')
    ax.set_xlabel('Building Type')
    figout = f'C:\\Users\\rsjon_000\\Documents\\point-disorder\\point_disorder_paper\\figures\\nashvillebox.png'
    plt.savefig(figout)

    is_major_mean = filt[maj].iod.mean()
    other_mean = filt[~maj].iod.mean()

    #thresh = np.average([is_major_mean, other_mean], 0, [1, 3])
    classified_as_aux = [i > thresh if not np.isnan(i) else np.nan for i in filt.iod]

    aux_dict = {'classed_aux': classified_as_aux,
                'is_aux': ~maj}
    aux_df = pd.DataFrame(aux_dict)
    aux_df = aux_df.dropna()
    aux_df['classed_aux'] = pd.Series(aux_df['classed_aux'], dtype=bool)
    kappa = metrics.cohen_kappa_score(aux_df['classed_aux'], aux_df['is_aux'])

    print(f'KAPPA: {round(kappa,2)}')


    out_file = base_file + f'_SCORED_km{ka}coop{coop}r{neighbor_search_dist}kappa{round(kappa,2)}thresh{round(thresh,2)}.shp'
    filt.to_file(out_file)


    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    color_map = cm.get_cmap('RdYlGn_r')
    im1 = ax[1].scatter(pts[:, 0], pts[:, 1], c=scores, cmap=color_map, vmin=0, vmax=1, edgecolors='black')
    ax[1].set_title(f'r={neighbor_search_dist}, ka={ka}, coop={coop}\n'
                    f'punishment={punishment}, punish_out_of_hull={punish_out_of_hull}\n'
                    f'euc={euc}, reorientation={reorientation}')
    ax[1].set_aspect('equal')

    norm = plt.Normalize(0, 1)
    sm = ScalarMappable(norm=norm, cmap=color_map)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.02, use_gridspec=True)
    cbar.ax.set_title('IoD')

    assessment = [actual-classified for actual, classified in zip(aux_df['is_aux']*2, aux_df['classed_aux']*1)]
    assessment_map = {1:'TP',
                      -1:'FP',
                      2:'FN',
                      0:'TN'}
    assessment = [assessment_map[assess] for assess in assessment]
    assessment_cols_map = {'TP': 'royalblue',
                           'TN': 'paleturquoise',
                           'FP': 'peachpuff',
                           'FN': 'indianred'}
    col_vec = [assessment_cols_map[assess] for assess in assessment]

    pts_dict = {'x': pts[:, 0],
                'y': pts[:, 1],
                'scores': scores}
    pts_df = pd.DataFrame(pts_dict)
    pts_df = pts_df.dropna()

    im0 = ax[0].scatter(pts_df.x, pts_df.y, c=col_vec, edgecolors='black')

    line1 = Line2D(range(1), range(1), color="white", marker='o', markerfacecolor=assessment_cols_map['TP'], markeredgecolor='black')
    line2 = Line2D(range(1), range(1), color="white", marker='o', markerfacecolor=assessment_cols_map['TN'], markeredgecolor='black')
    line3 = Line2D(range(1), range(1), color="white", marker='o', markerfacecolor=assessment_cols_map['FP'], markeredgecolor='black')
    line4 = Line2D(range(1), range(1), color="white", marker='o', markerfacecolor=assessment_cols_map['FN'], markeredgecolor='black')
    lines = [line1, line2, line3, line4]
    line_names = ['True Positive', 'True Negative', 'False Positive', 'False Negative']

    ax[0].legend(lines, line_names, numpoints=1, loc=1, prop={'size': 12})
    ax[0].set_aspect('equal')

    ax[0].set_title(f'Threshold = {round(thresh,1)}\n'
                    f'κ = {round(kappa, 2)}')

    fig.savefig(f'C:\\Users\\rsjon_000\\Documents\\point-disorder\\point_disorder_paper\\figures\\{name}_kappa{round(kappa,2)}.png')