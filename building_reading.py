from osgeo import ogr
from skimage import io
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib.cm import ScalarMappable
from numpy import linspace
import geopandas as gpd

from tree_extraction import extract_crowns_from_dhm
from neighborhood_funcs import *
from pattern_generation import *


file = r"F:\entropy_veg\building_footprints\tn_footprint\koordinates_data\footprints_sylvan_hillswest_lockeland.shp"
out_file = r"F:\entropy_veg\building_footprints\tn_footprint\koordinates_data\footprints_sylvan_hillswest_lockeland_SCORED.shp"
df = gpd.read_file(file)


xmin = -1110441 - 5000
xmax = xmin + 8000 + 5000
ymin = 4148855 - 5000
ymax = ymin + 8000 + 5000

filt = df

pars = {'neighbor_search_dist': 100,
               'ka': 10,
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

scores, neighborhoods, scatter_key, score_key = point_disorder_index(pts[:, 0:2],
                                                                     neighbor_search_dist,
                                                                     ka=ka,
                                                                     coop=coop,
                                                                     punishment=punishment,
                                                                     punish_out_of_hull=punish_out_of_hull,
                                                                     euclidean=euc,
                                                                     reorient_tol=reorientation)

fig, ax = plt.subplots(1, 1, figsize=(16, 8))
color_map = cm.get_cmap('RdYlGn_r')
im1 = ax.scatter(pts[:, 0], pts[:, 1], c=scores, cmap=color_map, vmin=0, vmax=1, edgecolors='black')
# for i,(x,y) in enumerate(pts):
#    ax.annotate(i, (x, y))
ax.set_title(f'r={neighbor_search_dist}, ka={ka}, coop={coop}\n'
                f'punishment={punishment}, punish_out_of_hull={punish_out_of_hull}\n'
                f'euc={euc}, reorientation={reorientation}')
ax.set_aspect('equal')

plt.tight_layout()

norm = plt.Normalize(0, 1)
sm = ScalarMappable(norm=norm, cmap=color_map)
cbar = fig.colorbar(sm, ax=ax)
cbar.ax.set_title('IoD')

fig.savefig(f'C:\\Users\\rsjon_000\\Documents\\point-disorder\\point_disorder_paper\\figures\\{name}.png')

filt['iod'] = scores
filt.to_file(out_file)