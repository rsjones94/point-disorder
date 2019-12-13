from skimage import io
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from tree_extraction import extract_crowns_from_dhm
from neighborhood_funcs import *
from comparison import BidirectionalDict

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
"""

neighbor_search_dist = 100
im_xlim = (2500, 4000)
im_ylim = (500, 2000)
im_path = r'F:\entropy_veg\lidar\las_products\USGS_LPC_TN_27County_blk2_2015_2276581SE_LAS_2017\USGS_LPC_TN_27County_blk2_2015_2276581SE_LAS_2017_dhm.tif'

###############

image_gray = io.imread(im_path)
image_gray[image_gray > 500] = 0  # remove weird tall anomalies
image_gray[image_gray < 3] = 0  # remove anything under 3m (noise)

sub_image_gray = image_gray[im_xlim[0]:im_xlim[1], im_ylim[0]:im_ylim[1]]

print('Extracting points')
tree_pts = extract_crowns_from_dhm(sub_image_gray)
print('Composing neighborhoods')
neighborhoods = compose_neighborhoods(tree_pts[:, 0:2], neighbor_search_dist)
print('Generating comparison keys and scoring')
scatter_key = generate_scatter_key(neighborhoods)
score_key = generate_score_key(scatter_key, neighbor_search_dist / 10, coop=3)
scores = score_points(neighborhoods, score_key)

fig, ax = plt.subplots(1, 1)
ax.imshow(sub_image_gray, cmap='gray')
print('Drawing')
color_map = cm.get_cmap('plasma')
for i, tree in enumerate(tree_pts):
    x, y, r = tree
    raw_col = scores[i]
    col = color_map(scores[i])
    if np.isnan(raw_col):
        col = 'dodgerblue'
    c = plt.Circle((x, y), r, color=col, linewidth=1, fill=True)
    ax.add_patch(c)
ax.set_axis_off()
plt.show()
fig.savefig('F:\entropy_veg\scored.png')

fig, ax = plt.subplots(1, 1)
ax.scatter(range(len(scores)), scores)
plt.show()
fig.savefig('F:\entropy_veg\scatter.png')
