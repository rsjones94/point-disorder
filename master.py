from skimage import io
import matplotlib.pyplot as plt

from tree_extraction import extract_crowns_from_dhm
from neighborhood_funcs import compose_neighborhoods, compare_scatters, score_comparison

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

compare_1 = 50
compare_rel = 5

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
neighborhoods = compose_neighborhoods(tree_pts[:,0:2], neighbor_search_dist)
compare_2 = neighborhoods[compare_1]['neighbors'][compare_rel]

fig, ax = plt.subplots(1, 1)
ax.imshow(sub_image_gray)
print('Drawing')
for i,tree in enumerate(tree_pts):
    x, y, r = tree
    if i == compare_1:
        fill = True
        r *= 3
        color = 'blue'
        print(f'FILLING 1 AT {x,y,r}')
    elif i == compare_2:
        fill = True
        r *= 3
        color = 'orange'
        print(f'FILLING 2 AT {x,y,r}')
    else:
        fill = False
        color = 'red'
    c = plt.Circle((x, y), r, color=color, linewidth=1, fill=fill)
    ax.add_patch(c)

ax.set_axis_off()
plt.show()

s1 = neighborhoods[compare_1]['coords']
s2 = neighborhoods[compare_2]['coords']

comp = compare_scatters(s1, s2, True)
score = score_comparison(comp, ka=neighbor_search_dist/10, coop=3, punishment=1, punish_out_of_hull=False)
print(f'Score: {score}')
