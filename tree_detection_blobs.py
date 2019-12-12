from math import sqrt
import math
from math import atan2, degrees

from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
from skimage import io
import matplotlib.pyplot as plt
from scipy import stats
from scipy import spatial

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max

# ref : https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_blob.html
#image = data.hubble_deep_field()[0:500, 0:500]
#image_gray = rgb2gray(image)

neighbor_search_dist = 100

im_path = r'F:\entropy_veg\lidar\las_products\USGS_LPC_TN_27County_blk2_2015_2276581SE_LAS_2017\USGS_LPC_TN_27County_blk2_2015_2276581SE_LAS_2017_dhm.tif'
image_gray = io.imread(im_path)
image_gray[image_gray > 500] = 0
image_gray[image_gray < 3] = 0

image_gray = image_gray[2500:, 500:2000]
#image_gray = image_gray[500:2000, 4500:6000]
#image_gray = image_gray[3100:3500, 1100:1500]


def distance(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def angle(p1, p2):
    xDiff = p2[0] - p1[0]
    yDiff = p2[1] - p1[1]
    #return degrees(atan2(yDiff, xDiff))
    return atan2(yDiff, xDiff)

io.imshow(image_gray)
io.show()



# blobs
print('Computing laplace of gaussian')
#blobs_log = blob_log(image_gray, max_sigma=35, min_sigma=3, num_sigma=10, threshold=2, overlap=.01)
blobs_log = blob_log(image_gray, max_sigma=35, min_sigma=6, num_sigma=10, threshold=2, overlap=.01)
# Compute radii in the 3rd column.
blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
print('Computed')

fig, ax = plt.subplots(1, 1)
# ax.set_title('Laplacian of Gaussian')
ax.imshow(image_gray)
print('Drawing')
for blob in blobs_log:
    y, x, r = blob
    c = plt.Circle((x, y), r, color='red', linewidth=1, fill=False)
    ax.add_patch(c)
ax.set_axis_off()

plt.tight_layout()
plt.show()


y, x = blobs_log[:,0], blobs_log[:,1]
y = 1500-y
# Define the borders
deltaX = (max(x) - min(x))/10
deltaY = (max(y) - min(y))/10
xmin = min(x) - deltaX
xmax = max(x) + deltaX
ymin = min(y) - deltaY
ymax = max(y) + deltaY
print(xmin, xmax, ymin, ymax)
# Create meshgrid
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([x, y])
kernel = stats.gaussian_kde(values)
f = np.reshape(kernel(positions).T, xx.shape)*10e9

fig = plt.figure(figsize=(8,8))
ax = fig.gca()
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
ax.imshow(np.rot90(f), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
cset = ax.contour(xx, yy, f, colors='k')
ax.clabel(cset, inline=1, fontsize=10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.title('2D Gaussian Kernel density estimation')
for blob in blobs_log:
    ya, xa, r = blob
    #c = plt.Circle((x, y), 3, color='red', linewidth=1, fill=False)
    c = plt.Circle((xa, 1500-ya), 5, color='red', linewidth=1, fill=True)
    ax.add_patch(c)

plt.show()

pt_coords = blobs_log[:,0:2]
tree = spatial.cKDTree(pt_coords,
                       leafsize=16,
                       compact_nodes=True,
                       copy_data=False,
                       balanced_tree=True)

print('Finding neighbors')
neighbor_list = [tree.query_ball_point([x,y], neighbor_search_dist) for y,x in pt_coords]
for i,l in enumerate(neighbor_list):
    if i in l:
        l.remove(i)

distances_list = []
angles_list = []
print('Computing angles and distances')
for (y,x),group in zip(pt_coords,neighbor_list):
    distance_group = []
    angles_group = []
    for neighbor in group:
        nx = pt_coords[neighbor][1]
        ny = pt_coords[neighbor][0]
        d = distance([x,y],[nx,ny])
        a = angle([x,y],[nx,ny])
        distance_group.append(d)
        angles_group.append(a)
    distances_list.append(distance_group)
    angles_list.append(angles_group)

pt_data = {i:{'neighbors':neis, 'distances':dists, 'angles':angs}
           for i,(neis,dists,angs) in
           enumerate(zip(neighbor_list,distances_list,angles_list))}

print('Done')
