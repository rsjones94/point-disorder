from math import sqrt

from skimage.feature import blob_log
import numpy as np

def extract_crowns_from_dhm(dhm, max_sigma=35, min_sigma=6, num_sigma=10, threshold=2, overlap=.01):
    """
    Uses a Laplacian of Gaussian filter to extract tree crowns from a dhm.

    Args:
        dhm: A numpy array representing the dhm
        max_sigma: The maximum standard deviation for the Gaussian kernel. Keep high to detect larger crowns.
        min_sigma: the minimum standard deviation for the Gaussian kernel. Keep low to detect smaller crowns
        num_sigma: The number of intermediate values of standard deviations to consider between min_sigma and max_sigma
        threshold: The absolute lower bound for scale space maxima. Reduce to detect crowns lower to ground
        overlap: A value between 0 and 1. If the area of two blobs overlaps by a
            fraction greater than threshold, the smaller blob is eliminated.

    Returns:
        A numpy array where each row representing the x coord, y coord and radius of a detected tree.

    """
    blobs_log = blob_log(dhm,
                         max_sigma=max_sigma,
                         min_sigma=min_sigma,
                         num_sigma=num_sigma,
                         threshold=threshold,
                         overlap=overlap)
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
    blobs_log[:, 0], blobs_log[:, 1] = np.copy(blobs_log[:, 1]), np.copy(blobs_log[:, 0]) # swap northing and easting columns so the order is x,y,r

    return blobs_log
