import numpy as np
from sklearn.neighbors import NearestNeighbors

from neighborhood_funcs import compare_scatters

"""
Courtesy of Clay Flannigan (https://github.com/ClayFlannigan/icp/blob/master/icp.py)
"""

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def cut_points_down(A, B, distance_metric='euclidean'):
    """
    Registers points in two two sets of data and then cuts out the unmatched points


    Args:
        A: a matrix
        B: another matrix
        distance_metric: registration metric

    Returns:
        Both data sets, but with unregistered points removed from the larger set

    """
    reordered = False
    if len(A) > len(B):
        reordered = True

    analysis = compare_scatters(A, B,
                                plot=False,
                                distance_metric=distance_metric,
                                reorient_tol=None)

    if reordered:
        B_matched, A_matched = analysis['paired_coords']
    else:
        A_matched, B_matched = analysis['paired_coords']

    A_matched = np.array(A_matched)
    B_matched = np.array(B_matched)

    return A_matched, B_matched


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001, distance_metric='euclidean'):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    #assert A.shape == B.shape

    A_matched, B_matched = cut_points_down(A, B, distance_metric=distance_metric)

    # get number of dimensions
    m = A_matched.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A_matched.shape[0]))
    dst = np.ones((m+1,B_matched.shape[0]))
    src[:m,:] = np.copy(A_matched.T)
    dst[:m,:] = np.copy(B_matched.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        A_matched, B_matched = cut_points_down(A, B, distance_metric=distance_metric)

        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(A_matched, src[:m,:].T)

    return T, distances, i


def realign_points(A, B, distance_metric='euclidean', tolerance=0.001, max_iterations=20):
    """
    ICP realigns points in A to B

    Args:
        A: a matrix
        B: another matrix
        distance_metric: registration metric

    Returns:
        A realigned to B
    """
    t, distances, iterations = icp(A, B, distance_metric=distance_metric, tolerance=tolerance, max_iterations=max_iterations)
    C = np.ones((len(A), 3))
    C[:, 0:2] = np.copy(A)
    readj = np.dot(t, C.T).T

    return readj[:,0:2]