import math

import scipy.spatial
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def distance(p0, p1):
    return math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)


def get_neighbors(pt_coords, radius):
    """
    Finds all points in a list within a certain radius for each point.

    Args:
        pt_coords: a numpy array where each row is the x-y coordinates of a point
        radius: the search radius

    Returns:
         A list of lists, where each sublist is the indices of the points
    that are within the radius around the point at the the sublist's index

    """
    tree = scipy.spatial.cKDTree(pt_coords,
                                 leafsize=16,
                                 compact_nodes=True,
                                 copy_data=False,
                                 balanced_tree=True)
    neighbor_list = [tree.query_ball_point([x, y], radius) for x, y in pt_coords]
    for i, l in enumerate(neighbor_list):
        if i in l:
            l.remove(i)

    return neighbor_list


def compute_relative_positions(pt_coords, relative_points):
    """
    Computes the relative x-y positions of a list of points against specified neighbors

    Args:
        pt_coords: a numpy array where each row represents the x-y coords of a point
        relative_points: a list of sublists, relative_points[i] the indices of the points in
        pt_coords that should be compared to pt_coords[i]

    Returns:
        A list of lists, where each sublist at index i has rows specifying the relative x-y
        positioning of the points at relative_points[i] against pt_coords[i]

    """
    rel_coords_list = []
    for (x, y), group in zip(pt_coords, relative_points):
        rel_coords_group = []
        for neighbor in group:
            nx = pt_coords[neighbor][0]
            ny = pt_coords[neighbor][1]

            delta_x = nx - x
            delta_y = ny - y

            rel_coords_group.append([delta_x, delta_y])
        rel_coords_list.append(rel_coords_group)

    return rel_coords_list


def compose_neighborhoods(pt_coords, radius):
    """
    Creates a list of dictionaries that describe the neighborhood at each index.

    Args:
        pt_coords: a numpy array of points, where each row is an x-y coordinate
        radius: the neighborhood radius

    Returns:
        A list dictionaries with two keys each: 'neighbors' and 'coords'
            neighbors: a list of indices indicating which points are in the neighborhood
            coords: relative x-y coordinates corresponding to each neighbor as np array

    """
    neighbors = get_neighbors(pt_coords, radius)
    coords = compute_relative_positions(pt_coords, neighbors)

    return [{'neighbors': n, 'coords': np.array(c)}
            for n, c in
            zip(neighbors, coords)]


def score_comparison(comp_dict, ka, coop=1, punishment=1, punish_out_of_hull=False):
    """
    Returns a similarity score for the comparison between two scatters produced by
    compare_scatters(). A 0 is perfectly similar, while a 1 is maximally dissimilar

    Args:
        comp_dict: the comparison dictionary returned by compare_scatters
        ka: the deviation at which a score of 0.5 is assigned
        coop: the cooperativity coefficient of the scoring curve
        punishment: the score assigned to unmatched points
        punish_out_of_hull: whether unmatched points that fall outside the hull should be punished

    Returns:
        A float between 0 and 1.

    """
    deviation_scores = [score_distance(d, ka, coop) for d in comp_dict['deviations']]
    if punish_out_of_hull:
        punish_scores = [punishment for i in range(comp_dict['n_unpaired'])]
    else:
        punish_scores = [punishment for i in range(comp_dict['n_unpaired_in_hull'])]

    deviation_scores.extend(punish_scores)

    return np.mean(deviation_scores)

def score_distance(d, ka, coop=1):
    """
    Given some distance d, returns a score on (0,1]. A d of 0 scores 0, and a d of inf scores 1.
    gamma defines the distance at which the score is 0.5. Modeled off the Hill equation

    Args:
        d: The value to score
        ka: The value at which the score is 0.5
        cooperativity: the cooperativity coeffient

    Returns:
        float

    """
    score = d ** coop / (ka ** coop + d ** coop)
    return score


def compare_scatters(s1, s2, plot=False):
    """
    Compares two arrays of x-y coords a score that quantifies their similarity.

    Args:
        s1: the first set of points as a numpy array
        s2: the second set of points as a numpy array

    Returns:
        A dictionary summarizing the results

    """
    swapped = False
    if len(s1) > len(s2):
        s1, s2 = s2, s1
        swapped = True

    C = cdist(s1, s2)
    _, assignment = linear_sum_assignment(C)

    #  some points in s2 may not be matched. We need to exclude them from the convex hull
    assigned_coords = [s2[i] for i in assignment]
    unpaired_cords = [s for i, s in enumerate(s2) if i not in assignment]
    assigned_coords.extend(s1)  # we know all of s1 is matched because its length is always < s2
    assigned_coords = np.array(assigned_coords)
    hull = scipy.spatial.ConvexHull(assigned_coords)
    unpaired_cords_in_hull = [s for s in unpaired_cords if in_hull(s, hull.points)]

    n_smaller = len(s1)
    n_bigger = len(s2)
    n_unpaired = len(unpaired_cords)
    n_unpaired_in_hull = len(unpaired_cords_in_hull)
    deviations = [distance(p, s2[assignment[i]]) for i, p in enumerate(s1)]

    if plot:
        plt.plot(s1[:, 0], s1[:, 1], 'bo', markersize=10)
        plt.plot(s2[:, 0], s2[:, 1], 'rs', markersize=7)
        for p in range(min([len(s1), len(s2)])):
            try:
                plt.plot([s1[p, 0], s2[assignment[p], 0]], [s1[p, 1], s2[assignment[p], 1]], 'k')
            except IndexError:
                pass
        plt.axes().set_aspect('equal')
        for simplex in hull.simplices:
            plt.plot(assigned_coords[simplex, 0], assigned_coords[simplex, 1], 'k-', color='red')
        plt.show()

    ret_dict = {'n_smaller': n_smaller,
                'n_bigger': n_bigger,
                'n_unpaired': n_unpaired,
                'n_unpaired_in_hull': n_unpaired_in_hull,
                'deviations': deviations}

    return ret_dict


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed

    Courtesy Juh_ on StackOverFlow
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p) >= 0
