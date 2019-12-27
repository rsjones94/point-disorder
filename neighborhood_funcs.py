import math
import time

import scipy.spatial
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment, differential_evolution

from comparison import BidirectionalDict
from pattern_generation import realign


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


def score_distance_p1p2(p1, p2, ka, coop=1):
    """


    Args:
        x:
        y:
        ka:
        coop:

    Returns:

    """
    d = distance(p1, p2)
    return score_distance(d, ka=ka, coop=coop)


def compare_scatters(s1, s2, plot=False, distance_metric='euclidean', reorient_tol=None):
    """
    Compares two arrays of x-y coords a score that quantifies their similarity.

    Args:
        s1: the first set of points as a numpy array
        s2: the second set of points as a numpy array
        plot: whether to plot the result
        distance_metric: the metric used to compute distance. 'euclidean' or callable
        reorient_tol: if a number is specified, s2 will be reoriented with an iterative Procrustes
            algorithm before comparison using reorient_tol as the tolerance

    Returns:
        A dictionary summarizing the results

    """
    # print('In comparison')

    if reorient_tol:
        #print('Reorienting before comparison...')
        s2, a, b, c = iterative_procrustes(s1, s2, distance_metric=distance_metric, tol=reorient_tol)

    swapped = False
    if len(s1) > len(s2):
        s1, s2 = s2, s1
        swapped = True

    C = cdist(s1, s2, metric=distance_metric)
    _, assignment = linear_sum_assignment(C)

    #  some points in s2 may not be matched. We need to exclude them from the convex hull
    assigned_coords = [s2[i] for i in assignment]
    pared_s2 = assigned_coords.copy()
    unpaired_cords = [s for i, s in enumerate(s2) if i not in assignment]
    assigned_coords.extend(s1)  # we know all of s1 is matched because its length is always < s2
    assigned_coords = np.array(assigned_coords)
    try:
        hull = scipy.spatial.ConvexHull(assigned_coords)
        unpaired_cords_in_hull = [s for s in unpaired_cords if in_hull(s, hull.points)]
    except scipy.spatial.qhull.QhullError:
        hull = None
        unpaired_cords_in_hull = unpaired_cords

    n_smaller = len(s1)
    n_bigger = len(s2)
    n_unpaired = len(unpaired_cords)
    n_unpaired_in_hull = len(unpaired_cords_in_hull)
    deviations = [distance(p, s2[assignment[i]]) for i, p in enumerate(s1)]
    if distance_metric is not 'euclidean':
        scored_vals = [distance_metric(p, s2[assignment[i]]) for i, p in enumerate(s1)]
    else:
        scored_vals = deviations

    if plot:
        plt.figure()
        if not swapped:
            plt.plot(s1[:, 0], s1[:, 1], 'bo', markersize=10)
            plt.plot(s2[:, 0], s2[:, 1], 'rs', markersize=7)
        else:
            plt.plot(s2[:, 0], s2[:, 1], 'bo', markersize=10)
            plt.plot(s1[:, 0], s1[:, 1], 'rs', markersize=7)
        try:
            for simplex in hull.simplices:
                plt.plot(assigned_coords[simplex, 0], assigned_coords[simplex, 1], 'k-', color='red', lw=2)
        except AttributeError:
            pass
        for p in range(min([len(s1), len(s2)])):
            try:
                plt.plot([s1[p, 0], s2[assignment[p], 0]], [s1[p, 1], s2[assignment[p], 1]], 'k')
            except IndexError:
                pass
        plt.axes().set_aspect('equal')
        plt.title(f'Mean deviation: {round(np.mean(deviations), 2)}\n'
                  f'Unpaired: {n_unpaired} ({n_unpaired_in_hull} in hull)')
        plt.show()

    ret_dict = {'n_smaller': n_smaller,
                'n_bigger': n_bigger,
                'n_unpaired': n_unpaired,
                'n_unpaired_in_hull': n_unpaired_in_hull,
                'deviations': deviations,
                'scored_vals': scored_vals,
                'paired_coords': (s1, pared_s2)}

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


def generate_scatter_key(neighborhoods, distance_metric='euclidean', reorient_tol=False):
    """
    Create a BidirectionalDict object where each key relates the neighborhoods of two points.
    A key i,j is only generated if i and j are in one another's neighborhood.

    Args:
        neighborhoods: a list where each entry is a dict with keys for point indices and coordinates
        distance_metric:
        reorient_tol:

    Returns:
        BidirectionalDict

    """
    result = BidirectionalDict()
    n_hoods = len(neighborhoods)
    start = time.time()
    for i, d in enumerate(neighborhoods):
        elap = time.time()
        mins = (elap - start) / 60
        pace = mins / (i+1)
        remaining_iters = n_hoods-(i+1)
        remaining_time = pace*remaining_iters
        print(f'{i} of {n_hoods} (n={len(d["neighbors"])}). {round(mins, 2)} min elapsed. {round(remaining_time, 2)} mins remaining.')
        for j in d['neighbors']:
            if (i, j) not in result:
                result[i, j] = compare_scatters(neighborhoods[i]['coords'],
                                                neighborhoods[j]['coords'],
                                                plot=False,
                                                distance_metric=distance_metric,
                                                reorient_tol=reorient_tol)
    final = time.time()
    total_time = final-start
    print(f'Total time: {round(total_time/60, 2)} minutes')

    return result


def generate_score_key(scatter_key, ka, coop=1, punishment=1, punish_out_of_hull=False):
    """
    Scores a scatter key

    Args:
        scatter_key: a scatter key

    Returns:
        Bidirectional dict of scored scatter relations

    """
    score_key = BidirectionalDict({key: score_comparison(val,
                                                         ka=ka,
                                                         coop=coop,
                                                         punishment=punishment,
                                                         punish_out_of_hull=punish_out_of_hull)
                                   for key, val in scatter_key.items()})
    return score_key


def score_points(neighborhoods, score_key):
    """
    Scores all input points using the bidirectional score key. If a point has no neighbors, it gets np.nan

    Args:
        neighborhoods:
        score_key:

    Returns:
        A list representing the score at each index

    """
    scores = []
    for i, n in enumerate(neighborhoods):
        sub = []
        for j in n['neighbors']:
            sub.append(score_key[i, j])
        if sub:
            scores.append(np.mean(sub))
        else:
            scores.append(np.nan)

    return scores


def point_disorder_index(pts, neighborhood_radius, ka=None, coop=1,
                         punishment=1, punish_out_of_hull=False, euclidean=True, reorient_tol=False):
    """
    Generates the quantified point disorder index for a list of points

    Args:
        pts: a numpy array of points, where each entry is (x,y)
        neighborhood_radius: the radius of the neighborhood
        ka: the deviation distance at which a score of 0.5 is assigned. If not specified,
            neighborhood_radius / 10 is used
        coop: the cooperativity of the scoring curve
        punishment: the score assigned to unpaired points
        punish_out_of_hull: a boolean indicating whether unpaired points outside the
            convex hull should be punished.
        euclidean: whether a euclidean distance metric should be used for bipartite graph analysis. If False,
            the Hill equation will be used instead

    Returns:
        A list of disorder scores for the input points
    """
    if not euclidean:
        distance_metric = lambda p1, p2: score_distance_p1p2(p1, p2, ka, coop)
    else:
        distance_metric = 'euclidean'

    if not ka:
        ka = neighborhood_radius / 10
    print('Composing neighborhoods')
    neighborhoods = compose_neighborhoods(pts, neighborhood_radius)
    print('Generating comparison keys')
    scatter_key = generate_scatter_key(neighborhoods, distance_metric=distance_metric, reorient_tol=reorient_tol)
    print('Scoring')
    score_key = generate_score_key(scatter_key,
                                   ka=ka,
                                   coop=coop,
                                   punishment=punishment,
                                   punish_out_of_hull=punish_out_of_hull)
    scores = score_points(neighborhoods, score_key)
    return scores, neighborhoods, scatter_key, score_key


def score_realignment(s1, s2, realignment_params, distance_metric='euclidean'):
    """
    Computes the mean distance between s1 and a realigned s2

    Args:
        s1: the reference dataset
        s2: data that is being realigned
        realignment_params: realignment to apply to s2: tuple or list [theta, del_x, del_y, reflect]
        distance_metric: the metric used to compute distance

    Returns:
        mean distance between paired points

    """
    print('Realigning for scoring...')
    s2_rel = realign(s2,
                     theta=realignment_params[0],
                     del_x=realignment_params[1],
                     del_y=realignment_params[2],
                     to_reflect=realignment_params[3])
    comparison = compare_scatters(s1, s2_rel, distance_metric=distance_metric)
    result = np.mean(comparison['scored_vals'])
    return result


def evolutionary_transformation(s1, s2, translate=True, rotate=True, reflect=True, distance_metric='euclidean'):
    """
    Aligns s2 such that the distance between point-pairs is minimized. Modified Procrustes analysis
    slow af

    Args:
        s1: a np array of points
        s2: a np array of points
        translate: whether translation is a valid realignment method
        rotate: whether rotation is a valid realignment method
        reflect: whether reflection is a valid realignment method
        rescale: whether rescaling is a valid realignment method
        distance_metric: the method used to evaluate distance. default is euclidean, otherwise a callable

    Returns:
        a tuple (realigned point set, optimal realignment params)

    """

    min_x_s1 = min(s1[:, 0])
    min_x_s2 = min(s2[:, 0])
    max_x_s1 = max(s1[:, 0])
    max_x_s2 = max(s1[:, 0])
    xdif = (max(max_x_s1, max_x_s2) - min(min_x_s1, min_x_s2))

    min_y_s1 = min(s1[:, 1])
    min_y_s2 = min(s2[:, 1])
    max_y_s1 = max(s1[:, 1])
    max_y_s2 = max(s1[:, 1])
    ydif = (max(max_y_s1, max_y_s2) - min(min_y_s1, min_y_s2))

    del_x_lim = (0, xdif)
    del_y_lim = (0, ydif)
    theta_lim = (0, 360)
    reflect_lim = (0, 1)

    print(theta_lim, del_x_lim, del_y_lim, reflect_lim)

    objective = lambda realignment: score_realignment(s1, s2, realignment, distance_metric)
    result = scipy.optimize.differential_evolution(objective,
                                                   (theta_lim, del_x_lim, del_y_lim,
                                                    reflect_lim),
                                                   maxiter=3)

    optimal_realignment_parameters = result.x
    optimal_set = realign(s2,
                          optimal_realignment_parameters[0],
                          optimal_realignment_parameters[1],
                          optimal_realignment_parameters[2],
                          optimal_realignment_parameters[3])

    return optimal_set, optimal_realignment_parameters


def mod_procrustes(data1, data2):
    """
    Modified procrustes fxn where you can get the rotation, scaling and translation out

    Args:
        data1: reference
        data2: data to be transformed

    Returns:
        mtx1
        mtx2
        R
        translation
        scaling

    """
    mtx1 = np.array(data1, dtype=np.double, copy=True)
    mtx2 = np.array(data2, dtype=np.double, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    # translate all the data to the origin
    mtx1_translation = np.mean(mtx1, 0)
    mtx2_translation = np.mean(mtx2, 0)

    mtx1 -= mtx1_translation
    mtx2 -= mtx2_translation

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1
    mtx2 /= norm2

    # transform mtx2 to minimize disparity
    R, s = scipy.linalg.orthogonal_procrustes(mtx1, mtx2)
    mtx2 = np.dot(mtx2, R.T) * s  # HERE, the projected mtx2 is estimated.

    # measure the dissimilarity between the two datasets
    disparity = np.sum(np.square(mtx1 - mtx2))

    return mtx1, mtx2, disparity, R, mtx1_translation - mtx2_translation, norm1 / norm2


def procrustes_transformation(s, rotation, translation, scale):
    """
    Transforms a point set using output from mod_procrustes

    Args:
        s:
        rotation:
        translation:
        scale:

    Returns:

    """
    mod_grid = s.copy()
    grid_mean = np.mean(mod_grid, 0)
    mod_grid = mod_grid - grid_mean
    mod_grid *= scale
    mod_grid = mod_grid @ np.linalg.inv(rotation)
    mod_grid += grid_mean
    mod_grid += translation

    return mod_grid


def iterative_procrustes(s1, s2, distance_metric='euclidean', tol=10e-3):
    """
    Procrustes alignment that does not require prior point assignment or equal number of points

    
    Args:
        s1: the reference point set
        s2: the set to be aligned
        distance_metric: how distances are measured for determining point assignment. 'euclidean' or callable
        tol: the improvement between iterations needed to prevent termination

    Returns:
        the realigned point set, rotation, translation, scale
        
    """
    static = s1.copy()
    to_transform = s2.copy()
    unmodified = to_transform.copy()

    reordered = False
    if len(s1) > len(s2):
        reordered = True

    analysis = compare_scatters(static, to_transform,
                                plot=False,
                                distance_metric=distance_metric,
                                reorient_tol=None)
    p_st, p_tr = analysis['paired_coords']
    if reordered:
        p_st, p_tr = p_tr, p_st
    initial_score = np.mean(analysis['scored_vals'])
    first_score = initial_score

    improvement = tol * 2
    it = 0
    while improvement > tol:
        try:
            mtx1, mtx2, disp, R, trans, scale = mod_procrustes(p_st, p_tr)
        except ValueError:
            R_cum = np.array([np.array([1,0]), np.array([0,1])])
            trans_cum = np.array([0,0])
            scale_cum = 1.0
            break

        if it == 0:
            R_cum = R
            trans_cum = trans
            scale_cum = scale
        else:
            R_cum = R_cum @ R
            trans_cum += trans
            scale_cum += scale

        to_transform = procrustes_transformation(to_transform, R, trans, scale)

        analysis = compare_scatters(static, to_transform,
                                    plot=False,
                                    distance_metric=distance_metric,
                                    reorient_tol=None)
        p_st, p_tr = analysis['paired_coords']
        if reordered:
            p_st, p_tr = p_tr, p_st
        final_score = np.mean(analysis['scored_vals'])

        improvement = initial_score - final_score
        initial_score = final_score

        #print(f'IMPROVEMENT on iteration {it}: {round(improvement, 5)}')

        it += 1
    try:
        if first_score < final_score: # if we actually don't improve anything
            #print('\n\nNO IMPROVEMENT\n\n')
            to_transform = unmodified
            R_cum = np.array([np.array([1,0]), np.array([0,1])])
            trans_cum = np.array([0,0])
            scale_cum = 1.0
    except UnboundLocalError:
        pass

    return to_transform, R_cum, trans_cum, scale_cum
