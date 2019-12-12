import scipy.spatial


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
            coords: relative x-y coordinates corresponding to each neighbor

    """
    neighbors = get_neighbors(pt_coords, radius)
    coords = compute_relative_positions(pt_coords, neighbors)

    return [{'neighbors': n, 'coords': c}
            for n, c in
            zip(neighbors, coords)]


def score_scatters(s1, s2):
    """
    Compares to arrays of x-y coords and returns the average distance between the best pairs
    of points found. Different penalization options are available for unpaired points.

    Args:
        s1:
        s2:

    Returns:

    """
