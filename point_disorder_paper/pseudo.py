
def index_of_disorder(point_set, radius):
    disorder_scores = []
    for point in point_set:
        neighborhood = find_neighborhood(point_set, point, radius)
        # get the subset of points in point_set within radius of point
        comparison_scores = []
        for neighbor in neighborhood:
            comparison_neighborhood = find_neighborhood(point_set, neighbor, radius)
            registrations = register_points(neighbor, comparison_neighborhood)
            comparison_score = mean(registrations.registration_costs)
            comparison_score.append(comparison_score)
        disorder_score = mean(comparison_scores)
        disorder_scores.append(disorder_score)