import math

import numpy as np
import matplotlib.pyplot as plt


def generate_grid(xmax, ymax, step_x, step_y):
    """
    Generates a numpy array representing a grid

    Args:
        xmax: maximum x limit
        ymax: maximum y limit
        step_x: the step in the x direction
        step_y: the step in the y direction

    Returns:
        a numpy array of grid coordinates
    """

    grid = np.mgrid[step_x - xmax: xmax: step_x * 2, step_y - ymax: ymax: step_y * 2]
    pts = np.array([(x, y) for x, y in zip(grid[0].flatten(), grid[1].flatten())])

    return pts


def translate(array, del_x, del_y):
    """
    Translates a numpy array
    
    Args:
        array: array to be translated
        del_x: the change in x
        del_y: the change in y

    Returns:
        Translated array

    """
    c = array.copy()
    c[:, 0] += del_x
    c[:, 1] += del_y

    return c


def rotate(array, theta):
    """
    Rotates a numpy array

    Args:
        array: array to be rotated
        theta: angle for array to be rotated by, counterclockwise in degrees

    Returns:
        rotated array

    """
    theta /= 57.2958
    c = array.copy()
    r = np.array([[math.cos(theta), -math.sin(theta)],
                  [math.sin(theta), math.cos(theta)]])
    c = c @ r.T

    return c


grid = generate_grid(10, 5, .2, .5)

gridA = rotate(grid, 45)

gridB = rotate(grid,-45)
gridB = translate(gridB,5,0)

total = np.append(gridA,gridB,0)

fig, ax = plt.subplots(1, 1)
ax.scatter(total[:, 0], total[:, 1])
ax.set_aspect('equal')
plt.show()
