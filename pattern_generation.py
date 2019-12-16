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

def re():
    return 1

def realign(array, theta, del_x, del_y, to_reflect):
    """
    Reflects, rotates and then translates an array in that order

    Args:
        array: array to be realigned
        del_x: the change in x
        del_y: the change in y
        theta: the rotation angle
        to_reflect: whether to reflect (>= 0.5) or not (< 0.5)

    Returns:
        the realigned data

    """
    c = array.copy()
    c = reflect(c, holder=to_reflect)
    c = rotate(c, theta)
    c = translate(c, del_x, del_y)
    return c


def reflect(array, holder=1):
    """
    Reflects a np array across the y-axis

    Args:
        array: array to be reflected
        holder: a holder variable so the function can be used in optimization algorithms. If <0.5, does not reflect.

    Returns:
        Reflected array

    """
    c = array.copy()
    if holder < 0.5:
        c[:, 0] = -c[:, 0]

    return c


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
