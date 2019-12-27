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


def peturb_constant(grid, amplitude, radius=0):
    """
    Takes a grid of point and applies a constant peturbation to it. Only peturbs points greater than a distance of radius
    from the origin

    Args:
        grid: input grid
        amplitude: standard deviation of amplitude
        radius: radius of non-peturbation

    Returns:
        peturbed grid

    """
    d = [(x**2 + y**2)**0.5 for x, y in grid]
    pt_copy = []
    for i,(pt,di) in enumerate(zip(grid,d)):
        if di > radius:
            adder = [np.random.normal(pt[0], amplitude),
                     np.random.normal(pt[1], amplitude)]
        else:
            adder = pt
        pt_copy.append(adder)
    pts = np.array(pt_copy)
    return pts


def peturb_gradational(grid, amplitude_per_distance, radius=0):
    """
    Takes a grid of point and applies a gradational peturbation to it. Only peturbs points greater than a distance of radius
    from the origin

    Args:
        grid: input grid
        amplitude: standard deviation of amplitude gradient
        radius: radius of non-peturbation

    Returns:
        peturbed grid

    """
    d = [(x**2 + y**2)**0.5 for x, y in grid]
    pt_copy = []
    for i,(pt,di) in enumerate(zip(grid,d)):
        if di > radius:
            adder = [np.random.normal(pt[0], amplitude_per_distance * (di - radius)),
                     np.random.normal(pt[1], amplitude_per_distance * (di - radius))]
        else:
            adder = pt
        pt_copy.append(adder)
    pts = np.array(pt_copy)
    return pts


def decimate_grid(grid, frac):
    """
    Randomly deletes points in a grid

    Args:
        grid: input grid
        perc: fraction of points to keep

    Returns:
        decimated grid

    """
    grid = grid.copy()
    np.random.shuffle(grid)
    keep = int(len(grid)*frac)
    ptc = []
    new_d = []
    for i in range(keep):
        ptc.append(grid[i])
        new_d.append(grid[i])
    pts = np.array(ptc)
    return pts


def wavify_grid(grid, amplitude, period):
    """
    Add adds a sinusoidal signal to a grid. The signal is based on the x axis only

    Args:
        grid: input grid
        amplitude: ampltidue of the signal
        period: period of the signal

    Returns: a fun and wavy grid

    """

    grid = np.array([[pt[0] + amplitude*math.sin((2*math.pi/period)*pt[1]),
                    pt[1]] for i,pt in enumerate(grid)])
    return grid


def make_circle(r, n):
    """


    Args:
        r: radius
        n: number of points

    Returns: np array

    """

    phis = np.linspace(0, 2 * np.pi, n)
    rhos = np.ones(n) * r

    def pol2cart(rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return np.array([x, y])

    cart_grid = np.array([pol2cart(rho, phi) for rho, phi in zip(rhos, phis)])
    return cart_grid


def concentric_circles(num_circs, lin_density, distance_between_circles):

    r = distance_between_circles
    circ = 2 * np.pi * r
    n_per_circle = int(np.round(lin_density * circ))
    print(f'Circ:{circ}. r:{r}. n:{n_per_circle}')
    grid = make_circle(r, n_per_circle)
    print('inital')

    for n in range(num_circs):
        r += distance_between_circles
        circ = 2 * np.pi * r
        n_per_circle = int(np.round(circ * lin_density))
        print(f'Circ:{circ}. r:{r}. n:{n_per_circle}')
        appender = make_circle(r, n_per_circle)
        print(f'Appender:{appender.shape}. Grid:{grid.shape}')
        grid = np.append(grid, appender,0)
        print('appended')

    return grid

"""
a = concentric_circles(16, 0.5, 6)
plt.scatter(a[:,0], a[:,1])
"""