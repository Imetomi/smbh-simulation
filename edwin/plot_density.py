"""
    Plot the density of the bow shock
"""

import numpy
from matplotlib import pyplot, cm
from amuse.plot import *
import math


def density(x, y):
    return numpy.sqrt(-y - x**2 + 0j).real * 10 * numpy.exp(20 * y + 18 * x**2)


def orbit_location(semi_major_axis, eccentricity, true_anomaly):
    r = semi_major_axis
    x = r * math.cos(true_anomaly)
    y = r * math.sin(true_anomaly)
    return x, y


def create_ticks(value_array, resolution):
    tick_locations = list(range(0, len(value_array), resolution))
    tick_values = numpy.rint(value_array[::resolution])
    return tick_locations, tick_values


def plot_color_map(
    star, pdf_name=None, x=[-2, 2], y=[-3, 1], formula=density, resolution=100
):
    X = numpy.linspace(x[0], x[1], resolution * (x[1] - x[0]))
    Y = numpy.linspace(y[0], y[1], resolution * (y[1] - y[0]))
    X, Y = numpy.meshgrid(X, Y)

    Z = formula(X, Y)

    fig = pyplot.figure()
    ax = fig.add_subplot(111)

    cax = ax.imshow(Z, origin="lower")

    ax.scatter((star[0] - x[0]) * resolution, (star[1] - y[0]) * resolution, c="w")
    fig.colorbar(cax)
    pyplot.xlabel("x")
    pyplot.xticks(*create_ticks(X[0], resolution))
    pyplot.ylabel("y")
    pyplot.yticks(*create_ticks(Y[:, 0], resolution))
    pyplot.show()


plot_color_map([0, -0.5])
