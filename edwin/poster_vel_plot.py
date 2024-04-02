import sys

from amuse.units import units
from matplotlib import pyplot, rc
from amuse import plot as aplot

from amuse.io import read_set_from_file
from amuse.support.console import set_printing_strategy

def velocity_plot(star, gas, figname):
    pyplot.figure(figsize=(10, 3))
    distance = (gas.position - star.position).lengths()

    rad_velocity = []|units.ms
    for pos, vel in zip(gas.position-star.position, gas.velocity-star.velocity):
        rad_direction = pos/pos.length()
        scalar_projection = vel.dot(rad_direction)
        vector_projection = scalar_projection*rad_direction
        vector_rejection = vel - vector_projection

        rad_velocity.append(scalar_projection)

    aplot.plot(distance/star.radius, rad_velocity, "o", c="black")
    r_star = star.radius.value_in(units.RSun)
    pyplot.axvline(1, ls='-', c='black')
    pyplot.axvline(2, ls=':', c='black')
    pyplot.axvline(5, ls=':', c='black')

    pyplot.xlim(0, 10)
    aplot.ylim([0, 1600]|units.kms)
    pyplot.xlabel("distance from star ($R_*$)")
    pyplot.ylabel("radial velocity (km/s)")

    pyplot.savefig(figname)

    pyplot.close()

def poster_velocity_plot(snapshot_file, time, filename):
    gas, stars = read_set_from_file(snapshot_file, 'amuse', names=['gas', 'stars'])
    for i, (g, s) in enumerate(zip(gas.history, stars.history)):
        if g.collection_attributes.timestamp >= time:
            print("creating plot" , filename, "with time", g.collection_attributes.timestamp)

            velocity_plot(s, g, filename)
            break


if __name__ == '__main__':
    set_printing_strategy("custom", preferred_units = [units.MSun, units.RSun, units.day, units.kms])
    rc('font', family='sans')
    rc('text', usetex=True)

    poster_velocity_plot("sst_acc.amuse", 0.7|units.day, "wind_acc.svg")


