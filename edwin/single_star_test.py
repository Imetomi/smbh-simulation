from amuse.datamodel.particles import Particle
from amuse.units import units, quantities, nbody_system

from amuse.community.fi.interface import Fi
from amuse.community.seba.interface import SeBa
from amuse.couple.bridge import Bridge
from amuse.support.console import set_printing_strategy
from amuse.io import write_set_to_file

from amuse.ext import stellar_wind

converter = nbody_system.nbody_to_si(1e-7 | units.MSun, 500 | units.RSun)


def create_star():
    star = Particle()
    star.mass = 19 | units.MSun
    star.radius = 10.7 | units.RSun
    star.luminosity = 5.4e5 | units.LSun
    star.temperature = 4.8e4 | units.K
    star.wind_mass_loss_rate = 2e-5 | units.MSun / units.yr
    star.terminal_wind_velocity = 1.3e3 | units.kms
    star.position = [0, 0, 0] | units.RSun
    star.velocity = [0, 0, 0] | units.ms

    return star


def create_sph_code(timestep):
    sph = Fi(converter)
    sph.parameters.timestep = timestep
    sph.parameters.eps_is_h_flag = True
    sph.parameters.periodic_box_size = 100000 | units.RSun

    return sph


def make_snapshot(filename, stars, gas, time):
    gas.collection_attributes.timestamp = time
    stars.collection_attributes.timestamp = time

    names = ["gas", "stars"]
    particle_sets = [gas, stars]

    write_set_to_file(particle_sets, filename, "amuse", names=names)


def setup_codes(accelerate=False, timestep=0.01 | units.day):
    star = create_star()

    sph = create_sph_code(timestep)
    gas = sph.gas_particles

    star_wind = stellar_wind.new_stellar_wind(
        1e-10 | units.MSun, target_gas=gas, timestep=timestep
    )
    star_wind.particles.add_particle(star)

    bridge = Bridge(use_threading=False)
    bridge.add_system(sph, (star_wind,))
    bridge.add_system(star_wind)
    bridge.timestep = timestep

    return bridge, star_wind, gas


def evolve_wind(
    filename, end_time=1 | units.day, plot_timestep=0.02 | units.day, **kwargs
):
    bridge, star_wind, gas = setup_codes(**kwargs)

    print("Creating initial wind")
    star_wind.create_initial_wind(25)

    time = 0 | units.yr
    while time <= end_time:
        print("Evolving to time", time)
        bridge.evolve_model(time)

        make_snapshot(filename, star_wind.particles, gas, time)

        time += plot_timestep


def guess_input_numbers():
    star = create_star()
    timescale = star.radius / star.terminal_wind_velocity
    massscale = timescale * star.wind_mass_loss_rate

    print(timescale)
    print(massscale)


if __name__ == "__main__":
    set_printing_strategy("custom", preferred_units=[units.MSun, units.RSun, units.day])

    # evolve_wind("sst_simple.amuse")
    evolve_wind("sst_acc.amuse", accelerate=True)
