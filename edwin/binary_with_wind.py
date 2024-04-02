import subprocess

from matplotlib import pyplot
from amuse.plot import pynbody_column_density_plot

from amuse.units import units, constants, nbody_system
from amuse.datamodel import Particles

from amuse.community.gadget2.interface import Gadget2
from amuse.community.fi.interface import Fi
from amuse.community.sse.interface import SSE
from amuse.community.twobody.interface import TwoBody
from amuse.community.kepler.interface import Kepler
from amuse.couple.bridge import Bridge


from amuse.ext.evrard_test import uniform_unit_sphere
from amuse.ext.sink import new_sink_particles


converter = nbody_system.nbody_to_si(1 | units.MSun, 1 | units.RSun)
hydro_converter = nbody_system.nbody_to_si(1.0e-9 | units.MSun, 500 | units.RSun)


def make_movie(filename="hydro", directory="./plots"):
    print("   Creating movie from snapshots")
    try:
        collection_name = "mf://" + filename + "*.png"
        movie_name = "../" + filename + ".avi"
        subprocess.call(
            [
                "mencoder",
                collection_name,
                "-ovc",
                "lavc",
                "-o",
                movie_name,
                "-msglevel",
                "all=1",
            ],
            cwd=directory,
        )
    except Exception as exc:
        print(("   Failed to create movie, error was:", str(exc)))


class WindBinary(object):

    def __init__(
        self,
        masses,
        max_radius,
        semi_major_axis,
        eccentricity=0.0,
        dt=1.0 | units.day,
        mgas=1.0e-10 | units.MSun,
        system_size_in_a=5,
    ):
        self.timestep = dt
        self.mgas = mgas
        self.time = 0 | units.yr
        self.removed_particles = Particles()

        self.create_stars(masses, max_radius)
        self.create_binary_orbit(semi_major_axis, eccentricity)
        self.create_orbit_integrator()
        self.setup_hydro(1e-4 | units.MSun, 0.01 * semi_major_axis)
        self.connect_codes()

        self.system_size = system_size_in_a * semi_major_axis
        self.system_escape_velocity = (
            2.0 * constants.G * self.stars.mass.sum() / self.system_size
        ).sqrt()

    def create_stars(self, masses, max_radius):
        stev = SSE()
        stev.commit_parameters()

        particles = Particles(2)
        particles.mass = masses
        self.stars = stev.particles.add_particles(particles)
        stev.commit_particles()

        print(("Evolve primary to radius:", max_radius))
        primary = self.stars[0]
        while primary.radius < max_radius:
            last_mass = primary.mass
            last_time = primary.age
            stev.evolve_model()
            dm = last_mass - primary.mass
            dt = last_time - primary.age

        self.v_wind = 1.1 * (2.0 * constants.G * primary.mass / primary.radius).sqrt()
        self.massloss_rate = dm / dt
        self.accumulated_massloss = 0 | units.MSun
        stev.stop()

    def create_binary_orbit(self, semi_major_axis, eccentricity):
        johannes = Kepler(converter)
        johannes.initialize_code()
        johannes.set_longitudinal_unit_vector(1.0, 0.0, 0.0)
        johannes.set_transverse_unit_vector(0.0, 1.0, 0.0)
        johannes.set_normal_unit_vector(0.0, 0.0, 1.0)

        total_mass = self.stars.mass.sum()
        johannes.initialize_from_elements(total_mass, semi_major_axis, eccentricity)
        johannes.transform_to_time(0 | units.s)

        secondary = self.stars[1]
        secondary.position = johannes.get_separation_vector()
        secondary.velocity = johannes.get_velocity_vector()
        self.stars.move_to_center()

        porb = (
            2.0
            * constants.pi
            * (semi_major_axis**3 / (constants.G * total_mass)).sqrt()
        ).in_(units.yr)
        self.porb = porb

    def create_orbit_integrator(self):
        # Initialise the orbit integrator
        integrator = TwoBody(converter)
        primary = self.stars[0]
        secondary = self.stars[1]
        integrator.particles.add_particles(self.stars)
        integrator.commit_particles()
        self.gravity = integrator

        # Create sink particle for *2
        self.sink = new_sink_particles(integrator.particles)

        # Create communication channels to transfer data from the dynamics
        self.from_model_to_gravity = self.stars.new_channel_to(integrator.particles)
        self.from_gravity_to_model = integrator.particles.new_channel_to(self.stars)

    def setup_hydro(self, particle_mass, epsilon):
        self.hydro = Fi(
            hydro_converter, redirection="file", redirect_file="sph_code_out.log"
        )
        self.hydro.parameters.epsilon_squared = epsilon**2
        self.hydro.parameters.timestep = self.timestep

        # make sure we start with some particles
        self.accumulated_massloss = 1 * self.mgas

    def connect_codes(self):
        self.gravhydro = Bridge(use_threading=False)
        self.gravhydro.add_system(self.hydro, (self.gravity,))
        self.gravhydro.add_system(self.gravity)
        # gravhydro.add_system(gravity, (hydro,) )
        self.gravhydro.timestep = min(self.timestep, 2 * self.hydro.parameters.timestep)

    def create_gas_particle(self):
        self.accumulated_massloss += self.massloss_rate * self.timestep
        Ngas = int(-self.accumulated_massloss / self.mgas)
        if Ngas < 0:
            return Particles()

        # Correct accumulated mass loss for number of created particles
        Mgas = self.mgas * Ngas
        self.accumulated_massloss += Mgas
        new_particles = Particles(Ngas)
        new_particles.mass = self.mgas
        # new_particles.h_smooth=0. | units.RSun

        # Distribute particles uniformly on the surfcae of a unit sphere
        dx, dy, dz = uniform_unit_sphere(Ngas).make_xyz()
        r = (dx**2 + dy**2 + dz**2) ** 0.5
        dx /= r
        dy /= r
        dz /= r

        # Create wind particles with radially outward velocity
        primary = self.stars[0]
        new_particles.x = primary.x + (dx * primary.radius)
        new_particles.y = primary.y + (dy * primary.radius)
        new_particles.z = primary.z + (dz * primary.radius)
        for particle in new_particles:
            r = particle.position - primary.position
            r = r / r.length()
            particle.u = 0.5 * (self.v_wind) ** 2
            particle.velocity = primary.velocity + r * self.v_wind
        return new_particles

    def is_escaping(self, position, velocity):
        r_squared = (position**2).sum()
        is_outside = r_squared > self.system_size**2
        if not is_outside:
            return False

        is_moving_away = (velocity * position > 0 | units.m**2 / units.s).all()
        if not is_moving_away:
            return False

        v_squared = (velocity**2).sum()
        has_escape_velocity = v_squared > self.system_escape_velocity**2
        return has_escape_velocity

    def remove_gas_particles(self):
        gas = self.hydro.gas_particles
        self.sink.accrete(gas)

        too_far = gas.select(self.is_escaping, ["position", "velocity"])
        if len(too_far) > 0:
            self.removed_particles.add_particles(too_far)
            gas.remove_particles(too_far)

    def evolve(self, time):
        while self.time < time:
            self.time += self.timestep
            print(
                (
                    "evolving to time ",
                    self.time.in_(units.day),
                    "using",
                    len(self.hydro.gas_particles),
                    "gas particles.",
                )
            )
            gas = self.create_gas_particle()
            self.hydro.gas_particles.add_particles(gas)

            self.gravhydro.evolve_model(self.time)
            self.remove_gas_particles()
            self.from_gravity_to_model.copy()

            self.plot_binary(
                "plots/binary_hydro_{0:=07}.png".format(self.time.value_in(units.day))
            )

    def plot_binary(self, figname=None, with_column_density=False):
        primary = self.stars[0]
        secondary = self.stars[1]
        gas = self.hydro.gas_particles

        fig = pyplot.figure(figsize=(10, 10))

        if with_column_density:
            pynbody_column_density_plot(gas, width=30 | units.AU, vmin=13, vmax=25)
        else:
            pyplot.xlabel("x (AU)")
            pyplot.ylabel("y (AU)")
        primary_circle = pyplot.Circle(
            primary.position[:2].value_in(units.AU),
            primary.radius.value_in(units.AU),
            color="r",
        )
        pyplot.gcf().gca().add_artist(primary_circle)
        pyplot.plot(
            gas.x.value_in(units.AU),
            gas.y.value_in(units.AU),
            ".",
            color="k",
            linewidth=0.1,
        )
        pyplot.plot(
            [secondary.x.value_in(units.AU)], [secondary.y.value_in(units.AU)], "or"
        )
        pyplot.plot([0], [0], "+")

        pyplot.xlim([-15, 15])
        pyplot.ylim([-15, 15])
        if figname:
            pyplot.savefig(figname)
        else:
            pyplot.show()
        pyplot.close()


if __name__ == "__main__":
    binary = WindBinary(
        [3.0, 100.0] | units.MSun,
        300.0 | units.RSun,
        1000.0 | units.RSun,
        0.0,
        system_size_in_a=1,
    )

    binary.evolve(3000 | units.day)

    print()
    print(("total removed:", len(binary.removed_particles)))

    make_movie("binary_hydro")
