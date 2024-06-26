from amuse.units.units import *
from amuse.units import constants

def convert_to_freq(wavelengths = [355.1, 468.6, 616.5, 748.1, 893.1] | nano(m)):
    """
    This function converts wavelength to frequency, using the speed of
    light in vacuum.
    """
    print(f"The speed of light in vacuum: {constants.c}")
    print("wavelength -->  frequency")
    for wavelength in wavelengths:
        print(f"{wavelength}  --> {(constants.c/wavelength).as_quantity_in(giga(Hz))}")


convert_to_freq()