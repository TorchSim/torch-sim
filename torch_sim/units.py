# ruff: noqa: N815, PIE796
"""Unit systems and conversions.

Defines a units system and returns a dictionary of conversion factors.
Units are defined similar to https://docs.lammps.org/units.html.
"""

from enum import Enum
from math import pi, sqrt
from typing import Self


class BaseConstant:
    """CODATA Recommended Values of the Fundamental Physical Constants: 2014.

    References:
        http://arxiv.org/pdf/1507.07956.pdf
        https://wiki.fysik.dtu.dk/ase/_modules/ase/units.html#create_units
    """

    c = 299792458.0  # speed of light, m/s
    mu0 = 4.0e-7 * pi  # permeability of vacuum
    grav = 6.67408e-11  # gravitational constant
    h_planck = 6.626070040e-34  # Planck constant, J s
    e = 1.6021766208e-19  # elementary charge
    m_e = 9.10938356e-31  # electron mass
    m_p = 1.672621898e-27  # proton mass
    n_av = 6.022140857e23  # Avogadro number
    k_B = 1.38064852e-23  # Boltzmann constant, J/K
    amu = 1.660539040e-27  # atomic mass unit, kg


bc = BaseConstant


class UnitConversion:
    """Unit conversion class for different unit systems.

    Distance:
    Ang (Angstrom)
    met (meter)

    Time:
    ps (picosecond)
    s (second)
    fs (femtosecond)

    Pressure:
    atm (atmosphere)
    pa (pascal)
    bar (bar)
    GPa (GigaPascal)

    Energy:
    cal (calorie)
    kcal (kilocalorie)
    eV (electron volt)
    """

    # Distance
    Ang_to_met = 1e-10
    Ang2_to_met2 = Ang_to_met * Ang_to_met
    Ang3_to_met3 = Ang_to_met * Ang2_to_met2

    # Time
    ps_to_s = 1e-12
    fs_to_s = 1e-15

    # Pressure
    bar_to_pa = 1e5
    atm_to_pa = 101325
    pa_to_GPa = 1e-9
    eV_per_Ang3_to_GPa = (bc.e / Ang3_to_met3) * pa_to_GPa

    # Energy
    cal_to_J = 4.184
    kcal_to_cal = 1e3
    eV_to_J = bc.e


uc = UnitConversion


class MetalUnits(float, Enum):
    """Metal unit system using Angstroms, eV, amu, and proton charge."""

    def __new__(cls, value: float) -> Self:
        """Create new MetalUnits enum value."""
        return float.__new__(cls, value)

    # Base units
    mass = 1.0  # Default mass in amu
    distance = 1.0  # Default distance in Angstrom
    energy = 1.0  # Default energy in eV
    charge = 1.0  # Default charge in proton charge

    # Derived units
    time = sqrt(energy * bc.e / (bc.amu * uc.Ang2_to_met2)) * uc.ps_to_s  # picoseconds
    velocity = distance / time  # Ang/ps
    force = energy / distance  # eV/Ang
    torque = energy  # eV
    temperature = bc.k_B / bc.e  # Boltzmann in eV/K
    pressure = uc.bar_to_pa * (energy * uc.Ang3_to_met3 / bc.e)  # bar
    electric_field = charge * distance  # e*Ang


class RealUnits(float, Enum):
    """Real unit system using Angstroms, kcal/mol, and proton charge."""

    def __new__(cls, value: float) -> Self:
        """Create new RealUnits enum value."""
        return float.__new__(cls, value)

    # Base units
    mass = 1.0  # Default mass in grams/mol
    distance = 1.0  # Default distance in Angstrom
    energy = 1.0  # Default energy in kcal/mol
    charge = 1.0  # Default charge in proton charge

    # Derived units
    time = (
        sqrt(
            energy / (bc.amu * uc.Ang2_to_met2 * bc.n_av / (uc.cal_to_J * uc.kcal_to_cal))
        )
        * uc.fs_to_s
    )  # femtoseconds
    velocity = distance / time  # Ang/fs
    force = energy / distance  # kcal/mol/Ang
    torque = energy  # kcal/mol
    temperature = bc.k_B * bc.n_av / uc.cal_to_J / uc.kcal_to_cal  # kcal/mol K
    pressure = (
        uc.Ang3_to_met3 * (bc.n_av / (uc.cal_to_J * uc.kcal_to_cal)) * uc.bar_to_pa
    )  # bar
    electric_field = charge * distance  # e*Ang


class UnitSystem:
    """Container class for unit systems."""

    metal = MetalUnits
    real = RealUnits
