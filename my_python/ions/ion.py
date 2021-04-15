import numpy as np
import unyt as u

from .ion_utils import *
from .ion_data_provider import IonDataProvider

from my_python.utils import require_attrs

class Ion:
    def __init__(self, name):
        try:
            elem, ion_stage = name.split()   
        except ValueError as e:
            raise ValueError(f"Ion {name} is invalid") from e

        self.name = name
        self.element = elem
        self.atomic_number = element_to_atomic_num(elem)
        self.charge = rn_to_num(ion_stage, sub_one=True)
        if self.charge > self.atomic_number:
            raise ValueError(f"Cannot have ion {name} with charge {self.charge} greater than atomic number {self.atomic_number}")

        self.available_fields = set()
        try:
            data = IonDataProvider.get_data(name)
            for field in data:
                setattr(self, field, data[field])
                self.available_fields.add(field)
        except KeyError as e:
            raise ValueError(f"Ion {self} is unrecognised") from None

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Ion({self.name})"

    def __getattr__(self, name):
        try:
            return self.__getattribute__(name)
        except AttributeError:
            raise AttributeError(f"Attribute '{name}' is not defined for ion {self}") from None
        
    @require_attrs('phion_xsec_params')
    def photoionisation_cross_section(self, nu):
        """
        Calculate the photoionisation cross-section using data from (TODO reference).
        nu, float or ndarray, or unyt_quantity or unyt_array:
            The frequencies to evaluate the cross-section at.
            If float or ndarray, units are assumed to be Hz.
        """
        nu = np.atleast_1d(nu)
        if not isinstance(nu, (u.unyt_array, u.unyt_quantity)):
            nu = nu * u.Hz

        # E_th is the first element of `self.phion_xsec_params` but we ignore it and use the value
        # from `self.ionisation_potential` instead, so that `self.photoionisation_cross_section(
        # self.ionisation_potential)` gives the expected result.
        E_th, E_max, E0, s0, ya, P, yw, y0, y1 = self.phion_xsec_params
        E_th = self.ionisation_potential
        xsec = np.zeros_like(nu.ndview)
        energy = nu.to('eV', equivalence='spectral').value

        x = (energy / E0) - y0
        y = (x**2 + y1**2)**0.5
        Fy = ((x - 1.0)**2 + yw**2) * y**(0.5 * P - 5.5) * (1.0 + (y / ya)**0.5)**(-P)

        w = ((energy >= E_th) & (energy <= E_max))
        xsec[w] = 1.0e-18 * s0 * Fy[w]

        return xsec * u.cm**2

    @require_attrs('r_recomb_params|d_recomb_params')
    def recombination_rate(self, temp):
        """
        Calculate the total recombination rate, including radiative, dielectronic and
        charge transfer (TODO) terms where appropriate.

        temp, float or ndarray, or unyt_quantity or unyt_array:
            The temperatures to evaluate the recombination rates at.
            If float or ndarray, units are assumed to be K.
        """
        temp = np.atleast_1d(temp)
        if type(temp) is np.ndarray:
            temp = temp * u.K
        temp = temp.to_equivalent(u.K, 'thermal').value

        if 'r_recomb_params' in self.available_fields:
            form = self.r_recomb_params[0]

            if form == 0: # Badnell fit
                A, b0, T0, T1, c, T2 = self.r_recomb_params[1]

                b = b0 + c * np.exp(-T2 / temp)
                term_0 = (1 + (temp / T0)**0.5)**(1 - b)
                term_1 = (1 + (temp / T1)**0.5)**(1 + b)
                rate_r = A * ((temp / T0)**0.5 * term_0 * term_1)**-1
            elif form == 1: # Simple power law
                A, beta = self.r_recomb_params[1]

                rate_r = A * (temp / 1.0e4)**-beta
        else:
            rate_r = np.zeros(temp.shape)

        if 'd_recomb_params' in self.available_fields:
            c_i, E_i = self.d_recomb_params
            
            rate_d = np.sum(c_i * np.exp(-E_i / temp))
        else:
            rate_d = np.zeros(temp.shape)
        
        return (rate_r + rate_d) * u.cm**3 * u.s**-1
