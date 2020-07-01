import numpy as np
import os
import re

import unyt as u

from my_python.utils import RegisteredFunctor

_elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
             'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
             'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
             'Kr', 'Mo', 'Xe']
_anums = list(range(1, 31)) + [36, 42, 54]
_anum_elem_mapping = dict(zip(_anums, _elements))
_elem_anum_mapping = dict(map(reversed, _anum_elem_mapping.items()))

def atomic_num_to_element(num):
    """Return the symbol for the element with atomic number `num`."""
    try:
        return _anum_elem_mapping[num]
    except KeyError as e:
        raise ValueError(f"Element number {num} is not implemented") from e

def element_to_atomic_num(elem):
    """Return the symbol for the element with atomic number `num`."""
    try:
        return _elem_anum_mapping[elem]
    except KeyError as e:
        raise ValueError(f"Element {elem} is not recognised") from e

_dec_rn_mapping = dict(zip([1, 4, 5, 9, 10, 40, 50, 90],
                           ['I', 'IV', 'V', 'IX', 'X', 'XL', 'L', 'XC']))
_rn_dec_mapping = dict(map(reversed, _dec_rn_mapping.items()))

def rn_to_num(numeral, sub_one=True):
    """
    Convert a roman numeral to the corresponding decimal number.
    If `sub_one` is True, the number returned is one less than the directly equivalent value.
    This option should be set if the number is intended to repesent a charge number (eg H^0),
    and should be left False if it is intended to represent a species number (eg HI).
    """
    if type(numeral) is not str:
        raise ValueError(f"Expected str for type(num), got {type(numeral)}")

    if sub_one:
        num = -1
    else:
        num = 0
    
    try:
        while numeral:
            highnum = max(filter(lambda num: numeral.startswith(num), _rn_dec_mapping),
                          key=lambda v: _rn_dec_mapping[v])
            num += _rn_dec_mapping[highnum]
            numeral = numeral.replace(highnum, '', 1)
    except ValueError as e:
        raise ValueError("Values above 99 are not implemented") from e
    return num

def num_to_rn(num, add_one=False):
    """
    Convert a decimal number to the corresponding roman numeral.
    If `sub_one` is True, the numeral returned is one more than the directly equivalent value.
    This option should be set if the provided `num` repesents a charge number (eg H^0),
    and should be left False if it represents a species number (eg HI).
    """
    if type(num) is not int:
        raise ValueError(f"Expected int for type(num), got {type(num)}")  
    if num > 99:
        raise ValueError("Values above 99 are not implemented")

    if add_one:
        num += 1

    rn = ''
    while num > 0:
        highval = max(filter(lambda x: x <= num, _dec_rn_mapping))
        rn += _dec_rn_mapping[highval]
        num -= highval
    return rn

class IonDataProvider(RegisteredFunctor):
    _data = dict()

    @classmethod
    def load_data(cls):
        for provider in cls.registry:
            provider_data = cls.registry[provider]()
            for field in provider_data:
                for species in provider_data[field]:
                    if species not in cls._data:
                        cls._data[species] = dict()
                    cls._data[species][field] = provider_data[field][species]

    @classmethod
    def get_data(cls, species_name):
        return cls._data[species_name]

    @classmethod
    def __call__(cls):
        return NotImplementedError("__call__ on base DataProvider")

class IpProvider(IonDataProvider, name='ip'):    
    @classmethod
    def __call__(cls):
        """
        Load data for ionisation potentials.
        From http://physics.nist.gov/PhysRefData/ASD/ionEnergy.html
        """
        ip_regex = re.compile(r'((?:\d+\.\d+)|(?:\d+(?=\(\d+\))))')
        data_dict = {}
        with open(os.path.join(os.path.dirname(__file__), 'atomic_ip.dat'), 'r') as datafile:
            for line in datafile.readlines():
                if line.startswith('#'):
                    continue # skip comment lines
                lsplit = line.split('|')
                ion = lsplit[1].strip()
                ip_str = lsplit[8].strip()

                try:
                    ip = float(ip_regex.search(ip_str).group(1))
                    data_dict[ion] = ip * u.eV
                except ValueError as e:
                    raise ValueError(f"offending string: {ip_str}") from e
        return {'ionisation_potential': data_dict}

class PhionProvider(IonDataProvider, name='phion'):
    @classmethod
    def __call__(cls):
        """
        Load data for photoionisation cross sections.
        From Dima Verner - full reference TODO.
        """
        data_dict = {}
        data_arr = np.loadtxt(os.path.join(os.path.dirname(__file__), 'phionxsec.dat'))
        for i in range(data_arr.shape[0]):
            elem = atomic_num_to_element(int(data_arr[i,0]))
            ion_stage = num_to_rn(int(data_arr[i,0]) - int(data_arr[i,1]), add_one=True)

            ion_id = f"{elem} {ion_stage}"
            data_dict[ion_id] = data_arr[i, 2:]
        return {'phion_xsec_params': data_dict}

class RadiativeRecombProvider(IonDataProvider, name='radi_recomb'):
    @classmethod
    def __call__(cls):
        """
        Load data for radiative recombination coefficients.
        From http://amdpp.phys.strath.ac.uk/tamoc/DATA/RR/
        """
        data_dict = {}
        with open(os.path.join(os.path.dirname(__file__), 'recomb_radi.dat'), 'r') as datafile:
            for line in datafile.readlines():
                if line.startswith('#'):
                    continue # skip comment lines
                lsplit = line.split()
                if lsplit[2] != '1':
                    continue # skip M > 1 (these are metastable states above the ground state)
                elem = atomic_num_to_element(int(lsplit[0]))
                ion_stage = num_to_rn(int(lsplit[0]) - int(lsplit[1]))
                ion_id = f"{elem} {ion_stage}"
                
                coeffs = np.array(lsplit[4:], dtype=np.float)
                coeffs.resize(6)
                data_dict[ion_id] = (0, coeffs)
        # manually add missing data
        data_dict["Si I"] = (1, np.array([5.90E-13,0.601]))
        return {'r_recomb_params': data_dict}

class DielectronicRecombProvider(IonDataProvider, name='diel_recomb'):
    @classmethod
    def __call__(cls):
        """
        Load data for dielectronic recombination coefficients.
        From http://amdpp.phys.strath.ac.uk/tamoc/DATA/DR/
        """
        data_dict = {}
        with open(os.path.join(os.path.dirname(__file__), 'recomb_diel_ci.dat'), 'r') as c_file, \
             open(os.path.join(os.path.dirname(__file__), 'recomb_diel_Ei.dat'), 'r') as e_file:
            for c_line, e_line in zip(c_file.readlines(), e_file.readlines()):
                if c_line.startswith('#'):
                    continue # skip comment lines
                c_split = c_line.split()
                e_split = e_line.split()
                if c_split[2] != '1':
                    continue # skip M > 1 (these are metastable states above the ground state)
                elem = atomic_num_to_element(int(c_split[0]))
                ion_stage = num_to_rn(int(c_split[0]) - int(c_split[1]))
                ion_id = f"{elem} {ion_stage}"

                n_coeffs = len(c_split[4:])
                coeffs = np.zeros((n_coeffs, 2))
                coeffs[:,0] = np.array(c_split[4:], dtype=np.float)
                coeffs[:,1] = np.array(e_split[4:], dtype=np.float)
                data_dict[ion_id] = coeffs
        return {'d_recomb_params': data_dict}

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
            raise ValueError(f"Ion {name} is unrecognised") from e

    def photoionisation_cross_section(self, nu):
        """
        Calculate the photoionisation cross-section using data from (TODO reference).

        Parameters:

        nu, ndarray or UnytArray: The frequencies to evaluate the cross-section at.
        If an ndarray, units are assumed to be Hz.
        """
        nu = np.atleast_1d(nu)
        if type(nu) is np.ndarray:
            nu = nu * u.Hz
        
        try:
            Et, Emx, E0, s0, ya, P, yw, y0, y1 = self.phion_xsec_params
        except AttributeError as e:
            raise NotImplementedError(f"Photoionisation cross-section not defined for ion {ion}") from e

        xsec = np.zeros(nu.shape)
        energy = nu.to_equivalent('eV', 'spectral').value

        x = (energy / E0) - y0
        y = (x**2 + y1**2)**0.5
        Fy = ((x - 1.0)**2 + yw**2) * y**(0.5 * P - 5.5) * (1.0 + (y / ya)**0.5)**(-P)

        w = np.where((energy >= Et) & (energy <= Emx))
        xsec[w] = 1.0e-18 * s0 * Fy[w]

        return xsec * u.cm**2

    def recombination_rate(self, temp):
        """
        Calculate the total recombination rate, including radiative, dielectronic and
        charge transfer (TODO) terms where appropriate.

        Parameters:
        temp, ndarray or `Unyt.UnitArray`: The temperatures to evaluate the
        recombination rates at. If an ndarray, units are assumed to be K.
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
