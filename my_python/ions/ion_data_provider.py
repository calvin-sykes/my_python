from collections import defaultdict
import re
import os

import numpy as np
import unyt as u

from .ion_utils import *
from ..utils import RegisteredFunctor

def _makehash():
    return defaultdict(_makehash)

class IonDataProvider(RegisteredFunctor):
    _data = _makehash()

    @classmethod
    def load_data(cls):
        for provider in cls.registry:
            provider_data = cls.registry[provider]()
            for field in provider_data:
                for species in provider_data[field]:
                    cls._data[species][field] = provider_data[field][species]

    @classmethod
    def get_data(cls, species_name):
        if len(cls._data[species_name]):
            return cls._data[species_name]
        else:
            raise KeyError

    @classmethod
    def __call__(cls):
        return NotImplementedError("__call__ on base DataProvider")

class IpProvider(IonDataProvider, name='ionisation_potential'):    
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

class PhionProvider(IonDataProvider, name='photoionisation_cross_section'):
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

class RadiativeRecombProvider(IonDataProvider, name='radiative_recombination'):
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

class DielectronicRecombProvider(IonDataProvider, name='dielectronic_recombination'):
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

# all instances of IonDataProvider must be defined before this line
IonDataProvider.load_data()
