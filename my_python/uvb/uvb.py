import numpy as np
import unyt as u

from scipy.integrate import romberg as quad
from scipy.interpolate import interp1d

from my_python.ions import Ion

import os

_filenames = {
    'HM01': 'HM2001.txt',
    'HM12': 'HM2012.txt',
    'MH15': 'MH2015.txt'
}

_lda_unit = u.angstrom
_Jnu_unit = u.ergs / (u.s * u.cm**2 * u.Hz * u.sr)

def _load_cuba(filename, z):
    cuba_file = open(os.path.join(os.path.dirname(__file__), filename))

    lda = []
    Jnu = []

    counter = 0
    for line in cuba_file:
        # skip comment lines
        if '#' not in line:
            counter += 1
        else:
            continue
        lsplit = line.split()
        if counter == 1:
            # get redshifts from first line
            zarr = np.array(lsplit).astype(np.float)
        else:
            # each line has nu and then Jnu at each z
            lda.append(float(lsplit[0]))
            Jnu.append(list(map(float, lsplit[1:])))

    # add units, converting wavelengths to frequencies
    nu = np.flipud(np.array(lda) * _lda_unit).to(u.Hz, equivalence='spectral')
    Jnu = np.flipud(np.array(Jnu)) * _Jnu_unit

    # Interpolate along redshift direction
    interp = interp1d(zarr, Jnu, axis=-1)

    return nu, interp(z) * _Jnu_unit

def list_available_spectra():
    return list(_filenames.keys())

class UVB:
    def __init__(self, spectrum, redshift):
        self.z = redshift
        self.spectrum_name = spectrum
        
        self._nu = None
        self._Jnu = None

    def _load(self, z):
        try:
            filename = _filenames[self.spectrum_name]
        except KeyError:
            raise ValueError(f"Unknown spectrum {self.spectrum_name}")

        nu, Jnu = _load_cuba(filename, z)
        self._nu = nu
        self._Jnu = Jnu

        self._Jnu_interp = interp1d(self._nu, self._Jnu)

    @property
    def tab_nu(self):
        """Return the array of wavelengths at which the UVB is tabulated."""
        if self._nu is None:
            self._load(self.z)
        return self._nu

    @property
    def tab_spectrum(self):
        """Return the array of specific intensities."""
        if self._Jnu is None:
            self._load(self.z)
        return self._Jnu

    def spectrum(self, nu):
        """Return the spectrum at frequencies nu."""

        if self._nu is None:
            self._load(self.z)

        nu = np.atleast_1d(nu)
        if type(nu) is np.ndarray:
            nu = nu * u.Hz

        return self._Jnu_interp(nu) * _Jnu_unit

    def HI_photoionisation_rate(self, nu_th=None, nsteps=100000):
        """Return the photoionisation rate for HI."""

        hydrogen = Ion('H I')

        if nu_th is None:
             nu_th = hydrogen.ionisation_potential.to('Hz', equivalence='spectral')

        nu_values = np.geomspace(nu_th, self.tab_nu.max(), nsteps)
        Jnu_values = self.spectrum(nu_values)
        xsec_values = hydrogen.photoionisation_cross_section(nu_values)

        gamma_integrand = Jnu_values * xsec_values / (u.h * nu_values)
        return (4 * np.pi * u.sr * np.trapz(gamma_integrand, nu_values)).in_units(u.s**-1)
