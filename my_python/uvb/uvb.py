import numpy as np
import unyt as u

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

u.define_unit('Ryd', Ion('H I').ionisation_potential)
u.define_unit('photon', 1.0 * u.dimensionless)

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
    def __init__(self, spectrum_id, redshift, scale=1.0, alpha=0.0):
        """
        Construct an object representing a UVB spectrum.
        Parameters:
        spectrum_id, str: The identifier for the desired spectrum, which must be
        one of the values returned by `list_available_spectra`
        z, float: The redshift at which to interpolate the spectrum
        scale, float or str: The multiplicative scale factor to apply to the spectrum
        (default: 1.0). If 'gamma' and `alpha` is not 0, scale the UVB after the slope
        parameter has been applied such that the photoionisation rate is equal to that
        of the original spectrum.
        alpha, float: The slope parameter alpha_UV, defined as in Crighton+ 2015 (default: 0.0)
        """
        self.z = redshift
        self.name = spectrum_id
        self.scale = scale
        self.alpha = alpha

        if type(scale) is str and scale != 'gamma':
            raise ValueError(f"Invalid value {scale} provided for scale (must be `gamma`)")

        self._nu = None
        self._Jnu = None

    def _load(self, z):
        try:
            filename = _filenames[self.name]
        except KeyError:
            raise ValueError(f"Unknown spectrum {self.name}")

        nu, Jnu = _load_cuba(filename, z)
        self._nu = nu
        self._Jnu = Jnu

        if self.alpha != 0.0:
            e0, e1 = np.array([1, 10]) * u.Ryd
            earr = nu.to_equivalent('Ryd', 'spectral')
            rge0 = (earr > e0) & (earr <= e1)
            rge1 = earr > e1
            idx1 = np.searchsorted(earr, e1)
            
            self._Jnu[rge0] *= (earr[rge0] / e0)**self.alpha
            self._Jnu[rge1] *= (e1 / e0)**self.alpha
            
        if self.scale == 'gamma':
            # Calculate the photoionisation rate for the alpha=0 spectrum, and
            # rescale this spectrum such that the rate is unchanged
            gamma_fid = UVB(self.name, self.z).photoionisation_rate()

            # must set this to recalc photoionisation rate, but it will be
            # overwritten after scale has been calculated
            self._Jnu_interp = interp1d(self._nu, self._Jnu) 
            gamma_new = self.photoionisation_rate()
            self.scale = gamma_fid / gamma_new

        self._Jnu *= self.scale
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

    def photoionisation_rate(self, ion='H I', nu_th=None, nsteps=100000):
        """
        Return the photoionisation rate for a given ion species.
        
        Parameters:
        ion, str: The ion to calculate the ionisation rate for (default: HI)
        nu_th, float or `unyt.Quantity`: The frequency lower bound (default: threshold frequency for chosen ion)
        nsteps, int: Number of frequency intervals to use to integrate the UVB over (default: 100000)
        """

        ion_obj = Ion(ion)

        if 'phion_xsec_params' not in ion_obj.available_fields:
            raise ValueError(f"Cannot calculate ionisation rate for ion {ion}")

        if nu_th is None:
            nu_th = ion_obj.ionisation_potential
        elif type(nu_th) is float:
            nu_th *= u.Hz
        nu_th = nu_th.to_equivalent('Hz', 'spectral')

        nu_values = np.geomspace(nu_th, self.tab_nu.max(), nsteps)
        Jnu_values = self.spectrum(nu_values)
        xsec_values = ion_obj.photoionisation_cross_section(nu_values)

        gamma_integrand = Jnu_values * xsec_values / (u.h * nu_values)
        return (4 * np.pi * u.sr * np.trapz(gamma_integrand, nu_values)).in_units(u.s**-1)

    def photon_flux(self, nu_min=None, nu_max=None, nsteps=100000):
        """
        Return the photon flux phi = \int_{\nu_0}^{\nu_1} J_nu / (h * nu) d\nu.

        Parameters:
        nu_min, float or `unyt.Quantity`: The lower limit for the integration (default: lower limit of spectrum)
        nu_max, float or `unyt.Quantity`: The upper limit for the integration (default: upper limit of spectrum)
        nsteps, int: Number of frequency intervals to use to integrate the UVB over (default: 100000)
        """
        
        if type(nu_min) is float:
            nu_min *= u.Hz
        if type(nu_max) is float:
            nu_max *= u.Hz

        if nu_min is None or nu_min < self.tab_nu.min():
            nu_min = self.tab_nu.min()
        if nu_max is None or nu_max > self.tab_nu.max():
            nu_max = self.tab_nu.max()

        nu_min = nu_min.to_equivalent('Hz', 'spectral')
        nu_max = nu_max.to_equivalent('Hz', 'spectral')

        nu_values = np.geomspace(nu_min, nu_max, nsteps)
        Jnu_values = self.spectrum(nu_values)

        phi_integrand = Jnu_values / (u.h * nu_values)
        return np.trapz(phi_integrand, nu_values).in_units(u.photon * u.cm**-2 * u.s**-1 * u.sr**-1)

    def photon_weighted_cross_section(self, ion='H I', nu_th=None, nsteps=100000):
        """
        Return the photon-weighted average of the photoionisation cross-section for a given ion species.

        Parameters:
        ion, str: The ion to calculate the ionisation rate for (default: HI)
        nu_th, float or `unyt.Quantity`: The frequency lower bound (default: threshold frequency for chosen ion)
        nsteps, int: Number of frequency intervals to use to integrate the UVB over (default: 100000)
        """

        ion_obj = Ion(ion)

        if 'phion_xsec_params' not in ion_obj.available_fields:
            raise ValueError(f"Cannot calculate ionisation rate for ion {ion}")

        if nu_th is None:
            nu_th = ion_obj.ionisation_potential
        elif type(nu_th) is float:
            nu_th *= u.Hz
        nu_th = nu_th.to_equivalent('Hz', 'spectral')

        nu_values = np.geomspace(nu_th, self.tab_nu.max(), nsteps)
        Jnu_values = self.spectrum(nu_values)
        xsec_values = ion_obj.photoionisation_cross_section(nu_values)
        
        phi_values = Jnu_values / (u.h * nu_values)
        xsec_integrand = xsec_values * phi_values
        return (np.trapz(xsec_integrand, nu_values) / np.trapz(phi_values, nu_values)).in_units(u.cm**2)
