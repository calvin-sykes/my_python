import numpy as np
from scipy.interpolate import interp1d
import unyt as u

from ..ions import Ion

import os

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

class UVB:
    # class to encapsulate a UVB spectrum at given z
    def __init__(self, spectrum, redshift, scale=1.0, alpha=0.0):
        """
        Construct a UVB spectrum object.
        spectrum, str:
            The identifier for the desired spectrum, which must be one of the values returned by `self.list_available_spectra`.
        redshift, float:
            The redshift at which to interpolate the spectrum.
        scale, float or str:
            The multiplicative scale factor to apply to the spectrum (default: 1.0). If the string 'gamma' and `alpha` is not 0,
            scale the UVB after the slope parameter has been applied such that the photoionisation rate is equal to that of the original spectrum.
        alpha, float:
            The slope parameter alpha_UV, defined as in Crighton+ 2015 (default: 0.0)
        """
        self.z = redshift
        self.name = spectrum
        self.scale = scale
        self.alpha = alpha

        if type(scale) is str and scale != 'gamma':
            raise ValueError(f"Invalid value {scale} provided for scale (must be `gamma`)")
        
        self._nu = None
        self._Jnu = None

    def __str__(self):
        return f"{self.name} spectrum @ z={self.z}"

    def __repr__(self):
        return f"UVB({self.name}, {self.z}, {self.scale}, {self.alpha})"

    _filenames = {
        'HM01': 'HM2001.txt',
        'HM12': 'HM2012.txt',
        'MH15': 'MH2015.txt'
    }

    @classmethod
    def list_available_spectra(cls):
        return list(cls._filenames.keys())

    def _load(self, z):
        try:
            filename = self._filenames[self.name]
        except KeyError as e:
            raise ValueError(f"Spectrum {self.name} is unrecognised") from None

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
        """Return the tavbulated array of wavelengths."""
        if self._nu is None:
            self._load(self.z)
        return self._nu

    @property
    def tab_spectrum(self):
        """Return the tabulated array of specific intensities."""
        if self._Jnu is None:
            self._load(self.z)
        return self._Jnu

    @staticmethod
    def _handle_ion_arg(ion):
        """If arg is str, try to construct an `Ion` from it. If it is already an ion, just return it."""
        if isinstance(ion, str):
            ion = Ion(ion)
        elif not isinstance(ion, Ion):
            raise TypeError(f"expected str or Ion, got {type(arg).__name__}")
        return ion

    @staticmethod
    def _handle_nu_arg(nu):
        """
        If arg is float or ndarray, convert it to unyt_quantity or unyt_array with units Hz.
        If it is already a unyt type, return it converted to Hz.
        """
        if isinstance(nu, (u.unyt_quantity, u.unyt_array)):
            return nu.to('Hz', equivalence='spectral')
        else:
            return nu * u.Hz

    def spectrum(self, nu):
        """
        Return the interpolated spectrum at the given frequencies.
        nu_th, float or array-like, or unyt quantity or array-like:
            The frequency values at which to interpolate the spectrum. If value has no units it is interpreted as a
            frequency, otherwise an attempt to convert the units to Hz will be made.
        """
        if self._nu is None:
            self._load(self.z)

        nu = np.asarray(UVB._handle_nu_arg(nu))
        return self._Jnu_interp(nu) * _Jnu_unit

    def photoionisation_rate(self, ion='H I', nu_th=None, nsteps=100000):
        """
        Return the photoionisation rate for the species X, given by
            \Gamma(X) = \int_{\nu_th}^{\nu_{max}} 4 \pi J_\nu \sigma(X, \nu) / (h * \nu) d\nu
        ion, str or `ion.Ion`:
            the ion species to calculate the photoionisation rate for. (default: `Ion('H I')`)
        nu_th, float or array-like, or unyt quantity or array-like:
            The lower bound for the integral over frequency. If value has no units it is interpreted as a frequency,
            otherwise an attempt to convert the units to Hz will be made.
            (default: frequency corresponding to ionisation potential for chosen ion)
        nsteps, int:
            The number of intervals to use for the integration.
        """
        ion_obj = UVB._handle_ion_arg(ion)

        if nu_th is None:
            nu_th = ion_obj.ionisation_potential.to('Hz', equivalence='spectral')
        nu_th = UVB._handle_nu_arg(nu_th)

        nu_values = np.geomspace(nu_th, self.tab_nu.max(), nsteps) * u.Hz # geomspace trashes units :/
        Jnu_values = self.spectrum(nu_values)
        xsec_values = ion_obj.photoionisation_cross_section(nu_values)

        gamma_integrand = Jnu_values * xsec_values / (u.h * nu_values)
        return (4 * np.pi * u.sr * np.trapz(gamma_integrand, nu_values)).in_units(u.s**-1)

    def photoheating_rate(self, ion='H I', nu_th=None, nsteps=100000):
        """
        Return the photoheating rate for the species X, given by:
            \H(X) = \int_{\nu_th}^{\nu_{max}} 4 \pi J_\nu \sigma(X, \nu) / (h * \nu) d\nu
        ion, str or `ion.Ion`:
            the ion species to calculate the photoionisation rate for. (default: `Ion('H I')`)
        nu_th, float or array-like, or unyt quantity or array-like:
            The lower bound for the integral over frequency. If value has no units it is interpreted as a frequency,
            otherwise an attempt to convert the units to Hz will be made.
            (default: frequency corresponding to ionisation potential for chosen ion)
        nsteps, int:
            The number of intervals to use for the integration.
        """
        ion_obj = UVB._handle_ion_arg(ion)

        if nu_th is None:
            nu_th = ion_obj.ionisation_potential.to('Hz', equivalence='spectral')
        nu_th = UVB._handle_nu_arg(nu_th)

        nu_values = np.geomspace(nu_th, self.tab_nu.max(), nsteps) * u.Hz # geomspace trashes units :/
        Jnu_values = self.spectrum(nu_values)
        xsec_values = ion.photoionisation_cross_section(nu_values)

        H_integrand = Jnu_values * xsec_values * (nu_values - nu_th) / nu_values
        return (4 * np.pi * u.sr * np.trapz(H_integrand, nu_values)).in_units(u.eV * u.s**-1)

    def photon_flux(self, nu_min=None, nu_max=None, nsteps=100000):
        """
        Return the integrated photon flux, given by
            \phi = \int_{\nu_{min}}^{\nu_{max}} J_nu / (h * nu) d\nu.
        nu_min, float or `unyt.Quantity`:
            The lower bound for the integral over frequency. If value has no units it is interpreted as a frequency,
            otherwise an attempt to convert the units to Hz will be made. (default: lower limit of spectrum)
        nu_max, float or `unyt.Quantity`:
            The upper bound for the integral over frequency. If value has no units it is interpreted as a frequency,
            otherwise an attempt to convert the units to Hz will be made. (default: upper limit of spectrum)
        nsteps, int:
            Number of frequency intervals to use to integrate the UVB over (default: 100000)
        """
        if nu_min is None:
            nu_min = self.tab_nu.min()
        if nu_max is None:
            nu_max = self.tab_nu.max()

        nu_min = UVB._handle_nu_arg(nu_min)
        nu_max = UVB._handle_nu_arg(nu_max)

        if nu_min < self.tab_nu.min():
            nu_min = self.tab_nu.min()
        if nu_max > self.tab_nu.max():
            nu_max = self.tab_nu.max()

        nu_values = np.geomspace(nu_min, nu_max, nsteps) * u. Hz # geomspace trashes units :/
        Jnu_values = self.spectrum(nu_values)

        phi_integrand = Jnu_values / (u.h * nu_values)
        return np.trapz(phi_integrand, nu_values).in_units(u.photon * u.cm**-2 * u.s**-1 * u.sr**-1)

    def photon_weighted_cross_section(self, ion='H I', nu_th=None, nsteps=100000):
        """
        Return the photon-weighted average of the photoionisation cross-section for a given ion species, given by
            \bar{xsec} = \int_{nu_th}^{\nu_{max}} J_\nu \sigma(\nu) / (h * \nu) d\nu / 
        ion, str or `ion.Ion`:
            The ion to calculate the ionisation rate for (default: HI)
        nu_th, float or array-like, or unyt quantity or array-like:
            The lower bound for the integral over frequency. If value has no units it is interpreted as a frequency,
            otherwise an attempt to convert the units to Hz  will be made.
            (default: frequency corresponding to ionisation potential for chosen ion)
        nsteps, int:
            Number of frequency intervals to use to integrate the UVB over (default: 100000)
        """
        ion_obj = UVB._handle_ion_arg(ion)

        if nu_th is None:
            nu_th = ion_obj.ionisation_potential.to('Hz', equivalence='spectral')
        nu_th = UVB._handle_nu_arg(nu_th)

        nu_values = np.geomspace(nu_th, self.tab_nu.max(), nsteps) * u. Hz # geomspace trashes units :/
        Jnu_values = self.spectrum(nu_values)
        xsec_values = ion_obj.photoionisation_cross_section(nu_values)
        
        phi_values = Jnu_values / (u.h * nu_values)
        xsec_integrand = xsec_values * phi_values
        return (np.trapz(xsec_integrand, nu_values) / np.trapz(phi_values, nu_values)).in_units(u.cm**2)
