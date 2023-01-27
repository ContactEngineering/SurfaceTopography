"""
To compute the moments of the power_spectrum

# TODO: Outdated !
>>> from SurfaceTopography.Generation import fourier_synthesis
>>> t = fourier_synthesis((256, 256), (256,256), hurst=0.8, rms_height=1, short_cutoff=4, long_cutoff=64)
>>> hrms = t.rms_height_from_profile()
>>> hrms_from_psd = np.sqrt(compute_1d_moment(*t.power_spectrum_from_profile(resampling=None), order=0))
>>> assert abs(hrms_from_psd / hrms - 1)  < 0.1

"""
import numpy as np
import scipy.integrate


def compute_1d_moment(q, C1d, order=1, cumulative=False):
    power = order
    integ = scipy.integrate.cumtrapz if cumulative else np.trapz
    return integ(C1d * q ** power, q) / np.pi


def compute_iso_moment(q, Ciso, order=1, cumulative=False):
    power = order + 1
    integ = scipy.integrate.cumtrapz if cumulative else np.trapz
    return integ(Ciso * q ** power / (2 * np.pi), q)
    return variance
