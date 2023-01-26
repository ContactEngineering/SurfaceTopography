"""
To compute the moments of the power_spectrum

>>> from SurfaceTopography.Generation import fourier_synthesis
>>> t = fourier_synthesis((256, 256), (256,256), hurst=0.8, rms_height=1, short_cutoff=4, long_cutoff=64)
>>> hrms = t.rms_height_from_profile()
>>> hrms_from_psd = np.sqrt(compute_1d_moment(*t.power_spectrum_from_profile(resampling=None), order=0))
>>> assert abs(hrms_from_psd / hrms - 1)  < 0.1

"""
import numpy as np
import scipy.integrate

def compute_1d_moment(q, C1d, order=1):
    # TODO These 1d moments are wrong
    power = order
    variance = np.trapz(C1d * q ** power, q)
    return variance

def compute_1d_moment_cumulative(q, C1d, order=1):
    power = order
    return scipy.integrate.cumtrapz(C1d * q ** power, q)
def compute_iso_moment(q, Ciso, order=1):
    power = order + 1
    variance = np.trapz(Ciso * q ** power / (2 * np.pi), q)
    return variance
def compute_iso_moment_cumulative(q, Ciso, order=1):
    power = order + 1
    return scipy.integrate.cumtrapz(Ciso * q ** power / (2 * np.pi), q)