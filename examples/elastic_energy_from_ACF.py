# %% [markdown]
# # Reference data
#

# %% [raw]
# file:///Users/antoines/Documents/2201_paper_cf_vs_dalvi/work/220716_dalvi_surfaces/data/dalvi_surfaces.html

# %%
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
from ContactMechanics import PeriodicFFTElasticHalfSpace
from SurfaceTopography import SurfaceContainer
from SurfaceTopography.Models.SelfAffine import (
    SelfAffine,
    )

# %%
rolloff_wavelength, shortcut_wavelength, hurst_exponent = (1e-5, 5e-7, 0.6)

n_pixels = 1024
physical_size = .5e-4
pixel_size = physical_size / n_pixels


model_psd = SelfAffine(**{
            'cr':5e-27,
            'shortcut_wavelength': shortcut_wavelength,
            'rolloff_wavelength': rolloff_wavelength,
            'hurst_exponent': hurst_exponent})

Es = 1e6 / (1-0.5**2)
roughness = model_psd.generate_roughness(**{
        'seed': 1,
        'n_pixels': n_pixels,
        'pixel_size': pixel_size,
        #'unit': 'm'
})
roughness._unit = "m"
fig, ax = plt.subplots()

q, c = roughness.power_spectrum_from_area()
ax.loglog(q,c)
ax.loglog(q, model_psd.power_spectrum_isotropic(q))

# %%
roughness.unit

# %%

# %% [raw]
# !open /Users/antoines/Documents/labbook/221027_elastic_energy_from_ACF.html

# %%
roughness.scale_dependent_slope_from_profile()


# %%

# %%
def variance_half_derivative_from_small_scales_ACF(topography):
    # This is how martin Mueser defines the SDRP elastic energy
    
    r, acf = topography.autocorrelation_from_profile()
    nan_args = np.isnan(acf)
    r = np.delete(r, nan_args)
    acf = np.delete(acf, nan_args)
    print(acf)
    
    # we assume continuity below the smallest distance, i.e. that the slope is constant
    small_scale_contrib = (acf[0] / r[0] )
    
    return r, small_scale_contrib + np.concatenate([[0],scipy.integrate.cumtrapz(acf / r ** 2, x = r)])
    
#def variance_half_derivative_from_large_scales_ACF():


# %%
def container_variance_half_derivative_from_small_scales_ACF(container, unit="m", reliable=True):
    # This is how martin Mueser defines the SDRP elastic energy
    
    r, acf = container.autocorrelation(unit=unit, reliable=reliable)
    nan_args = np.isnan(acf)
    r = np.delete(r, nan_args)
    acf = np.delete(acf, nan_args)
    print(acf)
    
    # we assume continuity below the smallest distance, i.e. that the slope is constant
    small_scale_contrib = (acf[0] / r[0] )
    
    return r, small_scale_contrib + np.concatenate([[0],scipy.integrate.cumtrapz(acf / r ** 2, x = r)])


# %%
fig, ax = plt.subplots()


r, hhrms = variance_half_derivative_from_small_scales_ACF(roughness)
ax.plot(r, hhrms, ".")


ax.plot(r, 
[model_psd.variance_derivative(0.5, longcut_wavelength=2*r) for r in r], "*")

# %%
ax.set_xscale("log")
fig

# %%
ax.set_yscale("log")
fig

# %%

# %%
model_psd.variance_half_derivative()

# %%
roughness.variance_half_derivative_via_autocorrelation_from_profile()



# %%
variance_half_derivative_from_small_scales_ACF(roughness)[-1][-1]

# %% [markdown]
# Compare the final values

# %%
model_psd.variance_half_derivative()

# %%
roughness.variance_half_derivative_via_autocorrelation_from_profile()



# %%
variance_half_derivative_from_small_scales_ACF(roughness)[-1][-1]

# %%
from SurfaceTopography import read_container

# %%
uncd = read_container("/Users/antoines/Downloads/ce-5cz7a.zip")[0]

# %%
ncd = read_container("/Users/antoines/roughness_surface_containers/NCD_v1.zip")[0]

# %%
r, acf = ncd.autocorrelation(unit="m")
plt.loglog(r, acf)
nan_args = np.isnan(acf)
r = np.delete(r, nan_args)
acf = np.delete(acf, nan_args)
print(acf)

# we assume continuity below the smallest distance, i.e. that the slope is constant
small_scale_contrib = (acf[0] / r[0] )

sdrp_hhrms =  small_scale_contrib + np.concatenate([[0],scipy.integrate.cumtrapz(acf / r ** 2, x = r)])

# %%
from SurfaceTopography.Models.SelfAffine import SelfAffine

# %%
import dtoolcore

# %% [markdown]
# Parameters of the representative simulation

# %%
backend="s3://frct-simdata/"
uuid = "fe502bac-3708-42ca-be20-bc48bc79bd57"
from ruamel import yaml
# param of the simu
dataset = dtoolcore.DataSet.from_uri(backend + uuid)
readme = yaml.load(dataset.get_readme_content())

rp = readme["parameters"]["roughness"]
rp

# %%
from Adhesion.ReferenceSolutions import JKR

# %%
physical_size = rp["pixel_size"] * rp["n_pixels"]

# handling nondim 

w_SI = 0.055 # chosen to fit the 0.7MPa data with the analytical model
R_SI = 1.17 *1e-3
E_SI = 0.69 * 1e6
Es_SI  = E_SI / (1 - 0.5 **2)

length_unit = JKR.radius_unit(R_SI, Es_SI, w_SI)
psd2d_unit = JKR.height_unit(R_SI, Es_SI, w_SI) ** 2  * JKR.radius_unit(R_SI, Es_SI, w_SI)**2


model_psd = SelfAffine( 
**{'cr': 1.840883412060412e-07 * psd2d_unit ,
 'shortcut_wavelength': 0.00125 * length_unit,
 'rolloff_wavelength': 0.034618699988149 * length_unit,
 'hurst_exponent': 0.8,
  },
    longcut_wavelength=physical_size * length_unit)

# %% [markdown]
# ### The PNAS PSD is ok 

# %%
#plt.loglog(*roughness.autocorrelation_from_area())

fig, ax = plt.subplots()
plt.loglog(*ncd.power_spectrum(unit="m"), label="coneng")

q, C1D = psd_data[1:,0], psd_data[1:,1]
Ciso = C1D_to_Ciso(q, C1D)[1]
plt.loglog(q, C1D, label="PNAS")

plt.loglog(q, model_psd.power_spectrum_profile(q),"+", label="model PSD")

ax.set_ylabel("C1D")
ax.set_xlabel("q")
ax.legend()

# %%
#plt.loglog(*roughness.autocorrelation_from_area())

fig, ax = plt.subplots()


q, C1D = psd_data[1:,0], psd_data[1:,1]
Ciso = C1D_to_Ciso(q, C1D)[1]
plt.loglog(q, Ciso, label="PNAS")

plt.loglog(q, model_psd.power_spectrum_isotropic(q),"+", label="model PSD")

ax.set_ylabel("Ciso")
ax.set_xlabel("q")
ax.legend()

# %% [raw]
# r, acf = ncd.autocorrelation(unit="m")
# nan_args = np.isnan(acf)
# r = np.delete(r, nan_args)
# acf = np.delete(acf, nan_args)
#
#
#
# # we assume continuity below the smallest distance, i.e. that the slope is constant
# small_scale_contrib = (acf[0] / r[0] )
#
# r = r[]
# sdrp_hhrms =  small_scale_contrib + np.concatenate([[0],scipy.integrate.cumtrapz(acf / r ** 2, x = r)])
#

# %%
fig, ax = plt.subplots()

unit_factor = 1e9



r, sdrp_hhrms = container_variance_half_derivative_from_small_scales_ACF(ncd, unit="m")

ax.plot(r, sdrp_hhrms * unit_factor, ".")
ax.set_xscale("log")
r, sdrp_hhrms = container_variance_half_derivative_from_small_scales_ACF(ncd, unit="m", reliable=False)

ax.plot(r, sdrp_hhrms * unit_factor, "xr")
ax.set_xscale("log")

### COMPUTE FROM PNAS PSD

datadir = "/Users/antoines/roughness_surface_containers/DataForArticleInPNAS/Data/Topograpghy data"

def C1D_to_Ciso(q, C1D, qs=2 * np.pi /(0.4e-9)): # m
    Ciso = C1D * np.pi / (q * np.sqrt(1 - (q / qs)**2))
    return q, Ciso

def parse_comma_decimal(s):
    return float(s.replace(b",", b"."))

def load_csv(file):
    return np.loadtxt(file,
           delimiter=";",converters={0: parse_comma_decimal, 1:parse_comma_decimal },)




surface = "NCD"
psd_data = load_csv(f"{datadir}/{surface}.csv")
#ax_psd.plot(psd_data[:,0], psd_data[:,1], label=surface, c=c)

q, C1D = psd_data[1:,0], psd_data[1:,1]

Ciso = C1D_to_Ciso(q, C1D)[1]
# the first wavevectors in the experimental data are not logspaced and we hence need to skip them
sl = slice(5, -1)

# we will integrate from small to large scales
q = q[sl]
Ciso = Ciso[sl]
q = q[::-1]
Ciso = Ciso[::-1]

moment_order = 1
power = moment_order  + 1

ax.plot( np.pi / q[1:],
           - scipy.integrate.cumtrapz(Ciso * q ** power / (2 * np.pi), q) * unit_factor,
        "+"
           )

# Using the model PSD for the simulation: 

ax.plot(np.pi / q, [unit_factor * model_psd.variance_derivative(order=0.5, longcut_wavelength=np.pi / q) for q in q])


# testing numerical integration
Ciso = model_psd.power_spectrum_isotropic(q)
moment_order = 1
power = moment_order  + 1

ax.plot( np.pi / q[1:],
           - scipy.integrate.cumtrapz(Ciso * q ** power / (2 * np.pi), q) * unit_factor,
        "+", color="orange", label = "model PSD via num integrated Ciso"
           )


# texting going thrue C1D on the model PSD 
C1D = model_psd.power_spectrum_profile(q)
Ciso = C1D_to_Ciso(q, C1D)[1]
moment_order = 1
power = moment_order  + 1

ax.plot( np.pi / q[1:],
           - scipy.integrate.cumtrapz(Ciso * q ** power / (2 * np.pi), q) * unit_factor,
        "+r", label = "model PSD via Ciso from C1D"
           )

roughness = model_psd.generate_roughness( **{'n_pixels': 2048,
 'pixel_size': 0.000625 * length_unit}, seed=1)
r, hhrms = variance_half_derivative_from_small_scales_ACF(roughness)
ax.plot(r, hhrms * unit_factor, ".")

ax.legend()

# %% [markdown]
# ### The ACF of the model PSD is ok, but a factor 2 hhigher than the real one

# %%
fig,ax = plt.subplots()

plt.loglog(*roughness.autocorrelation_from_area(), "+")

r, acf = ncd.autocorrelation(unit="m",)
plt.loglog(r, acf, ".", c="red")

    
plt.savefig("test_acf.pdf")

# %%
ncd.autocorrelation(unit = "m")  / roughness.rms_height_from_profile()**2 

# %% [markdown]
# ## The reason is not the drop of the ACF due to the detrending, but cutting the drop of the ACF away makes a difference on the computed elstic energy

# %%
from SurfaceTopography. Exceptions import NoReliableDataError, UndefinedDataError
from SurfaceTopography.HeightContainer import UniformTopographyInterface
from SurfaceTopography.Support.Regression import resample, resample_radial

def autocorrelation_from_profile(self, reliable=True, resampling_method='bin-average', collocation='log',
                                 nb_points=None, nb_points_per_decade=10):
    r"""
    test
    """  # noqa: E501
    if self.has_undefined_data:
        raise UndefinedDataError('This topography has undefined data (missing data points). Autocorrelation cannot be '
                                 'computed for topographies with missing data points.')

    try:
        nx, ny = self.nb_grid_pts
        sx, sy = self.physical_sizes
    except ValueError:
        nx, = self.nb_grid_pts
        sx, = self.physical_sizes

    p = self.heights()

    if self.is_periodic:
        # Compute height-height autocorrelation function from a convolution
        # using FFT. This is periodic by nature.
        surface_qy = np.fft.fft(p, axis=0)
        C_qy = abs(surface_qy) ** 2  # pylint: disable=invalid-name
        A_xy = np.fft.ifft(C_qy, axis=0).real / nx

        # Convert height-height autocorrelation to height-difference
        # autocorrelation:
        #     <(h(x) - h(x+d))^2>/2 = <h^2(x)> - <h(x)h(x+d)>
        A_xy = A_xy[0] - A_xy

        A = A_xy[:nx // 2]
        A[1:nx // 2] += A_xy[nx - 1:(nx + 1) // 2:-1]
        A /= 2

        r = sx * np.arange(nx // 2) / nx
        max_distance = sx / 2
    else:
        # Compute height-height autocorrelation function. We need to zero
        # pad the FFT for the nonperiodic case in order to separate images.
        surface_qy = np.fft.fft(p, n=2 * nx - 1, axis=0)
        C_qy = abs(surface_qy) ** 2  # pylint: disable=invalid-name
        A_xy = np.fft.ifft(C_qy, axis=0).real

        # Correction to turn height-height into height-difference
        # autocorrelation:
        #     <(h(x) - h(x+d))^2>_d/2 = <h^2(x)>_d - <h(x)h(x+d)>_d
        # but we need to take care about h_rms^2=<h^2(x)>, which in the
        # nonperiodic case needs to be computed only over a subsection of
        # the surface. This is because the average < >_d now depends on d,
        # which determines the number of data points that are actually
        # included into the computation of <h(x)h(x+d)>_d. h_rms^2 needs to
        # be computed over the same data points.
        p_sq = p ** 2
        A0_xy = (p_sq.cumsum(axis=0)[::-1] + p_sq[::-1].cumsum(axis=0)[::-1]) / 2

        # Convert height-height autocorrelation to height-difference
        # autocorrelation
        A = ((A0_xy - A_xy[:nx]).T / (nx - np.arange(nx))).T

        r = sx * np.arange(nx) / nx
        max_distance = sx

    if self.is_periodic:
        long_cutoff = max_distance
    else:
        long_cutoff = sx / 4 if reliable else max_distance 
        
    # The factor of two comes from the fact that the short cutoff is estimated
    # from the curvature but the ACF is the slope, see 10.1016/j.apsadv.2021.100190
    if resampling_method is None:
        short_cutoff = self.short_reliability_cutoff()
        if reliable :
            if short_cutoff is None: 
                short_cutoff = 0
            mask = np.logical_and(r > short_cutoff / 2, r < long_cutoff)
            if mask.sum() == 0:
                raise NoReliableDataError('Dataset contains no reliable data.')
            r = r[mask]
            A = A[mask]
        if self.dim == 2:
            return r, A.mean(axis=1)
        else:
            return r, A
    else:
        short_cutoff = self.short_reliability_cutoff(2 * sx / nx) if reliable else 2 * sx / nx
    
        if collocation == 'log':
            # Exclude zero distance because that does not work on a log-scale
            r = r[1:]
            A = A[1:]
        if self.dim == 2:
            r = np.resize(r, (A.shape[1], r.shape[0])).T.ravel()
            A = np.ravel(A)
        r, _, A, _ = resample(r, A, min_value=short_cutoff / 2, max_value=long_cutoff, collocation=collocation,
                              nb_points=nb_points, nb_points_per_decade=nb_points_per_decade, method=resampling_method)
        return r, A
    
UniformTopographyInterface.register_function('autocorrelation_from_profile', autocorrelation_from_profile)


# %%

# %%
fig,ax = plt.subplots()

plt.loglog(*roughness.autocorrelation_from_area(), "+", label="model")
r, acf = ncd.autocorrelation(unit="m", reliable=True, )
plt.plot(r, acf, ".", color="green", label="average reliable")
r, acf = ncd.autocorrelation(unit="m", reliable=False)
plt.loglog(r, acf, ".", c="red")

for i in range(0, len(ncd._topographies), 2):
    r, acf = ncd._topographies[i].to_unit("m").autocorrelation_from_profile(reliable=False)
    plt.loglog(r, acf, c="red", lw=0.5)
    r, acf = ncd._topographies[i].to_unit("m").autocorrelation_from_profile(reliable=True)
    # print( ncd._topographies[i].short_reliability_cutoff())
    # This dataset has no short reliability cutoff
    plt.loglog(r, acf, c="green", lw=0.5)
ax.set_ylabel(r"ACF")
ax.set_xlabel(r"\ell")
plt.savefig("test_acf_without_tilted.pdf")

# %%
fig, ax = plt.subplots()

unit_factor = 1e9



r, sdrp_hhrms = container_variance_half_derivative_from_small_scales_ACF(ncd, unit="m")

ax.plot(r, sdrp_hhrms * unit_factor, ".", color="green", label="coneng acf, reliable")
ax.set_xscale("log")
r, sdrp_hhrms = container_variance_half_derivative_from_small_scales_ACF(ncd, unit="m", reliable=False)

ax.plot(r, sdrp_hhrms * unit_factor, ".", color="red", label="coneng acf, unreliable")
ax.set_xscale("log")

### COMPUTE FROM PNAS PSD

datadir = "/Users/antoines/roughness_surface_containers/DataForArticleInPNAS/Data/Topograpghy data"

def C1D_to_Ciso(q, C1D, qs=2 * np.pi /(0.4e-9)): # m
    Ciso = C1D * np.pi / (q * np.sqrt(1 - (q / qs)**2))
    return q, Ciso

def parse_comma_decimal(s):
    return float(s.replace(b",", b"."))

def load_csv(file):
    return np.loadtxt(file,
           delimiter=";",converters={0: parse_comma_decimal, 1:parse_comma_decimal },)

surface = "NCD"
psd_data = load_csv(f"{datadir}/{surface}.csv")
#ax_psd.plot(psd_data[:,0], psd_data[:,1], label=surface, c=c)

q, C1D = psd_data[1:,0], psd_data[1:,1]

Ciso = C1D_to_Ciso(q, C1D)[1]
# the first wavevectors in the experimental data are not logspaced and we hence need to skip them
sl = slice(5, -1)

# we will integrate from small to large scales
q = q[sl]
Ciso = Ciso[sl]
q = q[::-1]
Ciso = Ciso[::-1]

moment_order = 1
power = moment_order  + 1

ax.plot( np.pi / q[1:],
           - scipy.integrate.cumtrapz(Ciso * q ** power / (2 * np.pi), q) * unit_factor,
        "+", label="PNAS C1D -> Ciso -> fourier"
           )

# Using the model PSD for the simulation: 

ax.plot(np.pi / q, [unit_factor * model_psd.variance_derivative(order=0.5, longcut_wavelength=np.pi / q) for q in q], label="model PSD, analytical", c="k")


# testing numerical integration
Ciso = model_psd.power_spectrum_isotropic(q)
moment_order = 1
power = moment_order  + 1

ax.plot( np.pi / q[1:],
           - scipy.integrate.cumtrapz(Ciso * q ** power / (2 * np.pi), q) * unit_factor,
        "-", color="orange", label = "model PSD via num integrated Ciso"
           )


# texting going thrue C1D on the model PSD 
C1D = model_psd.power_spectrum_profile(q)
Ciso = C1D_to_Ciso(q, C1D)[1]
moment_order = 1
power = moment_order  + 1

ax.plot( np.pi / q[1:],
           - scipy.integrate.cumtrapz(Ciso * q ** power / (2 * np.pi), q) * unit_factor,
        "-r", label = "model PSD via Ciso from C1D"
           )

roughness = model_psd.generate_roughness( **{'n_pixels': 2048,
 'pixel_size': 0.000625 * length_unit}, seed=1)
r, hhrms = variance_half_derivative_from_small_scales_ACF(roughness)
ax.plot(r, hhrms * unit_factor, ".", c="k", label="synthetic, eel from ACF")

ax.legend(bbox_to_anchor=[1,1,0,0])
ax.set_ylabel("variance half derivative")
ax.set_xlabel("length")
fig.savefig("variance_half_derivative.pdf")

# %% [markdown]
# The ACF of the model PSD is around factor 2 above the ACF from the true data. 

# %%
# !open .

# %%
ncd.autocorrelation(unit="m")[-1] / model_psd.rms_height() ** 2

# %% [markdown]
# Now which ACF is right ? 

# %%
sdrp_hhrms[-1] # 4nm for uncd

# %% [markdown]
# ncd._topographies

# %% [markdown]
# hmmm at the GRC talk I had 40nm

# %% [markdown]
# Test with the NCD numerical model PSD? 
#
# I think it is probably just bad integration/ sampling ....

# %% [markdown]
# ## Test via C1D from Ciso

# %%
# !open .

# %%
