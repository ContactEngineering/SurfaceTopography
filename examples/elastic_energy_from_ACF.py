# %% [markdown]
# This is WIP.

# %%
import matplotlib.pyplot as plt

# %%
import numpy as np
import scipy.integrate
from ContactMechanics import PeriodicFFTElasticHalfSpace

# %%
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
})

fig, ax = plt.subplots()

q, c = roughness.power_spectrum_from_area()
ax.loglog(q,c)
ax.loglog(q, model_psd.power_spectrum_isotropic(q))


# %%
# deterministic, brute force computation of the elastic energy
hs = PeriodicFFTElasticHalfSpace(
    nb_grid_pts=roughness.nb_grid_pts,
    young=Es,
    physical_sizes=roughness.physical_sizes)

# %%
forces = hs.evaluate_force(roughness.heights())

# %%
# Elastic energy per surface area
Eel_brute_force = hs.evaluate_elastic_energy(forces, roughness.heights()) / np.prod(roughness.physical_sizes)

# %%
Eel_analytic = Es / 4 * model_psd.variance_half_derivative()
print(Eel_brute_force)
print(Eel_analytic)

# %%
np.testing.assert_allclose(Eel_analytic, Eel_brute_force, rtol=1e-1)

# %% [markdown]
# Computing from the SDRP slope
#

# %%
r, slope = pixel_spacing, sdrp_slope = roughness.scale_dependent_slope_from_area()

# %%
scipy.integrate.trapz(slope**2, x=r)

# %%
r[0] * slope[0]**2

# %%
fig, ax = plt.subplots()

ax.plot(pixel_spacing, sdrp_slope, ".")

ax.axvline(model_psd.rolloff_wavelength, c="k")

# %%
ax.set_xscale("log")
ax.set_yscale("log")

fig

# %% [markdown]
# integrate from large scales to small scales: 

# %%
large_scale_slope_integral = - scipy.integrate.cumtrapz(sdrp_slope[::-1] ** 2, x = pixel_spacing[::-1])[::-1]
flat_cutoff_contribution = (sdrp_slope**2 * pixel_spacing)[:-1] # todo is this the right index ?  

large_scale_variance_half_derivative = 0.5 * (large_scale_slope_integral + flat_cutoff_contribution)

# %%
large_scale_slope_integral

# %%
large_scale_variance_half_derivative[0]

# %%
large_scale_variance_half_derivative[0]

# %%
fig, ax = plt.subplots()

ax.axhline(large_scale_variance_half_derivative[0])
ax.set_xlabel(r"pixel spacing $\ell$")
ax.axhline(model_psd.variance_derivative(order=0.5,), c = "r", ls = "--")
ax.plot(pixel_spacing[1:], large_scale_variance_half_derivative)

ax.plot(pixel_spacing, 
       [model_psd.variance_derivative(order=0.5, shortcut_wavelength= dx * 2 ) for dx in pixel_spacing]
       )


# %%
flat_cutoff_contribution

# %%
model_psd.variance_derivative(order=0.5, shortcut_wavelength= dx * 2 ) 

# %%
ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel(r"pixel spacing $\ell$")
ax.set_ylabel(r" $\ell$")

fig

# %%
scipy.integrate.cumtrapz

# %%
0.5 * large_scale_slope_integral[0]

# %%
model_psd.variance_half_derivative()

# %%
model_psd.variance_derivative(order=0.5)

# %% [markdown]
# fits pretty well indeed

# %% [markdown]
# ## Do the curves collapse for a large enough self-affine region ? 

# %%
rolloff_wavelength, shortcut_wavelength, hurst_exponent = (1e-5, 4e-7, 0.6)

n_pixels = 2048
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
})

fig, ax = plt.subplots()

q, c = roughness.power_spectrum_from_area()
ax.loglog(q,c)
ax.loglog(q, model_psd.power_spectrum_isotropic(q))


# %%
r, slope = pixel_spacing, sdrp_slope = roughness.scale_dependent_slope_from_area()
large_scale_slope_integral = - scipy.integrate.cumtrapz(sdrp_slope[::-1] ** 2, x = pixel_spacing[::-1])[::-1]
flat_cutoff_contribution = (sdrp_slope**2 * pixel_spacing)[:-1] # todo is this the right index ?  

large_scale_variance_half_derivative = 0.5 * (large_scale_slope_integral + flat_cutoff_contribution)

# %%
fig, ax = plt.subplots()

ax.axhline(large_scale_variance_half_derivative[0])
ax.set_xlabel(r"pixel spacing $\ell$")
ax.axhline(model_psd.variance_derivative(order=0.5,), c = "r", ls = "--")
ax.plot(pixel_spacing[1:], large_scale_variance_half_derivative)

ax.plot(pixel_spacing, 
       [model_psd.variance_derivative(order=0.5, shortcut_wavelength= dx * 2 ) for dx in pixel_spacing]
       )

ax.set_xscale("log")

ax.set_xlabel(r"pixel spacing $\ell$")
ax.set_ylabel(r" $\ell$")


# %%

ax.set_yscale("log")
fig

# %% [markdown]
# ## Example on UNCD

# %%
from SurfaceTopography import read_published_container, read_container

# %% [raw]
# read_published_container("https://contact.engineering/manager/surface/1306/")
#

# %% [raw]
# uncd = read_container("/Users/antoines/Downloads/ce-5cz7a.zip")
#

# %% [raw]
# [containers = c for url in 
# [
# "https://contact.engineering/go/wcqj3/",
# "https://contact.engineering/go/8sc7t/",
# "https://contact.engineering/go/cjy6s/",
# "https://contact.engineering/go/mz7z5/",
# ]]

# %%
uncd = read_container("/Users/antoines/Downloads/ce-5cz7a.zip")[0]


# %%
uncd.autocorrelation(unit="m")

# %%
