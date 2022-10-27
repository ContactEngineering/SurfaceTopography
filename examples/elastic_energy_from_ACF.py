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
shortcut_wavelength, hurst_exponent = (2e-6, 0.8)

# %%
n_pixels = 1024
physical_size = .5e-4
pixel_size = physical_size / n_pixels

# %%
# test rolloff
model_psd = SelfAffine(**{
            'cr':5e-27,
            'shortcut_wavelength': shortcut_wavelength,
            'rolloff_wavelength': 2e-6,
            'hurst_exponent': hurst_exponent})

# %%
Es = 1e6 / (1-0.5**2)
roughness = model_psd.generate_roughness(**{
        'seed': 1,
        'n_pixels': n_pixels,
        'pixel_size': pixel_size,
})

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
pixel_spacing, sdrp_slope = roughness.scale_dependent_slope_from_area()

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

large_scale_elastic_energy = large_scale_slope_integral + flat_cutoff_contribution

# %%
fig, ax = plt.subplots()

ax.plot(pixel_spacing[1:], large_scale_elastic_energy)

# %%
ax.set_xscale("log")
ax.set_yscale("log")
fig

# %%
scipy.integrate.cumtrapz

# %%
0.5 * large_scale_slope_integral[0]

# %%
model_psd.variance_half_derivative()

# %% [markdown]
# fits pretty well indeed

# %%
