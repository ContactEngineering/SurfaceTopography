import numpy as np
from ContactMechanics import PeriodicFFTElasticHalfSpace

from SurfaceTopography.Models.SelfAffine import (
    SelfAffinePSD,

    )
import pytest

@pytest.mark.parametrize(
"shortcut_wavelength, hurst_exponent",
[
    (2e-6, 0.8),
    (2e-6, 0.5),
    (2e-6, 0.3),
    (1e-7, 0.1),
    (1e-7, 0.5),
    (1e-7, 0.9),
    (1e-7, 1.),
]
)
def test_elastic_energy_large_H(shortcut_wavelength, hurst_exponent):

    n_pixels = 1024
    physical_size = .5e-4
    pixel_size = physical_size / n_pixels

    # test rolloff
    model_psd = SelfAffinePSD(**{
             'cr':5e-27,
             'shortcut_wavelength': shortcut_wavelength,
             'rolloff_wavelength': 2e-6,
             'hurst_exponent': hurst_exponent})

    Es = 1e6 / (1-0.5**2)
    roughness = model_psd.generate_roughness(**{
            'seed': 1,
            'n_pixels': n_pixels,
            'pixel_size': pixel_size,
    })

    # deterministic, brute force computation of the elastic energy
    hs = PeriodicFFTElasticHalfSpace(
        nb_grid_pts=roughness.nb_grid_pts,
        young=Es,
        physical_sizes=roughness.physical_sizes)

    forces = hs.evaluate_force(roughness.heights())

    # Elastic energy per surface area
    Eel_brute_force = hs.evaluate_elastic_energy(forces, roughness.heights()) / np.prod(roughness.physical_sizes)

    Eel_analytic = Es / 4 * model_psd.variance_half_derivative()
    print(Eel_brute_force)
    print(Eel_analytic)

    np.testing.assert_allclose(Eel_analytic, Eel_brute_force, rtol=1e-1)



def test_elastic_energy_from_logspaced():

    hurst_exponent = 0.8

    n_pixels = 4096
    physical_size = 4e-4
    pixel_size = physical_size / n_pixels
    shortcut_wavelength = 8 * pixel_size
    Es = 1.

    # test rolloff
    params = {
             'cr': 5e-27,
             'shortcut_wavelength': shortcut_wavelength,
             'rolloff_wavelength': shortcut_wavelength * 8,
             'hurst_exponent': hurst_exponent,
             'seed': 1,
             'n_pixels': n_pixels,
             'pixel_size': pixel_size,
             'n_pixels_fourier_interpolation': n_pixels}

    roughness = generate_roughness(**params)

    # ideal_psd = SelfAffinePSD(**params, longcut_wavelength=n_)

    # import matplotlib.pyplot as plt

    # deterministic, brute force computation of the elastic energy for reference
    hs = PeriodicFFTElasticHalfSpace(
        nb_grid_pts=roughness.nb_grid_pts,
        young=Es,
        physical_sizes=roughness.physical_sizes)

    forces = hs.evaluate_force(roughness.heights())

    # Elastic energy per surface area
    Eel_brute_force = hs.evaluate_elastic_energy(forces, roughness.heights()) / np.prod(roughness.physical_sizes)

    #q_grid, psd_grid = roughness.power_spectrum_from_area(resampling_method=None)
    #Eel_grid = elastic_energy_from_psd(q_grid, psd_grid)

    nb_points_per_decade = 25
    q_log, psd_log = roughness.power_spectrum_from_area(nb_points_per_decade=nb_points_per_decade)

    skip_first = nb_points_per_decade

    Eel_log = elastic_energy_from_psd_logspaced(q_log[slice(skip_first, None)], psd_log[slice(skip_first, None)])

    if False:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.loglog(q_log, psd_log)
        ax.axvline(q_log[skip_first])
        plt.show()

    np.testing.assert_allclose(Eel_log, Eel_brute_force, rtol=0.1)
