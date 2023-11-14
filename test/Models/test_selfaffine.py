#
# Copyright 2022-2023 Antoine Sanner
#
# ### MIT license
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import numpy as np

from SurfaceTopography.Models.SelfAffine import SelfAffine
import pytest


def test_shortcut_derivative():
    rolloff_wavelength, shortcut_wavelength, hurst_exponent = (1e-4, 2e-6, 0.8)
    # derivative_order = 1
    n_pixels = 1024
    physical_size = .5e-4
    pixel_size = physical_size / n_pixels

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    model_psd = SelfAffine(**{
        'cr': 5e-27,
        'shortcut_wavelength': shortcut_wavelength,
        # 'longcut_wavelength': rolloff_wavelength,
        'rolloff_wavelength': rolloff_wavelength,
        'hurst_exponent': hurst_exponent})

    roughness = model_psd.generate_roughness(**{
        'seed': 1,
        'n_pixels': n_pixels,
        'pixel_size': pixel_size,
    })

    r, s = roughness.scale_dependent_slope_from_area()
    ax.plot(r, s ** 2, "--")
    ax.axhline(model_psd.variance_derivative(order=1))
    ax.plot(r, [model_psd.variance_derivative(order=1, shortcut_wavelength=dx * 2) for dx in r])
    ax.set_xscale("log")
    ax.set_yscale("log")

    fig.savefig("test_der.png")
    # TODO: make a quantitative assertion ?


# TODO: unify and
@pytest.mark.parametrize(
    "physical_size, rolloff_wavelength, shortcut_wavelength, hurst_exponent",
    [
        (.5e-4, 2e-6, 2e-6, 0.),  # pure flat PSD, no self-affine region
        # purely self-affine PSD.
        (.5e-4, .5e-4, 1e-6, 0.1),
        (.5e-4, .5e-4, 1e-6, 0.5),
        (.5e-4, .5e-4, 1e-6, 0.9),
        (.5e-4, .5e-4, 1e-6, 1.,),
        # mixture
        (.5e-4, 2e-5, 1e-6, 0.1),
        (.5e-4, 2e-5, 1e-6, 0.5),
        (.5e-4, 2e-5, 1e-6, 0.9),
        (.5e-4, 2e-5, 1e-6, 1.,),
    ])
def test_variance_derivatives(physical_size, rolloff_wavelength, shortcut_wavelength, hurst_exponent):
    """
    We test our formulas for the comutu
    """
    n_pixels = 1024
    pixel_size = physical_size / n_pixels

    # test rolloff
    model_psd = SelfAffine(**{
        'cr': 5e-27,
        'shortcut_wavelength': shortcut_wavelength,
        'rolloff_wavelength': rolloff_wavelength,
        'hurst_exponent': hurst_exponent})

    roughness = model_psd.generate_roughness(**{
        'seed': 1,
        'n_pixels': n_pixels,
        'pixel_size': pixel_size,
    })

    variance_analytical = model_psd.variance_derivative(order=2)
    variance_numerical = roughness.rms_laplacian() ** 2

    print(variance_analytical)
    print(variance_numerical)

    np.testing.assert_allclose(variance_analytical, variance_numerical, rtol=1e-1)

#
# # TODO: Not yet at the right place.
# # TODO: This is a tool to integrate log-spaced PSDs usually extracted from experimental data.
# @pytest.mark.skip
# def test_elastic_energy_from_logspaced():
#
#     hurst_exponent = 0.8
#
#     n_pixels = 4096
#     physical_size = 4e-4
#     pixel_size = physical_size / n_pixels
#     shortcut_wavelength = 8 * pixel_size
#     Es = 1.
#
#     # test rolloff
#     params = {
#              'cr': 5e-27,
#              'shortcut_wavelength': shortcut_wavelength,
#              'rolloff_wavelength': shortcut_wavelength * 8,
#              'hurst_exponent': hurst_exponent,
#              'seed': 1,
#              'n_pixels': n_pixels,
#              'pixel_size': pixel_size,
#              'n_pixels_fourier_interpolation': n_pixels}
#
#     roughness = generate_roughness(**params)
#
#     # ideal_psd = SelfAffine(**params, longcut_wavelength=n_)
#
#     # import matplotlib.pyplot as plt
#
#     # deterministic, brute force computation of the elastic energy for reference
#     hs = PeriodicFFTElasticHalfSpace(
#         nb_grid_pts=roughness.nb_grid_pts,
#         young=Es,
#         physical_sizes=roughness.physical_sizes)
#
#     forces = hs.evaluate_force(roughness.heights())
#
#     # Elastic energy per surface area
#     Eel_brute_force = hs.evaluate_elastic_energy(forces, roughness.heights()) / np.prod(roughness.physical_sizes)
#
#     #q_grid, psd_grid = roughness.power_spectrum_from_area(resampling_method=None)
#     #Eel_grid = elastic_energy_from_psd(q_grid, psd_grid)
#
#     nb_points_per_decade = 25
#     q_log, psd_log = roughness.power_spectrum_from_area(nb_points_per_decade=nb_points_per_decade)
#
#     skip_first = nb_points_per_decade
#
#     Eel_log = elastic_energy_from_psd_logspaced(q_log[slice(skip_first, None)], psd_log[slice(skip_first, None)])
#
#     if False:
#         import matplotlib.pyplot as plt
#         fig, ax = plt.subplots()
#         ax.loglog(q_log, psd_log)
#         ax.axvline(q_log[skip_first])
#         plt.show()
#
#     np.testing.assert_allclose(Eel_log, Eel_brute_force, rtol=0.1)
