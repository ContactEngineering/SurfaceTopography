import numpy as np

from SurfaceTopography import read_container, SurfaceContainer
from SurfaceTopography.Generation import fourier_synthesis
from SurfaceTopography.Models.SelfAffine import (
    SelfAffine,
)
import pytest
from NuMPI import MPI

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")


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
def test_variance_half_derivative_from_acf(shortcut_wavelength, hurst_exponent):
    from ContactMechanics import PeriodicFFTElasticHalfSpace

    n_pixels = 2048  # We need a good discretisation
    physical_size = .5e-4
    pixel_size = physical_size / n_pixels

    # test rolloff
    model_psd = SelfAffine(**{
        'cr': 5e-27,
        'shortcut_wavelength': shortcut_wavelength,
        'rolloff_wavelength': 2e-6,
        'hurst_exponent': hurst_exponent})

    Es = 1e6 / (1 - 0.5 ** 2)
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
    Eel_from_acf_profile = Es / 4 * roughness.variance_half_derivative_via_autocorrelation_from_profile()

    Eel_from_acf_area = Es / 4 * roughness.variance_half_derivative_via_autocorrelation_from_area()

    print("brute force contact mechanics:", Eel_brute_force)
    print("analytic from PSD:", Eel_analytic)
    print("profile ACF:", Eel_from_acf_profile)
    print("areal ACF:", Eel_from_acf_area)

    np.testing.assert_allclose(Eel_analytic, Eel_brute_force, rtol=1e-1)

    np.testing.assert_allclose(Eel_from_acf_profile, Eel_brute_force, rtol=1e-1)
    np.testing.assert_allclose(Eel_from_acf_area, Eel_brute_force, rtol=1e-1)





@pytest.mark.skip()
def test_variance_half_derivative_from_container(file_format_examples):
    c, = read_container(f'{file_format_examples}/container1.zip')
    # c, = read_container("/Users/antoines/Downloads/ce-5cz7a.zip")
    c.variance_half_derivative_via_autocorrelation_from_profile()
