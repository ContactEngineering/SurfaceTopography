import pytest
import numpy as np

from SurfaceTopography import UniformLineScan


@pytest.fixture
def uniform_line_scan():
    x = np.linspace(0, 4 * np.pi, 101)
    h = np.sin(x)
    return UniformLineScan(h, 4 * np.pi).scale(2.0)


@pytest.mark.parametrize(['use_scale', 'use_detrend'],
                         [(True, True), (True, False), (False, True), (False, False)])
def test_api_uniform_line_scan(uniform_line_scan, use_scale, use_detrend):
    """Just check whether an expected subset of pipeline functions can be executed.

    Parameters
    ----------
    uniform_line_scan
    use_scale: bool
        If True, check for a scaled topography.
    use_detrend: bool
        If True, check for a detrended topography.
        If `use_scale` is True, the detrending is applied afterwards.
    """
    if use_scale:
        uniform_line_scan = uniform_line_scan.scale(2)
    if use_detrend:
        uniform_line_scan = uniform_line_scan.detrend('height')

    expected_attributes = [
        "rms_height_from_profile",
        "rms_curvature_from_profile",
        "power_spectrum_from_profile"
    ]

    for attr in expected_attributes:
        assert hasattr(uniform_line_scan, attr)
