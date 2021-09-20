import pytest
import numpy as np

from SurfaceTopography import UniformLineScan, NonuniformLineScan, Topography

common_properties = [
    "communicator",
    "dim",
    "info",
    "is_uniform",
    "is_periodic",
    "is_reentrant",
    "nb_grid_pts",
    "physical_sizes",
    "unit",
]

uniform_properties = [
    "area_per_pt",
    "has_undefined_data",
    "is_domain_decomposed",
    "pixel_size",
]

nonuniform_properties = [
    "x_range",
]

properties = common_properties + uniform_properties + nonuniform_properties

common_functions = [
    "bandwidth",
    "detrend",
    "derivative",
    "heights",
    "max",
    "mean",
    "min",
    "positions",
    "positions_and_heights",
    "scale",
    "squeeze",
    "scale_dependent_statistical_property",
    "to_unit",
    "to_netcdf",
]

uniform_functions = [
    "filter",
    "interpolate_bicubic",
    "interpolate_fourier",
    "longcut",
    "mirror_stitch",
    "shortcut",
    "window",
]

nonuniform_functions = [
    "to_uniform",
]

profile_functions = [
    "autocorrelation_from_profile",
    "checkerboard_detrend_profile",
    "power_spectrum_from_profile",
    "rms_height_from_profile",
    "rms_slope_from_profile",
    "rms_curvature_from_profile",
    "variable_bandwidth_from_profile",
]

area_functions = [
    "checkerboard_detrend_area",
    "power_spectrum_from_area",
    "power_spectrum_from_profile",
    "rms_height_from_area",
    "rms_gradient",
    "rms_laplacian",
    "rms_curvature_from_area",
    "variable_bandwidth_from_area",
    "power_spectrum_from_area",
    "power_spectrum_from_profile",
    "to_dzi",
]

functions = common_functions + uniform_functions + nonuniform_functions + profile_functions + area_functions


#######################################################################
# Fixtures
#######################################################################


def id_topography_modifiers(fixture_value):
    """Used to see whether scaled or detrended in test output.
    """
    use_scale, use_detrend = fixture_value
    s = ""
    if not use_scale:
        s += "no"
    s += "scale-"
    if not use_detrend:
        s += "no"
    s += "detrend"
    return s


def apply_param(topography, param):
    use_scale, use_detrend = param
    if use_scale:
        topography = topography.scale(2)
        assert hasattr(topography, 'scale_factor')
    if use_detrend:
        topography = topography.detrend('height')
        assert hasattr(topography, 'detrend_mode')
    return topography


@pytest.fixture(params=[(True, True), (True, False), (False, True), (False, False)],
                ids=id_topography_modifiers)
def uniform_line_scan(request):
    """Returns a uniform line scan, with all combinations of scaled and detrended.

    Detrended is always executed after scaling, if requested.

    Returns
    -------
    A uniform line scan.
    """
    x = np.linspace(0, 4 * np.pi, 11)
    h = np.sin(x)
    uniform_line_scan = UniformLineScan(h, 4 * np.pi)
    return apply_param(uniform_line_scan, request.param)


@pytest.fixture(params=[(True, True), (True, False), (False, True), (False, False)],
                ids=id_topography_modifiers)
def nonuniform_line_scan(request):
    """Returns a nonuniform line scan, with all combinations of scaled and detrended.

    Detrended is always executed after scaling, if requested.

    Returns
    -------
    A nonuniform line scan.
    """
    x = np.array((0, 0.1, 0.2, 0.4, 0.5))
    h = np.sin(x)
    nonuniform_line_scan = NonuniformLineScan(x, h)
    return apply_param(nonuniform_line_scan, request.param)


@pytest.fixture(params=[(True, True), (True, False), (False, True), (False, False)],
                ids=id_topography_modifiers)
def uniform_2d_topography(request):
    """Returns a uniform 2D topography, with all combinations of scaled and detrended.

    Detrended is always executed after scaling, if requested.

    Returns
    -------
    A uniform 2D topography.
    """
    y = np.arange(10).reshape((1, -1))
    x = np.arange(5).reshape((-1, 1))
    arr = -2 * y + 0 * x  # only slope in y direction
    topography = Topography(arr, (5, 10)).detrend('center')
    return apply_param(topography, request.param)


#######################################################################
# Tests
#######################################################################

def checkattr(obj, attr):
    assert hasattr(obj, attr)
    if attr in properties:
        assert isinstance(getattr(type(obj), attr), property)
    if attr in functions:
        assert callable(getattr(obj, attr))


@pytest.mark.parametrize('expected_attribute',
                         common_properties + common_functions + uniform_properties + uniform_functions +
                         profile_functions)
def test_api_uniform_line_scan(uniform_line_scan, expected_attribute):
    """Just check whether an expected subset of pipeline functions can be executed.

    Parameters
    ----------
    uniform_line_scan: SurfaceTopography.UniformLineScanAndTopography.UniformLineScan
    expected_attribute: str
        name of the attribute which we check
    """
    checkattr(uniform_line_scan, expected_attribute)


@pytest.mark.parametrize('expected_attribute',
                         common_properties + common_functions + nonuniform_properties + nonuniform_functions +
                         profile_functions)
def test_api_nonuniform_line_scan(nonuniform_line_scan, expected_attribute):
    """Just check whether an expected subset of pipeline functions can be executed.

    Parameters
    ----------
    nonuniform_line_scan: SurfaceTopography.UniformLineScanAndTopography.NonuniformLineScan
    expected_attribute: str
        name of the attribute which we check
    """
    checkattr(nonuniform_line_scan, expected_attribute)


@pytest.mark.parametrize('expected_attribute',
                         common_properties + common_functions + uniform_properties + uniform_functions +
                         profile_functions + area_functions)
def test_api_uniform_2d_topography(uniform_2d_topography, expected_attribute):
    """Just check whether an expected subset of pipeline functions can be executed.

    Parameters
    ----------
    uniform_2d_topography: SurfaceTopography.UniformLineScanAndTopography.NonuniformLineScan
    expected_attribute: str
        name of the attribute which we check
    """
    checkattr(uniform_2d_topography, expected_attribute)
