#
# Copyright 2020-2023 Lars Pastewka
#           2019-2020 Antoine Sanner
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

"""
Tests reliability cutoff and its use to restrict the range of data in the
analysis pipeline functions.
"""

import os

import numpy as np
import pytest

from NuMPI import MPI

from SurfaceTopography import read_container, read_topography, NonuniformLineScan, UniformLineScan, Topography
from SurfaceTopography.Container.SurfaceContainer import InMemorySurfaceContainer
from SurfaceTopography.Exceptions import NoReliableDataError

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")


def test_scanning_probe_reliability_cutoff(file_format_examples):
    surf = read_topography(os.path.join(file_format_examples, 'di-1.di'))
    np.testing.assert_allclose(surf.scanning_probe_reliability_cutoff(40), 90.700854)

    # Should be None because there is no tip radius information
    assert surf.short_reliability_cutoff() is None

    cut = surf.short_reliability_cutoff(0.2)
    # Should be the maximum of the actual value and the value that was passed
    np.testing.assert_almost_equal(cut, 0.2)


@pytest.mark.parametrize('unit,fac', [('pm', 1e-3), ('nm', 1.0), ('um', 1e3), ('mm', 1e6)])
def test_unit_invariance(file_format_examples, unit, fac):
    """Test that we get the same reliability cutoff if we rescale the topography"""
    surf = read_topography(os.path.join(file_format_examples, 'di-1.di'))

    ref_value = 90.700854

    np.testing.assert_allclose(surf.to_unit(unit).scanning_probe_reliability_cutoff(40 / fac) * fac, ref_value,
                               rtol=1e-3)


def test_tip_radius_reliability_cutoff_from_instrument_metadata(file_format_examples):
    surf = read_topography(os.path.join(file_format_examples, 'di-1.di'), info={
        'instrument': {
            'parameters': {
                'tip_radius': {
                    'value': 40,
                    'unit': 'nm',
                }
            }
        }
    })
    dois = set()
    cut = surf.short_reliability_cutoff(dois=dois)
    assert dois == {'10.1016/j.apsadv.2021.100190', '10.1088/2051-672X/ac860a'}
    np.testing.assert_allclose(cut, 90.700854)

    # Make sure PSD returns only reliable portion
    q, _ = surf.power_spectrum_from_profile()
    assert q[-1] < 2 * np.pi / cut

    q, _ = surf.power_spectrum_from_area()
    assert q[-1] < 2 * np.pi / cut

    # Make sure ACF returns only reliable portion
    r, A = surf.autocorrelation_from_profile()
    assert r[0] >= cut / 2

    r, A = surf.autocorrelation_from_area()
    assert r[0] >= cut / 2

    # Make sure SDRP returns only reliable portion
    r, s = surf.scale_dependent_statistical_property(lambda x, y=None: np.mean(x * x))
    assert r[0] >= cut / 2


def test_resolution_reliability_cutoff_from_instrument_metadata(file_format_examples):
    resolution = 70
    surf = read_topography(os.path.join(file_format_examples, 'di-1.di'), info={
        'instrument': {
            'parameters': {
                'resolution': {
                    'value': resolution,
                    'unit': 'nm',
                }
            }
        }
    })
    cut = surf.short_reliability_cutoff()
    np.testing.assert_almost_equal(cut, resolution)

    # Make sure PSD returns only reliable portion
    q, _ = surf.power_spectrum_from_profile()
    assert q[-1] < 2 * np.pi / cut

    q, _ = surf.power_spectrum_from_area()
    assert q[-1] < 2 * np.pi / cut

    # Make sure ACF returns only reliable portion
    r, A = surf.autocorrelation_from_profile()
    assert r[0] >= cut / 2

    r, A = surf.autocorrelation_from_area()
    assert r[0] >= cut / 2

    # Make sure SDRP returns only reliable portion
    r, s = surf.scale_dependent_statistical_property(lambda x, y=None: np.mean(x * x))
    assert r[0] >= cut / 2


def test_reliability_cutoff_line_scan(file_format_examples):
    surf = read_topography(os.path.join(file_format_examples, 'xy-4.txt'), unit='um', info={
        'instrument': {
            'parameters': {
                'tip_radius': {
                    'value': 40,
                    'unit': 'nm',
                }
            }
        }
    })
    cut = surf.short_reliability_cutoff()
    np.testing.assert_allclose(cut, 0.126519, atol=1e-6)

    cut = surf.to_nonuniform().short_reliability_cutoff()
    # This differs from the above because the derivatives are computed at slightly different locations
    np.testing.assert_allclose(cut, 0.126527, atol=1e-6)

    cut = surf.to_nonuniform().short_reliability_cutoff(0.2)
    # Should be the maximum of the actual value and the value that was passed
    np.testing.assert_allclose(cut, 0.2)

    cut = surf.to_nonuniform().short_reliability_cutoff(0.1)
    # Should be the maximum of the actual value and the value that was passed
    np.testing.assert_allclose(cut, 0.126527, atol=1e-6)


def test_problem1(file_format_examples):
    surf = read_topography(os.path.join(file_format_examples, 'di-6.di'), info={
        'instrument': {
            'parameters': {
                'tip_radius': {
                    'value': 26,
                    'unit': 'nm',
                }
            }
        }
    })
    assert surf.short_reliability_cutoff() is None


def test_no_reliable_data_uniform():
    t = UniformLineScan([-0.16666667, -0.16666667, -0.16666667, 0.83333333, -0.16666667, -0.16666667, -0.16666667], 6,
                        unit='nm',
                        info=dict(instrument={'name': 'Bla',
                                              'type': 'microscope-based',
                                              'parameters': {'resolution': {'unit': 'µm', 'value': 10.0}}}))

    with pytest.raises(NoReliableDataError):
        t.power_spectrum_from_profile()

    with pytest.raises(NoReliableDataError):
        t.power_spectrum_from_profile(resampling_method=None)

    with pytest.raises(NoReliableDataError):
        t.autocorrelation_from_profile()

    with pytest.raises(NoReliableDataError):
        t.autocorrelation_from_profile(resampling_method=None)

    with pytest.raises(NoReliableDataError):
        t.variable_bandwidth_from_profile()

    with pytest.raises(NoReliableDataError):
        t.scale_dependent_statistical_property(lambda x: np.mean(x * x), n=1)

    c = InMemorySurfaceContainer([t])
    with pytest.raises(NoReliableDataError):
        c.power_spectrum(unit='um')

    with pytest.raises(NoReliableDataError):
        c.autocorrelation(unit='um')

    with pytest.raises(NoReliableDataError):
        c.variable_bandwidth(unit='um')

    with pytest.raises(NoReliableDataError):
        c.scale_dependent_statistical_property(lambda x: np.mean(x * x), n=1, unit='um')


def test_no_reliable_data_topography():
    t = Topography(
        np.array([[-0.16666667, -0.16666667, -0.16666667, 0.83333333, -0.16666667, -0.16666667, -0.16666667]] * 6),
        (6, 6),
        unit='nm',
        info=dict(instrument={'name': 'Bla',
                              'type': 'microscope-based',
                              'parameters': {'resolution': {'unit': 'µm', 'value': 10.0}}}))

    with pytest.raises(NoReliableDataError):
        t.power_spectrum_from_area()

    with pytest.raises(NoReliableDataError):
        t.autocorrelation_from_area()

    with pytest.raises(NoReliableDataError):
        t.variable_bandwidth_from_area()

    with pytest.raises(NoReliableDataError):
        t.scale_dependent_statistical_property(lambda x, y: np.mean(x * x + y * y), n=1)


def test_no_reliable_data_nonuniform():
    t = NonuniformLineScan([0., 1., 2., 3.5, 4., 5., 6.],
                           [-0.16666667, -0.16666667, -0.16666667, 0.83333333, -0.16666667, -0.16666667, -0.16666667],
                           unit='nm',
                           info=dict(instrument={'name': 'Bla',
                                                 'type': 'microscope-based',
                                                 'parameters': {'resolution': {'unit': 'µm', 'value': 10.0}}}))

    with pytest.raises(NoReliableDataError):
        t.power_spectrum_from_profile()

    with pytest.raises(NoReliableDataError):
        t.power_spectrum_from_profile(resampling_method=None)

    with pytest.raises(NoReliableDataError):
        t.autocorrelation_from_profile()

    with pytest.raises(NoReliableDataError):
        t.autocorrelation_from_profile(resampling_method=None)

    with pytest.raises(NoReliableDataError):
        t.variable_bandwidth_from_profile()

    with pytest.raises(NoReliableDataError):
        t.scale_dependent_statistical_property(lambda x: np.mean(x * x), n=1)

    c = InMemorySurfaceContainer([t])
    with pytest.raises(NoReliableDataError):
        c.power_spectrum(unit='um')

    with pytest.raises(NoReliableDataError):
        c.autocorrelation(unit='um')

    with pytest.raises(NoReliableDataError):
        c.variable_bandwidth(unit='um')

    with pytest.raises(NoReliableDataError):
        c.scale_dependent_statistical_property(lambda x: np.mean(x * x), n=1, unit='um')


def test_linear_2d_small_tip():
    t = Topography(np.array([[9, 9, 9, 9, 9],
                             [7, 7, 7, 7, 7],
                             [5, 5, 5, 5, 5],
                             [3, 3, 3, 3, 3],
                             [1, 1, 1, 1, 1],
                             [-1, -1, -1, -1, -1],
                             [-3, -3, -3, -3, -3],
                             [-5, -5, -5, -5, -5],
                             [-7, -7, -7, -7, -7],
                             [-9, -9, -9, -9, -9]]).T,
                   (1, 2), unit='um', info={
            'instrument': {
                'parameters': {
                    'tip_radius': {
                        'value': 26,
                        'unit': 'nm',
                    }
                }
            }}).detrend('center')

    # This has zero curvature, so everything should be reliable
    assert t.short_reliability_cutoff() is None

    q, C = t.power_spectrum_from_profile()
    assert np.isfinite(C).sum() > 0

    q, C = t.transpose().power_spectrum_from_profile()
    assert np.isfinite(C).sum() > 0

    q, C = t.power_spectrum_from_area()
    assert np.isfinite(C).sum() > 0


def test_linear_2d_large_tip():
    t = Topography(np.array([[9, 9, 9, 9, 9],
                             [7, 7, 7, 7, 7],
                             [5, 5, 5, 5, 5],
                             [3, 3, 3, 3, 3],
                             [1, 1, 1, 1, 1],
                             [-1, -1, -1, -1, -1],
                             [-3, -3, -3, -3, -3],
                             [-5, -5, -5, -5, -5],
                             [-7, -7, -7, -7, -7],
                             [-9, -9, -9, -9, -9]]).T,
                   (1, 2), unit='um', info={
            'instrument': {
                'parameters': {
                    'tip_radius': {
                        'value': 10,
                        'unit': 'mm',
                    }
                }
            }}).detrend('center')

    # This has zero curvature, so everything should be reliable
    assert t.short_reliability_cutoff() is None

    q, C = t.power_spectrum_from_profile()
    assert np.isfinite(C).sum() > 0

    q, C = t.transpose().power_spectrum_from_profile()
    assert np.isfinite(C).sum() > 0

    q, C = t.power_spectrum_from_area()
    assert np.isfinite(C).sum() > 0


def test_partially_reliable_data_container(file_format_examples):
    c, = read_container(f'{file_format_examples}/container-1.zip')
    c = c.read_all()  # read everything to memory so we can patch info dict

    # Patch info dictionary
    c[0]._info['instrument'] = {'parameters': {'tip_radius': {'value': 10, 'unit': 'um'}}}
    c[1]._info['instrument'] = {'parameters': {'tip_radius': {'value': 10, 'unit': 'um'}}}
    c[2]._info['instrument'] = {'parameters': {'tip_radius': {'value': 10, 'unit': 'um'}}}

    # Check that we raise NoReliableDataError for one of the topographies
    c[0].power_spectrum_from_profile()
    c[1].power_spectrum_from_profile()
    with pytest.raises(NoReliableDataError):
        c[2].power_spectrum_from_profile()

    # This should raise no error
    c.power_spectrum(unit='um')

    # Patch info dictionary such that all data is unreliable
    c[0]._info['instrument'] = {'parameters': {'tip_radius': {'value': 10, 'unit': 'mm'}}}
    c[1]._info['instrument'] = {'parameters': {'tip_radius': {'value': 10, 'unit': 'mm'}}}
    c[2]._info['instrument'] = {'parameters': {'tip_radius': {'value': 10, 'unit': 'mm'}}}

    # Check that we raise NoReliableDataError for one of the topographies
    with pytest.raises(NoReliableDataError):
        c[0].power_spectrum_from_profile()
    with pytest.raises(NoReliableDataError):
        c[1].power_spectrum_from_profile()
    with pytest.raises(NoReliableDataError):
        c[2].power_spectrum_from_profile()

    # This should now raise a NoReliableDataError
    with pytest.raises(NoReliableDataError):
        c.power_spectrum(unit='um')
