#
# Copyright 2021 Lars Pastewka
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

from SurfaceTopography import read_topography


def test_scanning_probe_reliability_cutoff(file_format_examples):
    surf = read_topography(os.path.join(file_format_examples, 'di1.di'))
    np.testing.assert_allclose(surf.scanning_probe_reliability_cutoff(40), 90.700854)

    # Should be None because there is no tip radius information
    assert surf.short_reliability_cutoff() is None

    cut = surf.short_reliability_cutoff(0.2)
    # Should be the maximum of the actual value and the value that was passed
    np.testing.assert_almost_equal(cut, 0.2)


def test_tip_radius_reliability_cutoff_from_instrument_metadata(file_format_examples):
    surf = read_topography(os.path.join(file_format_examples, 'di1.di'), info={
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
    surf = read_topography(os.path.join(file_format_examples, 'di1.di'), info={
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
    surf = read_topography(os.path.join(file_format_examples, 'example7.txt'), unit='um', info={
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
    np.testing.assert_allclose(cut, 0.126504, atol=1e-6)

    cut = surf.to_nonuniform().short_reliability_cutoff()
    # This differs from the above because the derivatives are computed at slightly different locations
    np.testing.assert_allclose(cut, 0.126505, atol=1e-6)

    cut = surf.to_nonuniform().short_reliability_cutoff(0.2)
    # Should be the maximum of the actual value and the value that was passed
    np.testing.assert_allclose(cut, 0.2)

    cut = surf.to_nonuniform().short_reliability_cutoff(0.1)
    # Should be the maximum of the actual value and the value that was passed
    np.testing.assert_allclose(cut, 0.126505, atol=1e-6)


def test_problem1(file_format_examples):
    surf = read_topography(os.path.join(file_format_examples, 'di6.di'), info={
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
