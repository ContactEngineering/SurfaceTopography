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


def test_reliability_cutoff_from_instrument_metadata(file_format_examples):
    surf = read_topography(os.path.join(file_format_examples, 'di1.di'), info={
        'instrument': {
            'tip_radius': {
                'value': 40,
                'unit': 'nm',
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
