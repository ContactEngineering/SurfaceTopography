#
# Copyright 2016, 2020 Lars Pastewka
#           2018, 2020 Antoine Sanner
#           2018, 2020 Michael Röttger
#           2015-2016 Till Junge
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

from SurfaceTopography import read_container


def test_bandwidth_and_unit_suggestion(file_format_examples):
    c, = read_container(f'{file_format_examples}/container1.zip')
    upper_um, lower_um = c.bandwidth(unit='µm')
    upper_mm, lower_mm = c.bandwidth(unit='mm')
    np.testing.assert_almost_equal(upper_um, 0.002)
    np.testing.assert_almost_equal(lower_um, 100)
    np.testing.assert_almost_equal(upper_um, upper_mm * 1000)
    np.testing.assert_almost_equal(lower_um, lower_mm * 1000)
    assert c.suggest_length_unit() == 'µm'
