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

import os

import numpy as np

from SurfaceTopography.IO import MitutoyoReader


def test_read_uniform(file_format_examples):
    reader = MitutoyoReader(os.path.join(file_format_examples, 'mitutoyo_mock.xlsx'))
    nx, = reader.channels[0].nb_grid_pts
    assert nx == 960

    topography = reader.topography()
    nx, = topography.nb_grid_pts
    assert nx == 960
    np.testing.assert_almost_equal(topography.rms_height_from_profile(), 0.16866328079293708)
    assert topography.is_uniform


def test_read_nonuniform(file_format_examples):
    reader = MitutoyoReader(os.path.join(file_format_examples, 'mitutoyo_nonuniform_mock.xlsx'))
    nx, = reader.channels[0].nb_grid_pts
    assert nx == 960

    topography = reader.topography()
    nx, = topography.nb_grid_pts
    assert nx == 960
    # BUG: very different rms height for slightly modified position
    np.testing.assert_almost_equal(topography.rms_height_from_profile(), 0.13711646786253162)
    assert not topography.is_uniform
