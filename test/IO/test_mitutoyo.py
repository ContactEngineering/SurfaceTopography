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
    # very different rms height for slightly modified position compared to uniform line scan
    # This is not a bug, but due to approximation methods
    np.testing.assert_almost_equal(topography.rms_height_from_profile(), 0.1371261153644389)
    assert not topography.is_uniform


def test_uniform_vs_nonuniform(file_format_examples):
    """Uniform and nonuniform reader positions should be equal with one exemption."""
    nonuniform_reader = MitutoyoReader(os.path.join(file_format_examples, 'mitutoyo_nonuniform_mock.xlsx'))
    uniform_reader = MitutoyoReader(os.path.join(file_format_examples, 'mitutoyo_mock.xlsx'))

    uniform_topography = uniform_reader.topography()
    nonuniform_topography = nonuniform_reader.topography()

    uniform_x, uniform_h = uniform_topography.positions_and_heights()
    nonuniform_x, nonuniform_h = nonuniform_topography.positions_and_heights()

    # I'd like
    #   np.testing.assert_almost_equal(uniform_topography.physical_sizes, nonuniform_topography.physical_sizes)
    # but currently it must be
    np.testing.assert_almost_equal(uniform_topography.physical_sizes[0], nonuniform_topography.physical_sizes[0] + 0.5)

    # Convention is to have uniform linescan begin at zero, nonuniform linescan
    # built from Mitutoyo file follows this convention as well by removing the
    # initial grid point's absolute position
    np.testing.assert_almost_equal(uniform_x[:99], nonuniform_x[:99])

    # the 100th positions has been slightly shifted by 0.1 um in the nonuniform
    # mock file hence the position is off at index 99
    np.testing.assert_almost_equal(uniform_x[99], nonuniform_x[99] - 0.1)

    np.testing.assert_almost_equal(uniform_x[100:], nonuniform_x[100:])

    # heights must be the same
    np.testing.assert_almost_equal(nonuniform_h, nonuniform_h)
