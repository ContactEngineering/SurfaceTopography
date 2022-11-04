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

import logging
import os

import numpy as np

from SurfaceTopography.IO import MitutoyoReader


logger = logging.getLogger(__name__)


def _make_marker(d):
    """Mark everything for comparison."""
    if isinstance(d, list):
        return [_make_marker(e) for e in d]
    elif isinstance(d, dict):
        return {k: _make_marker(v) for k, v in d.items()}
    else:
        return True


def _compare(source, target, marker=None):
    """Compare source and target partially, as marked by marker. Compares
    against everything in target if marker is None."""
    if marker is None:
        marker = _make_marker(target)
    if isinstance(marker, dict):
        for k, v in marker.items():
            if k not in source:
                logger.error("{} not in source '{}'.".format(k, source))
                return False
            if k not in target:
                logger.error("{} not in target '{}'.".format(k, source))
                return False

            logger.debug("Descending into sub-tree '{}' of '{}'.".format(
                source[k], source))
            # descend
            if not _compare(source[k], target[k], v):
                return False  # one failed comparison suffices

    elif isinstance(marker, list):  # source, target and marker must have same length
        logger.debug("Branching into element wise sub-trees of '{}'.".format(
            source))
        for s, t, m in zip(source, target, marker):
            if not _compare(s, t, m):
                return False  # one failed comparison suffices
    else:  # arrived at leaf, comparison desired?
        if marker is not False:  # yes
            logger.debug("Comparing '{}' == '{}' -> {}.".format(
                source, target, source == target))
            return source == target

    # comparison either not desired or successful for all elements
    return True


EXPECTED_CHANNEL_INFO = {
    'roughness_metrics': [
        {'key': 'Ra', 'value': 0.018, 'unit': 'µm'},
        {'key': 'Rq', 'value': 0.027, 'unit': 'µm'},
        {'key': 'Rz', 'value': 0.191, 'unit': 'µm'},
        {'key': 'Rpk', 'value': 0.052, 'unit': 'µm'},
        {'key': 'Rvk', 'value': 0.489, 'unit': 'µm'}],
    'cut_off': {'value': '0.08', 'unit': 'mm'},
    'acquisition_time': '2022-10-10 00:00:00',
    'unit': 'µm'
}


def test_read_uniform(file_format_examples):
    reader = MitutoyoReader(os.path.join(file_format_examples, 'mitutoyo_mock.xlsx'))

    assert len(reader.channels) == 1
    assert reader.channels[0] == reader.default_channel

    # test a few channel properties
    assert reader.default_channel.unit == 'µm'

    np.testing.assert_almost_equal(reader.default_channel.area_per_pt, 0.5)

    assert reader.default_channel.dim == 1

    physical_sizes = reader.default_channel.physical_sizes
    assert len(physical_sizes) == 1
    np.testing.assert_almost_equal(physical_sizes[0], 480.)

    nx, = reader.channels[0].nb_grid_pts
    assert nx == 960

    # test channel info
    channel_info = reader.channels[0].info
    assert _compare(channel_info, EXPECTED_CHANNEL_INFO)

    # test number of grid points in topography
    topography = reader.topography()
    nx, = topography.nb_grid_pts
    assert nx == 960

    # test rms roughness
    np.testing.assert_almost_equal(topography.rms_height_from_profile(), 0.16866328079293708)

    # test for uniform flag
    assert topography.is_uniform


def test_read_nonuniform(file_format_examples):
    reader = MitutoyoReader(os.path.join(file_format_examples, 'mitutoyo_nonuniform_mock.xlsx'))

    assert len(reader.channels) == 1
    assert reader.channels[0] == reader.default_channel

    # test a few channel properties
    assert reader.default_channel.unit == 'µm'
    assert reader.default_channel.dim == 1
    # ATTENTION: physical_sizes differs from uniform linescan above
    physical_sizes = reader.default_channel.physical_sizes
    assert len(physical_sizes) == 1
    np.testing.assert_almost_equal(physical_sizes[0], 479.5)

    nx, = reader.channels[0].nb_grid_pts
    assert nx == 960

    # test channel info
    channel_info = reader.channels[0].info
    assert _compare(channel_info, EXPECTED_CHANNEL_INFO)

    # test number of grid points in topography
    topography = reader.topography()
    nx, = topography.nb_grid_pts
    assert nx == 960

    # test rms roughness
    # very different rms height for slightly modified position compared to uniform line scan
    # This is not a bug, but due to approximation methods
    np.testing.assert_almost_equal(topography.rms_height_from_profile(), 0.1371261153644389)

    # test for uniform flag
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
