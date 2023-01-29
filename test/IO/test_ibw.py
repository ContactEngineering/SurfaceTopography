#
# Copyright 2019-2020 Lars Pastewka
#           2019 Michael Röttger
#           2019 Kai Haase
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

import unittest
import os

from SurfaceTopography.IO import detect_format
from SurfaceTopography.IO.IBW import IBWReader
from SurfaceTopography import read_topography
from SurfaceTopography import open_topography

import pytest
from NuMPI import MPI

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial funcionalities, please execute with pytest")

DATADIR = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))),
    'file_format_examples')


class IBWSurfaceTest(unittest.TestCase):

    def setUp(self):
        self.file_path = os.path.join(DATADIR, 'ibw-1.ibw')

    def test_read_filestream(self):
        """
        The reader has to work when the file was already opened as binary for
        it to work in TopoBank.
        """

        try:
            read_topography(self.file_path)
        except Exception as e:
            self.fail("read_topography() raised an exception (not passing a "
                      "file stream)!" + str(e))

        try:
            f = open(self.file_path, 'r')
            read_topography(f)
        except Exception as e:
            self.fail("read_topography() raised an exception (passing a "
                      "non-binary file stream)!" + str(e))
        finally:
            f.close()

        try:
            f = open(self.file_path, 'rb')
            read_topography(f)
        except Exception as e:
            self.fail("read_topography() raised an exception (passing a "
                      "binary file stream)!" + str(e))
        finally:
            f.close()

    def test_init(self):
        reader = IBWReader(self.file_path)

        self.assertEqual(reader._channel_names,
                         ['HeightRetrace', 'AmplitudeRetrace',
                          'PhaseRetrace', 'ZSensorRetrace'])
        self.assertEqual(reader._default_channel, 0)
        self.assertEqual(reader.data['wave_header']['next'], 114425520)

    def test_channels(self):
        reader = IBWReader(self.file_path)

        exp_size = 5.009784735812133e-08  # 50 nm, see also gwyddion result

        expected_channels = [
            {'name': 'HeightRetrace',
             'dim': 2,
             'physical_sizes': (exp_size, exp_size)},
            {'name': 'AmplitudeRetrace',
             'dim': 2,
             'physical_sizes': (exp_size, exp_size)},
            {'name': 'PhaseRetrace',
             'dim': 2,
             'physical_sizes': (exp_size, exp_size)},
            {'name': 'ZSensorRetrace',
             'dim': 2,
             'physical_sizes': (exp_size, exp_size)}]

        self.assertEqual(len(reader.channels), len(expected_channels))

        for exp_ch, ch in zip(expected_channels, reader.channels):
            self.assertEqual(exp_ch['name'], ch.name)
            self.assertEqual(exp_ch['dim'], ch.dim)
            self.assertAlmostEqual(exp_ch['physical_sizes'][0],
                                   ch.physical_sizes[0])
            self.assertAlmostEqual(exp_ch['physical_sizes'][1],
                                   ch.physical_sizes[1])

    def test_topography(self):

        reader = IBWReader(self.file_path)
        topo = reader.topography()

        self.assertAlmostEqual(topo.heights()[0, 0], -6.6641803e-10, places=3)

    def test_topography_all_channels(self):
        """
        Test whether a topography can be read from every channel.
        """
        reader = IBWReader(self.file_path)
        for channel_info in reader.channels:
            channel_info.topography()


def test_ibw_kpfm_file():
    """
    We had an issue with KPFM files, see
    https://github.com/pastewka/PyCo/pull/231#discussion_r354687995

    This test should ensure that it's fixed.
    """
    fn = os.path.join(DATADIR, 'spot_1-1000nm.ibw')
    reader = open_topography(fn)

    #
    # Try to read all channels
    #
    for channel_info in reader.channels:
        assert pytest.approx(channel_info.physical_sizes[0],
                             abs=0.01) == 2e-05  # 20 µm
        assert pytest.approx(channel_info.physical_sizes[1],
                             abs=0.01) == 2e-05  # 20 µm

        channel_info.topography()


def test_ibw_file_with_one_channel_without_name():
    """
    After implementing new IBW readers there was an issue
    https://github.com/pastewka/TopoBank/issues/413

    This test should ensure that it's fixed.
    """
    fn = os.path.join(DATADIR, "10x10-one_channel_without_name.ibw")

    reader = open_topography(fn)

    assert len(reader.channels) == 1

    ch_info = reader.channels[0]

    # we could use "Default" here, but what if there are multiple no names?
    assert ch_info.name == 'no name (1)'
    assert ch_info.dim == 2
    assert ch_info.nb_grid_pts == (10, 10)
    # TODO when the new ChannelInfo objects are used, we should check here if
    #  all expected fields are set correclty


class ibwSurfaceTest2(unittest.TestCase):
    def setUp(self):
        pass

    def test_read(self):
        reader = IBWReader(os.path.join(DATADIR, 'ibw-1.ibw'))
        surface = reader.topography()
        nx, ny = surface.nb_grid_pts
        self.assertEqual(nx, 512)
        self.assertEqual(ny, 512)
        sx, sy = surface.physical_sizes
        self.assertAlmostEqual(sx, 5.00978e-8)
        self.assertAlmostEqual(sy, 5.00978e-8)
        # self.assertEqual(surface.info['unit'], 'm')
        # Disabled unit check because I'm not sure
        # how to assign a valid unit to every channel - see IBW.py
        self.assertTrue(surface.is_uniform)

    def test_detect_format_then_read(self):
        f = open(os.path.join(DATADIR, 'ibw-1.ibw'), 'rb')
        fmt = detect_format(f)
        self.assertTrue(fmt, 'ibw')
        open_topography(f, format=fmt).topography()
        f.close()
