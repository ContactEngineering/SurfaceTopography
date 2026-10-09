#
# Copyright 2019-2020, 2023 Lars Pastewka
#           2020 Antoine Sanner
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

import os
import unittest

import pytest
from NuMPI import MPI

from SurfaceTopography import open_topography, read_topography
from SurfaceTopography.IO import detect_format
from SurfaceTopography.IO.IBW import IBWReader

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
        # The reader must not pin the wave data in memory; it is loaded on
        # demand in `topography()`
        self.assertFalse(hasattr(reader, 'data'))

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
        self.assertEqual(topo.info['instrument']['vendor'], 'Asylum Research')

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


#
# The tests below check the data interpretation against the Igor file module
# of Gwyddion (igorfile.c). They use synthetic waves written by
# `_write_ibw5`.
#

def _write_ibw5(arr, sfA=(1.0, 1.0, 1.0, 1.0), data_units=b'', dim_units=b'',
                labels=None, note=b''):
    """
    Write a minimal Igor binary wave (version 5) with float32 data.

    Parameters
    ----------
    arr : np.ndarray
        Data with one to three dimensions, indexed (row, column, layer)
        as in Igor.
    sfA : tuple of floats
        Spacing of the grid along each dimension.
    data_units : bytes
        Unit of the data values (at most three characters).
    dim_units : bytes
        Unit of the first two dimensions (at most three characters).
    labels : list of bytes, optional
        Element labels of the last dimension (the channel names).
    note : bytes
        Wave note.

    Returns
    -------
    file : io.BytesIO
        The binary wave.
    """
    import io
    import struct

    import numpy as np

    arr = np.asarray(arr, dtype='<f4')
    n_dim = list(arr.shape) + [0] * (4 - arr.ndim)
    data = arr.tobytes(order='F')  # Igor stores data column-major

    # Dimension labels: the first entry labels the dimension itself
    dim_labels_size = [0, 0, 0, 0]
    label_bytes = b''
    if labels is not None:
        for label in [b''] + labels:
            label_bytes += label.ljust(32, b'\0')
        dim_labels_size[arr.ndim - 1] = len(label_bytes)

    wave_header = bytearray(320)
    struct.pack_into('<l', wave_header, 12, arr.size)  # npnts
    struct.pack_into('<h', wave_header, 16, 2)  # float32
    struct.pack_into('<h', wave_header, 26, 1)  # whVersion
    wave_header[28:28 + 4] = b'test'  # bname
    struct.pack_into('<4l', wave_header, 68, *n_dim)
    struct.pack_into('<4d', wave_header, 84, *sfA)
    wave_header[148:148 + len(data_units)] = data_units
    wave_header[152:152 + len(dim_units)] = dim_units
    wave_header[156:156 + len(dim_units)] = dim_units

    bin_header = bytearray(64)
    struct.pack_into('<hhllll4l4llll', bin_header, 0, 5, 0,
                     len(wave_header) + len(data), 0, len(note), 0,
                     0, 0, 0, 0, *dim_labels_size, 0, 0, 0)
    # The checksum makes the 16-bit sum of both headers vanish
    checksum = -sum(struct.unpack('<192H', bytes(bin_header + wave_header)))
    struct.pack_into('<H', bin_header, 2, checksum & 0xffff)

    return io.BytesIO(bytes(bin_header + wave_header) + data + note + label_bytes)


def test_ibw_channel_data_units(file_format_examples):
    """
    Asylum Research writes the same unit to the wave header for all channels;
    as in Gwyddion, the physical unit of the data is derived from the channel
    name.
    """
    reader = open_topography(os.path.join(file_format_examples, 'ibw-1.ibw'))
    assert [ch.data_unit for ch in reader.channels] == ['m', 'm', 'deg', 'm']
    assert all(ch.unit == 'm' for ch in reader.channels)

    reader = open_topography(os.path.join(file_format_examples, 'spot_1-1000nm.ibw'))
    assert [ch.data_unit for ch in reader.channels] == \
        ['m', 'm', 'm', 'm', 'deg', 'deg', 'V', 'V']


def test_ibw_acquisition_time(file_format_examples):
    import datetime

    reader = open_topography(os.path.join(file_format_examples, 'ibw-1.ibw'))
    assert reader.channels[0].info['acquisition_time'] == \
        datetime.datetime(2015, 10, 29, 19, 39, 34)
    t = reader.topography()
    assert t.info['acquisition_time'] == datetime.datetime(2015, 10, 29, 19, 39, 34)
    assert t.info['instrument']['name'] == 'MFP3D'
    assert t.info['raw_metadata']['ScanRate'] == '1.0016'


def test_ibw_asylum_unit_from_note():
    """An explicit `<name>Unit` entry of the note overrides the default unit"""
    import numpy as np

    arr = np.arange(2 * 2 * 2, dtype=float).reshape(2, 2, 2)
    f = _write_ibw5(arr, sfA=(1e-6, 1e-6, 1, 1), data_units=b'm', dim_units=b'm',
                    labels=[b'HeightTrace', b'UserIn0Retrace'],
                    note=b'HeightUnit: nm\rUserIn0Unit: A\r')
    reader = IBWReader(f)
    assert [ch.data_unit for ch in reader.channels] == ['nm', 'A']
    t = reader.topography(channel_index=0)
    assert t.unit == 'm'
    np.testing.assert_allclose(t.heights(), np.fliplr(arr[:, :, 0]) * 1e-9)
    t = reader.topography(channel_index=1)
    np.testing.assert_allclose(t.heights(), np.fliplr(arr[:, :, 1]))


def test_ibw_different_data_and_dimension_units():
    """Data and dimension units may differ (previously an assertion error)"""
    import numpy as np

    arr = np.random.default_rng(0).random((4, 3, 1))
    f = _write_ibw5(arr, sfA=(0.5, 0.25, 1, 1), data_units=b'nm', dim_units=b'um')
    reader = IBWReader(f)
    (ch,) = reader.channels
    assert ch.unit == 'µm'
    assert ch.data_unit == 'nm'
    assert ch.nb_grid_pts == (4, 3)
    np.testing.assert_allclose(ch.physical_sizes, (2.0, 0.75))
    t = reader.topography()
    assert t.unit == 'µm'
    np.testing.assert_allclose(t.heights(), np.fliplr(arr[:, :, 0]) * 1e-3)


def test_ibw_two_dimensional_square_wave():
    """Like Gwyddion, a square two-dimensional wave is a single image"""
    import numpy as np

    arr = np.random.default_rng(1).random((3, 3))
    f = _write_ibw5(arr, sfA=(1e-6, 2e-6, 1, 1), data_units=b'm', dim_units=b'm')
    reader = IBWReader(f)
    assert len(reader.channels) == 1
    t = reader.topography()
    assert t.dim == 2
    np.testing.assert_allclose(t.physical_sizes, (3e-6, 6e-6))
    np.testing.assert_allclose(t.heights(), np.fliplr(arr))


def test_ibw_curves():
    """
    One- and non-square two-dimensional waves contain curves; these are
    reported as line scans
    """
    import numpy as np

    arr = np.random.default_rng(2).random((5, 2))
    f = _write_ibw5(arr, sfA=(1e-6, 1, 1, 1), data_units=b'm', dim_units=b'm',
                    labels=[b'Profile1', b'Profile2'])
    reader = IBWReader(f)
    assert [ch.name for ch in reader.channels] == ['Profile1', 'Profile2']
    for i, ch in enumerate(reader.channels):
        assert ch.dim == 1
        assert ch.nb_grid_pts == (5,)
        t = ch.topography()
        assert t.dim == 1
        assert t.unit == 'm'
        np.testing.assert_allclose(t.physical_sizes, (5e-6,))
        np.testing.assert_allclose(t.heights(), arr[:, i])

    arr = np.random.default_rng(3).random(7)
    f = _write_ibw5(arr, sfA=(1e-9, 1, 1, 1), data_units=b'm', dim_units=b'm')
    reader = IBWReader(f)
    (ch,) = reader.channels
    t = ch.topography()
    assert t.dim == 1
    np.testing.assert_allclose(t.physical_sizes, (7e-9,))
    np.testing.assert_allclose(t.heights(), arr)


def test_ibw_undefined_data():
    """NaNs are undefined data (Gwyddion masks them)"""
    import numpy as np

    arr = np.ones((3, 3, 1))
    arr[1, 2, 0] = np.nan
    f = _write_ibw5(arr, data_units=b'm', dim_units=b'm')
    t = IBWReader(f).topography()
    assert t.has_undefined_data
    assert t.heights().mask[1, 0]
