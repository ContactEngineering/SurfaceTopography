#
# Copyright 2019-2020 Lars Pastewka
#           2020 Michael Röttger
#           2019-2020 Antoine Sanner
#           2020 Kai Haase
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

import datetime
import io
import os
import pickle
import unittest
import warnings

import pytest
import numpy as np
from numpy.testing import assert_array_equal

from NuMPI import MPI

import SurfaceTopography.IO
from SurfaceTopography import open_topography, read_topography
from SurfaceTopography.IO import readers, detect_format, CannotDetectFileFormat
from SurfaceTopography.IO.FromFile import read_xyz, is_binary_stream

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")

DATADIR = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))),
    'file_format_examples')


@pytest.mark.parametrize("reader", readers)
def test_closes_file_on_failure(reader):
    """
    Tests for each reader class that he doesn't raise a Resourcewarning
    """
    fn = os.path.join(DATADIR, "wrongnpyfile.npy")
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter(
            "always")  # deactivate hiding of ResourceWarnings

        try:
            reader(fn)
        except Exception:
            pass
        # assert no warning is a ResourceWarning
        for wi in w:
            assert not issubclass(wi.category, ResourceWarning)


def test_uniform_stylus():
    t = read_topography(os.path.join(DATADIR, 'example7.txt'))
    assert t.is_uniform


def test_cannot_detect_file_format_on_txt():
    with pytest.raises(CannotDetectFileFormat):
        read_topography(os.path.join(DATADIR, 'nonsense_txt_file.txt'))


class IOTest(unittest.TestCase):

    @staticmethod
    def _convert_filelist(filelist):
        """
        Parameters
        ----------
        filelist
            list of strings with filenames withput path

        Returns
        -------
        List of filenames prepended with DATADIR
        """
        return [os.path.join(DATADIR, fn) for fn in filelist]

    def setUp(self):
        binary_examples = [
            'di1.di',
            'di2.di',
            'di3.di',
            'di4.di',
            'example.ibw',
            'spot_1-1000nm.ibw',
            # 'surface.2048x2048.h5',
            '10x10-one_channel_without_name.ibw',
            'example1.mat',
            'example.opd',
            'example.x3p',
            'example2.x3p',
            'opdx1.OPDx',
            'opdx2.OPDx',
            'mi1.mi',
            'N46E013.hgt',
            'example-2d.npy',
        ]

        text_examples = [
            'example.asc',
            'example1.txt',
            'example2.txt',
            'example3.txt',
            'example4.txt',
            'example5.txt',
            'example8.txt',
            # example8: from the reader's docstring, with extra newline at end
            'line_scan_1_minimal_spaces.asc',
            'opdx1.txt',
            'opdx2.txt',
            # Not yet working
            # 'example6.txt',
        ]

        self.binary_example_file_list = self._convert_filelist(binary_examples)
        self.text_example_file_list = self._convert_filelist(text_examples)
        self.text_example_memory_list = [
            """
            0 0
            1 2
            2 4
            3 6
            """
        ]

    def test_keep_file_open(self):
        for fn in self.text_example_file_list:
            # Text file can be opened as binary or text
            with open(fn, 'rb') as f:
                open_topography(f)
                self.assertFalse(f.closed, msg=fn)
            with open(fn, 'r') as f:
                open_topography(f)
                self.assertFalse(f.closed, msg=fn)
        for fn in self.binary_example_file_list:
            with open(fn, 'rb') as f:
                open_topography(f)
                self.assertFalse(f.closed, msg=fn)
        for datastr in self.text_example_memory_list:
            with io.StringIO(datastr) as f:
                open_topography(f)
                self.assertFalse(
                    f.closed,
                    msg="text memory stream for '{}' was closed".format(
                        datastr))

            # Doing the same when but only giving a binary stream
            with io.BytesIO(datastr.encode(encoding='utf-8')) as f:
                open_topography(f)
                self.assertFalse(
                    f.closed,
                    msg="binary memory stream for '{}' was closed".format(
                        datastr))

    def test_is_binary_stream(self):

        # just grep a random existing file here
        fn = self.text_example_file_list[0]

        self.assertTrue(is_binary_stream(open(fn, mode='rb')))
        self.assertFalse(
            is_binary_stream(open(fn, mode='r')))  # opened as text file

        # should also work with streams in memory
        self.assertTrue(
            is_binary_stream(io.BytesIO(b"11111")))  # some bytes in memory
        self.assertFalse(
            is_binary_stream(io.StringIO("11111")))  # some bytes in memory

    def test_can_be_pickled(self):
        file_list = self.text_example_file_list + self.binary_example_file_list

        for fn in file_list:
            reader = open_topography(fn)
            physical_sizes = None
            if reader.default_channel.dim != 1:
                physical_sizes = reader.default_channel.physical_sizes \
                    if reader.default_channel.physical_sizes is not None \
                    else (1.,) * reader.default_channel.dim

            topography = reader.topography(physical_sizes=physical_sizes)
            topographies = [topography]
            if hasattr(topography, 'to_uniform'):
                topographies += [topography.to_uniform(100, 0)]
            for t in topographies:
                s = pickle.dumps(t)
                pickled_t = pickle.loads(s)

                #
                # Compare some attributes after unpickling
                #
                # sometimes the result is a list of topographies
                multiple = isinstance(t, list)
                if not multiple:
                    t = [t]
                    pickled_t = [pickled_t]

                for x, y in zip(t, pickled_t):
                    for attr in ['dim', 'physical_sizes', 'is_periodic']:
                        assert getattr(x, attr) == getattr(y, attr)
                    if x.physical_sizes is not None:
                        assert_array_equal(x.positions(), y.positions())
                        assert_array_equal(x.heights(), y.heights())

    def test_periodic_flag(self):
        file_list = self.text_example_file_list + self.binary_example_file_list
        for fn in file_list:
            reader = open_topography(fn)
            physical_sizes = None
            if reader.default_channel.dim != 1:
                physical_sizes = reader.default_channel.physical_sizes \
                    if reader.default_channel.physical_sizes is not None \
                    else [1., ] * reader.default_channel.dim
            t = reader.topography(physical_sizes=physical_sizes, periodic=True)
            assert t.is_periodic, fn

            t = reader.topography(physical_sizes=physical_sizes,
                                  periodic=False)
            assert not t.is_periodic, fn

    def test_reader_arguments(self):
        """Check whether all readers have channel, physical_sizes and
        height_scale_factor arguments. Also check whether we can execute
        `topography` multiple times for all readers"""
        physical_sizes0 = (1.2, 1.3)
        for fn in self.text_example_file_list + self.binary_example_file_list:
            # Test open -> topography
            r = open_topography(fn)
            physical_sizes = None if r.channels[0].dim == 1 \
                else physical_sizes0
            t = r.topography(channel_index=0, physical_sizes=physical_sizes,
                             height_scale_factor=None)
            if physical_sizes is not None:
                self.assertEqual(t.physical_sizes, physical_sizes)
            # Second call to topography
            t2 = r.topography(channel_index=0, physical_sizes=physical_sizes,
                              height_scale_factor=None)
            if physical_sizes is not None:
                self.assertEqual(t2.physical_sizes, physical_sizes)
            assert_array_equal(t.heights(), t2.heights())
            # Test read_topography
            t = read_topography(fn, channel_index=0,
                                physical_sizes=physical_sizes,
                                height_scale_factor=None)
            if physical_sizes is not None:
                self.assertEqual(t.physical_sizes, physical_sizes)

    def test_readers_with_binary_file_object(self):
        """Check whether all readers have channel, physical_sizes and
        height_scale_factor arguments. Also check whether we can execute
        `topography` multiple times for all readers"""
        physical_sizes0 = (1.2, 1.3)
        for fn in self.text_example_file_list + self.binary_example_file_list:
            # Test open -> topography
            r = open_topography(open(fn, mode='rb'))
            physical_sizes = None if r.channels[0].dim == 1 \
                else physical_sizes0
            t = r.topography(channel_index=0, physical_sizes=physical_sizes,
                             height_scale_factor=None)
            if physical_sizes is not None:
                self.assertEqual(t.physical_sizes, physical_sizes)
            # Second call to topography
            t2 = r.topography(channel_index=0, physical_sizes=physical_sizes,
                              height_scale_factor=None)
            if physical_sizes is not None:
                self.assertEqual(t2.physical_sizes, physical_sizes)
            assert_array_equal(t.heights(), t2.heights(), err_msg=fn)

    def test_reader_topography_same(self):
        """
        Tests that properties like physical sizes, units and nb_grid_pts are
        the  same in the ChannelInfo and the loaded topography
        """

        for fn in self.text_example_file_list + self.binary_example_file_list:
            reader = open_topography(fn)

            for channel in reader.channels:
                topography = channel.topography(
                    physical_sizes=(1, 1) if channel.physical_sizes is None
                    else None)
                assert channel.nb_grid_pts == topography.nb_grid_pts
                if "unit" in channel.info.keys() or \
                        "unit" in topography.info.keys():
                    assert channel.info["unit"] == topography.info["unit"]

                if channel.physical_sizes is not None:
                    assert channel.physical_sizes == topography.physical_sizes


class UnknownFileFormatGivenTest(unittest.TestCase):

    def test_read(self):
        with self.assertRaises(SurfaceTopography.IO.UnknownFileFormatGiven):
            SurfaceTopography.IO.open_topography(
                os.path.join(DATADIR, "surface.2048x2048.h5"),
                format='Nonexistentfileformat')

    def test_detect_format(self):
        with self.assertRaises(SurfaceTopography.IO.UnknownFileFormatGiven):
            SurfaceTopography.IO.open_topography(
                os.path.join(DATADIR, "surface.2048x2048.h5"),
                format='Nonexistentfileformat')


def test_file_format_mismatch():
    with pytest.raises(SurfaceTopography.IO.FileFormatMismatch):
        SurfaceTopography.IO.open_topography(
            os.path.join(DATADIR, 'surface.2048x2048.h5'), format="npy")


class LineScanInFileWithMinimalSpacesTest(unittest.TestCase):
    def test_detect_format_then_read(self):
        self.assertEqual(detect_format(
            os.path.join(DATADIR, 'line_scan_1_minimal_spaces.asc')), 'xyz')

    def test_read(self):
        surface = read_xyz(
            os.path.join(DATADIR, 'line_scan_1_minimal_spaces.asc'))

        self.assertFalse(surface.is_uniform)
        self.assertEqual(surface.dim, 1)

        x, y = surface.positions_and_heights()
        self.assertGreater(len(x), 0)
        self.assertEqual(len(x), len(y))


@pytest.mark.parametrize("reader", readers)
def test_readers_have_name(reader):
    reader.name()


def test_di_date():
    t = read_topography(os.path.join(DATADIR, 'di1.di'))
    assert t.info['acquisition_time'] == datetime.datetime(2016, 1, 12, 9, 57,
                                                           48)


# yes, the German version still has "Value units"
@pytest.mark.parametrize("lang_filename_infix", ["english", "german"])
def test_gwyddion_txt_import(lang_filename_infix):
    fname = os.path.join(
        DATADIR,
        'gwyddion-export-{}.txt'.format(lang_filename_infix))

    #
    # test channel infos
    #
    reader = open_topography(fname)

    assert len(reader.channels) == 1
    channel = reader.default_channel

    assert channel.name == "My Channel Name"
    assert channel.info['unit'] == 'm'
    assert pytest.approx(
        channel.physical_sizes[0]) == 12.34 * 1e-6  # was given as µm
    assert pytest.approx(
        channel.physical_sizes[1]) == 5678.9 * 1e-9  # was given as nm

    #
    # test metadata of topography
    #
    topo = reader.topography()
    assert topo.info['unit'] == 'm'
    assert pytest.approx(
        topo.physical_sizes[0]) == 12.34 * 1e-6  # was given as µm
    assert pytest.approx(
        topo.physical_sizes[1]) == 5678.9 * 1e-9  # was given as nm

    #
    # test scaling and order of data
    #
    # The order of the lines in the text files mimic the lines as they
    # are shown in the gwyddion plot.
    #
    # In gwyddion's text export:
    # - first index corresponds to y dimension (rows), second index (columns)
    #   to x dimension
    # - y coordinates grow from top row to bottom row
    # - x coordinates grow from left column to column of array
    #
    # PyCo's heights() has a different order:
    # - first index corresponds to x dimension, second index to y dimension
    # - plot from the heights correspond to same image in gwyddion if plotted
    #   with "pcolormesh(t.heights.T)", but with origin in lower left, i.e. the
    #   image looks flipped vertically when compared to gwyddion
    #
    # => heights() must be same array as in file, but transposed
    #
    heights_in_file = [[1, 1.5, 3],
                       [-2, -3, -6],
                       [0, 0, 0],
                       [9, 9, 9]]

    expected_heights = np.array(heights_in_file).T

    np.testing.assert_allclose(topo.heights(), expected_heights)


def test_detect_dormat():
    assert detect_format(os.path.join(DATADIR, 'di1.di')) == 'di'
    assert detect_format(os.path.join(DATADIR, 'di2.di')) == 'di'
    assert detect_format(os.path.join(DATADIR, 'example.ibw')) == 'ibw'
    assert detect_format(os.path.join(DATADIR, 'example.opd')) == 'opd'
    assert detect_format(os.path.join(DATADIR, 'example.x3p')) == 'x3p'
    assert detect_format(os.path.join(DATADIR, 'example1.mat')) == 'mat'
    assert detect_format(os.path.join(DATADIR, 'example.asc')) == 'xyz'
    assert detect_format(os.path.join(DATADIR, 'line_scan_1_minimal_spaces.asc')) == 'xyz'
    assert detect_format(os.path.join(DATADIR, 'example-2d.npy')) == 'npy'
    assert detect_format(os.path.join(DATADIR, 'surface.2048x2048.h5')) == 'h5'
