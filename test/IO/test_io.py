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

import io
import os
import pickle
import tempfile
import unittest
import warnings

import pytest
import numpy as np
from numpy.testing import assert_array_equal

import NuMPI
from NuMPI import MPI

import SurfaceTopography.IO
from SurfaceTopography import open_topography, read_topography
from SurfaceTopography.IO import readers, detect_format, CannotDetectFileFormat, MetadataAlreadyFixedByFile
from SurfaceTopography.IO.common import is_binary_stream
from SurfaceTopography.IO.Text import read_matrix, read_xyz
from SurfaceTopography.UniformLineScanAndTopography import Topography

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")

DATADIR = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))),
    'file_format_examples')


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


binary_example_file_list = _convert_filelist([
    'di1.di',
    'di2.di',
    'di3.di',
    'di4.di',
    'di5.di',
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
    'example.zon',
    'example.nc',
] + [] if NuMPI._has_mpi4py else ['example-2d.npy'])  # MPI I/O does not support Python streams

text_example_file_list = _convert_filelist([
    'example1.txt',
    'example2.txt',
    'example3.txt',
    'example4.txt',
    'example5.txt',
    'example8.txt',
    # example8: from the reader's docstring, with extra newline at end
    'opdx1.txt',
    'opdx2.txt',
    'example-2d.xyz',
    # Not yet working
    # 'example6.txt',
])

text_example_without_size_file_list = _convert_filelist([
    'example.xyz',
    'line_scan_1_minimal_spaces.asc',
])

explicit_physical_sizes = _convert_filelist([
    'example5.txt',
    'example1.mat',
    'example-2d.npy'
])

text_example_memory_list = [
    """
            0 0
            1 2
            2 4
            3 6
            """
]


@pytest.mark.parametrize("reader", readers)
def test_no_resource_warning_on_failure(reader):
    """
    Tests for each reader class that it doesn't raise a ResourceWarning
    """
    fn = os.path.join(DATADIR, "wrongnpyfile.npy")
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")  # deactivate hiding of ResourceWarnings

        # noinspection PyBroadException
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


@pytest.mark.parametrize('fn', text_example_file_list + text_example_without_size_file_list)
def test_keep_text_file_open(fn):
    # Text file can be opened as binary or text
    with open(fn, 'rb') as f:
        open_topography(f)
        assert not f.closed, f"Text file {fn} was opened as binary file and is closed, but should not"
    with open(fn, 'r') as f:
        open_topography(f)
        assert not f.closed, f"Text file {fn} was opened as text file and is closed, but should not"


@pytest.mark.parametrize('fn', binary_example_file_list)
def test_keep_binary_file_open(fn):
    with open(fn, 'rb') as f:
        open_topography(f)
        assert not f.closed, f"Binary file {fn} was opened as binary file and is closed, but should not"


@pytest.mark.parametrize('datastr', text_example_memory_list)
def test_keep_stream_from_memory_open(datastr):
    with io.StringIO(datastr) as f:
        open_topography(f)
        assert not f.closed, "text memory stream for '{}' was closed".format(datastr)

    # Doing the same when but only giving a binary stream
    with io.BytesIO(datastr.encode(encoding='utf-8')) as f:
        open_topography(f)
        assert not f.closed, "binary memory stream for '{}' was closed".format(datastr)


def test_is_binary_stream():
    # just grep a random existing file here
    fn = text_example_file_list[0]

    assert is_binary_stream(open(fn, mode='rb'))
    assert not is_binary_stream(open(fn, mode='r'))  # opened as text file

    # should also work with streams in memory
    assert is_binary_stream(io.BytesIO(b"11111"))  # some bytes in memory
    assert not is_binary_stream(io.StringIO("11111"))  # some bytes in memory


@pytest.mark.parametrize('fn', text_example_file_list + text_example_without_size_file_list + binary_example_file_list)
def test_can_be_pickled(fn):
    reader = open_topography(fn)
    physical_sizes = None
    if reader.default_channel.physical_sizes is None:
        physical_sizes = (1.,) * reader.default_channel.dim

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


@pytest.mark.parametrize('fn', text_example_file_list + text_example_without_size_file_list + binary_example_file_list)
def test_reader_arguments(fn):
    """Check whether all readers have channel, physical_sizes and
    height_scale_factor arguments. Also check whether we can execute
    `topography` multiple times for all readers"""
    physical_sizes0 = (1.2, 1.3)

    # Test open -> topography
    r = open_topography(fn)
    physical_sizes = None if r.channels[0].physical_sizes is not None else physical_sizes0

    t = r.topography(channel_index=0, physical_sizes=physical_sizes,
                     height_scale_factor=None)
    if physical_sizes is not None:
        assert t.physical_sizes == physical_sizes
    # Second call to topography
    t2 = r.topography(channel_index=0, physical_sizes=physical_sizes,
                      height_scale_factor=None)
    if physical_sizes is not None:
        assert t2.physical_sizes == physical_sizes
    assert_array_equal(t.heights(), t2.heights())
    # Test read_topography
    t = read_topography(fn, channel_index=0,
                        physical_sizes=physical_sizes,
                        height_scale_factor=None)
    if physical_sizes is not None:
        assert t.physical_sizes == physical_sizes


@pytest.mark.parametrize('fn', text_example_file_list + text_example_without_size_file_list + binary_example_file_list)
def test_readers_with_binary_file_object(fn):
    """Check whether all readers have channel, physical_sizes and
    height_scale_factor arguments. Also check whether we can execute
    `topography` multiple times for all readers"""
    physical_sizes0 = (1.2, 1.3)

    # Test open -> topography
    r = open_topography(open(fn, mode='rb'))
    physical_sizes = None if r.channels[0].physical_sizes is not None else physical_sizes0
    t = r.topography(channel_index=0, physical_sizes=physical_sizes,
                     height_scale_factor=None)
    if physical_sizes is not None:
        assert t.physical_sizes == physical_sizes
    # Second call to topography
    t2 = r.topography(channel_index=0, physical_sizes=physical_sizes,
                      height_scale_factor=None)
    if physical_sizes is not None:
        assert t2.physical_sizes == physical_sizes
    assert_array_equal(t.heights(), t2.heights(), err_msg=fn)


@pytest.mark.parametrize('fn', text_example_file_list + text_example_without_size_file_list + binary_example_file_list)
def test_nb_grid_pts_and_physical_sizes_are_tuples_or_none(fn):
    r = open_topography(fn)
    assert isinstance(r.default_channel.nb_grid_pts, tuple), f'{fn} - {r.__class__}: {r.default_channel.nb_grid_pts}'
    if r.default_channel.physical_sizes is not None:
        assert isinstance(r.default_channel.physical_sizes, tuple), \
            f'{fn} - {r.__class__}: {r.default_channel.physical_sizes}'
        # If it is a tuple, it cannot contains None's
        assert np.all([p is not None for p in r.default_channel.physical_sizes]), \
            f'{fn} - {r.__class__}: {r.default_channel.physical_sizes}'


@pytest.mark.parametrize('fn', text_example_file_list + text_example_without_size_file_list + binary_example_file_list)
def test_reader_topography_same(fn):
    """
    Tests that properties like physical sizes, units and nb_grid_pts are
    the same in the ChannelInfo and the loaded topography.
    """

    reader = open_topography(fn)

    for channel in reader.channels:
        foo_str = reader.format() + "-%d" % (channel.index,)  # unique for each channel
        topography = channel.topography(
            physical_sizes=(1, 1) if channel.physical_sizes is None
            else None,
            info=dict(foo=foo_str))
        assert channel.nb_grid_pts == topography.nb_grid_pts

        # some checks on info dict in channel and topography
        assert topography.info['foo'] == foo_str
        if channel.unit is not None or topography.unit is not None:
            assert channel.unit == topography.unit
            assert channel.info['unit'] == topography.unit
            assert channel.unit == topography.info['unit']

        if channel.physical_sizes is not None:
            assert channel.physical_sizes == topography.physical_sizes

        if channel.height_scale_factor is not None and hasattr(topography, 'scale_factor'):
            assert channel.height_scale_factor == topography.scale_factor


@pytest.mark.parametrize('fn', text_example_file_list + text_example_without_size_file_list + binary_example_file_list)
def test_reader_args_doesnt_overwrite_data_from_file(fn):
    """
    Tests that if some properties like `physical_sizes and `height_scale_factor`
    are given in the file, they cannot be overridden by given arguments to
    the .topography() method.
    """
    reader = open_topography(fn)
    ch = reader.default_channel
    physical_sizes_arg_if_missing_in_file = (1.,) * ch.dim
    physical_sizes_arg = physical_sizes_arg_if_missing_in_file if ch.physical_sizes is None else None

    if ch.physical_sizes is not None:
        with pytest.raises(MetadataAlreadyFixedByFile):
            reader.topography(physical_sizes=physical_sizes_arg_if_missing_in_file)

    if ch.height_scale_factor is not None:
        with pytest.raises(MetadataAlreadyFixedByFile):
            if ch.physical_sizes is None:
                reader.topography(physical_sizes=physical_sizes_arg, height_scale_factor=10)
            else:
                # if an exception happens, we want it because of height scale factor
                reader.topography(height_scale_factor=10)

    # A small problem with this test is maybe that there are a few input
    # files which pass this test without any assert, so it looks like
    # a passed test, but it has no meaning. Since this are only three files
    # by now, I think this is okay, sorting this out would be difficult.


@pytest.mark.parametrize('fn', text_example_file_list + binary_example_file_list)
def test_periodic_flag(fn):
    reader = open_topography(fn)
    ch = reader.default_channel
    physical_sizes_arg_if_missing_in_file = (1.,) * ch.dim
    physical_sizes_arg = physical_sizes_arg_if_missing_in_file if ch.physical_sizes is None else None

    t = reader.topography(physical_sizes=physical_sizes_arg, periodic=True)
    assert t.is_periodic, fn

    t = reader.topography(physical_sizes=physical_sizes_arg, periodic=False)
    assert not t.is_periodic, fn


@pytest.mark.parametrize('fn', text_example_file_list + text_example_without_size_file_list + binary_example_file_list)
def test_reader_height_scale_factor_arg_for_topography(fn):
    """Test whether height_scale_factor can be given to .topography() and is effective.

    Also checking whether the reader channels have .height_scale_factor attribute and
    whether it is equal to the scaling factor known from topography.

    Also tests that info dict of channel and topography have no height_scale_factor,
    because this should be a channel property now.
    """
    reader = open_topography(fn)
    ch = reader.default_channel

    assert hasattr(ch, 'height_scale_factor')
    assert 'height_scale_factor' not in ch.info

    height_scale_factor_if_missing_in_file = 2  # just some number

    # calculate argument for .topography()
    height_scale_factor_arg = height_scale_factor_if_missing_in_file if ch.height_scale_factor is None else None

    # which factor we expect at the end
    exp_height_scale_factor = height_scale_factor_if_missing_in_file if ch.height_scale_factor is None \
        else ch.height_scale_factor

    # in order to call .topography(), we also need valid physical_sizes
    physical_sizes_arg_if_missing_in_file = (1.,) * ch.dim
    physical_sizes_arg = physical_sizes_arg_if_missing_in_file if ch.physical_sizes is None else None

    # The check whether an exception is raised if meta data like `physical_sizes`
    # and `height_scale_factor` has already been defined in the file and
    # one tries to override it, is done in another test.
    #
    # (only do this if not an NC file with units, since this is special: height_scale_factor
    # should only be choosable then.)
    if reader.format != 'nc' and 'unit' not in ch.info:
        topography = reader.topography(physical_sizes=physical_sizes_arg, height_scale_factor=height_scale_factor_arg)
        if hasattr(topography, 'scale_factor'):
            # sometimes we use height_scale_factor = 1 in the channel info in order
            # to denote that the height scale factor cannot be changed later.
            # This does not mean that the topography also really has been scaled, so
            # we compare the scale factor here only of it is available
            assert pytest.approx(exp_height_scale_factor) == topography.scale_factor, \
                "Difference in height scale factor between channel/argument and resulting topography"


@pytest.mark.parametrize('fn', text_example_file_list + text_example_without_size_file_list + binary_example_file_list)
def test_to_netcdf(fn):
    """Test that files can be stored as NetCDF and that reading then gives
    an identical topography object"""
    if fn in explicit_physical_sizes:
        t = read_topography(fn, physical_sizes=(1, 1))
    else:
        t = read_topography(fn)
    with tempfile.TemporaryDirectory() as d:
        tmpfn = f'{d}/netcdf_representation.nc'
        t.to_netcdf(tmpfn)
        t2 = read_topography(tmpfn)
        assert t == t2


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
    assert channel.unit == 'm'
    assert pytest.approx(
        channel.physical_sizes[0]) == 12.34 * 1e-6  # was given as µm
    assert pytest.approx(
        channel.physical_sizes[1]) == 5678.9 * 1e-9  # was given as nm

    #
    # test metadata of topography
    #
    topo = reader.topography()
    assert topo.unit == 'm'
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


def test_detect_format():
    assert detect_format(os.path.join(DATADIR, 'di1.di')) == 'di'
    assert detect_format(os.path.join(DATADIR, 'di2.di')) == 'di'
    assert detect_format(os.path.join(DATADIR, 'example.ibw')) == 'ibw'
    assert detect_format(os.path.join(DATADIR, 'example.opd')) == 'opd'
    assert detect_format(os.path.join(DATADIR, 'example.x3p')) == 'x3p'
    assert detect_format(os.path.join(DATADIR, 'example1.mat')) == 'mat'
    assert detect_format(os.path.join(DATADIR, 'example.xyz')) == 'xyz'
    assert detect_format(os.path.join(DATADIR, 'example-2d.xyz')) == 'xyz'
    assert detect_format(os.path.join(DATADIR, 'line_scan_1_minimal_spaces.asc')) == 'xyz'
    assert detect_format(os.path.join(DATADIR, 'example-2d.npy')) == 'npy'
    assert detect_format(os.path.join(DATADIR, 'surface.2048x2048.h5')) == 'h5'
    assert detect_format(os.path.join(DATADIR, 'example.zon')) == 'zon'


def test_to_matrix():
    y = np.arange(10).reshape((1, -1))
    x = np.arange(5).reshape((-1, 1))
    arr = -2 * y + 0 * x
    t = Topography(arr, (5, 10), unit='nm')
    # Check that we can export downstream the pipeline
    with tempfile.TemporaryDirectory() as d:
        t.to_matrix(f"{d}/topo.txt")
        t.detrend('center').to_matrix(f'{d}/topo.txt')
        t2 = read_matrix(f'{d}/topo.txt')
        np.testing.assert_allclose(t.detrend('center').heights(), t2.heights())
