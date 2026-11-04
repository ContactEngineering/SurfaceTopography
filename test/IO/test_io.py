#
# Copyright 2019-2023 Lars Pastewka
#           2022 Johannes Hörmann
#           2020-2021 Michael Röttger
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

import gzip
import io
import json
import numbers
import os
import pickle
import tempfile
import warnings

import NuMPI
import numpy as np
import pytest
from NuMPI.Testing.Fixtures import serial_only
from numpy.testing import assert_array_equal
from scipy.io import netcdf_file

import SurfaceTopography.IO
from SurfaceTopography import open_topography, read_topography
from SurfaceTopography.Exceptions import (
    CannotDetectFileFormat,
    MetadataAlreadyFixedByFile,
)
from SurfaceTopography.IO import detect_format, readers
from SurfaceTopography.IO.common import is_binary_stream
from SurfaceTopography.IO.Reader import ChannelInfo
from SurfaceTopography.IO.XYZ import XYZReader
from SurfaceTopography.Support.JSON import ExtendedJSONEncoder
from SurfaceTopography.UniformLineScanAndTopography import Topography

pytestmark = serial_only

DATADIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "file_format_examples"
)


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


binary_example_file_list = _convert_filelist(
    [
        "di-1.di",
        "di-2.di",
        "di-3.di",
        "di-4.di",
        "di-5.di",
        "di-6.di",
        "di-7.di",
        "ibw-1.ibw",
        "spot_1-1000nm.ibw",
        # 'surface.2048x2048.h5',
        "10x10-one_channel_without_name.ibw",
        "mat-1.mat",
        "opd-1.opd",
        "opd-2.opd",
        "opd-3.opd",
        "x3p-1.x3p",
        "x3p-2.x3p",
        "x3p-3.x3p",
        "x3p-4.x3p",
        "opdx-1.opdx",
        "opdx-2.opdx",
        "opdx-3.opdx",
        "mi-1.mi",
        "N46E013.hgt",
        "zon-1.zon",
        "nc-1.nc",
        "vk3-1.vk3",
        "vk4-1.vk4",
        "vk6-1.vk6",
        "sur-1.sur",
        "sur-2.sur",
        "sur-3.sur",
        "mitutoyo_mock.xlsx",
        "mitutoyo_nonuniform_mock.xlsx",
        "example_ps.tiff",
        "al3d-1.al3d",
        "nid-1.nid",
        "metropro-1.dat",
        "mdt-1.mdt",
        "gwy-1.gwy",
        "gwy-2.gwy",
        "plu-1.plu",
        "frt-1.frt",
        "frt-2.frt",
        "lext-1.lext",
        "lext-2.lext",
        "datx-1.datx",
        "datx-2.datx",
        "oir-1.oir",
        "poir-1.poir",
        "stp-1.stp",
        "top-1.top",
        "plux-1.plux",
        "jpk-1.jpk",
        "jpk-2.jpk",
        "qep-1.qep",
        "nmm-1.zip",
        # MPI I/O does not support Python streams
    ]
    + ([] if NuMPI._has_mpi4py else ["example-2d.npy"])
)

binary_without_stream_support_example_file_list = _convert_filelist(
    ["surface.2048x2048.h5"]
)

text_example_file_list = _convert_filelist(
    [
        "matrix-1.txt",
        "matrix-2.txt",
        "matrix-3.txt",
        "matrix-4.txt",
        "matrix-5.txt",
        "matrix-6.txt",
        # example8: from the reader's docstring, with extra newline at end
        "matrix-7.txt",
        "matrix-8.txt",
        "matrix-9.txt",
        "opdx-2.txt",
        "opdx-3.txt",
        "xyz-1.txt",
        "xyz-2.txt",
        "hfm-1.hfm",
        "dektak-1.csv",
        # Not yet working
        # 'not-yet-working-1.txt',
        "xy-1.txt",
        "xy-2.txt",
        "xy-3.txt",
        "xy-4.txt",
        "xy-5.txt",
        # 'xy-6.txt', # This has NaNs, which means equality tests fail
        "csv-1.csv",
        "csv-2.csv",
        "csv-3.csv",
    ]
)

explicit_physical_sizes = _convert_filelist(
    # Note: SRTM (.hgt) files do not carry physical sizes; the user needs
    # to provide them
    ["matrix-5.txt", "mat-1.mat", "example-2d.npy", "N46E013.hgt"]
)

text_example_memory_list = [
    """
            0 0
            1 2
            2 4
            3 6
            """
]


@pytest.mark.parametrize("reader", readers)
def test_no_resource_warning_on_failure(reader, file_format_examples):
    """
    Tests for each reader class that it doesn't raise a ResourceWarning
    """
    fn = os.path.join(file_format_examples, "wrongnpyfile.npy")
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


def test_uniform_stylus(file_format_examples):
    t = read_topography(os.path.join(file_format_examples, "xy-4.txt"))
    assert t.is_uniform


def test_read_gzip(file_format_examples):
    t = read_topography(gzip.open(os.path.join(file_format_examples, "matrix-4.txt.gz")))
    assert t.nb_grid_pts == (75, 305)


def test_cannot_detect_file_format_on_txt(file_format_examples):
    with pytest.raises(CannotDetectFileFormat):
        read_topography(os.path.join(file_format_examples, "nonsense_txt_file.txt"))


@pytest.mark.parametrize("fn", text_example_file_list)
def test_keep_text_file_open(fn):
    # Text file can be opened as binary or text
    with open(fn, "rb") as f:
        open_topography(f)
        assert (
            not f.closed
        ), f"Text file {fn} was opened as binary file and is closed, but should not"
    with open(fn, "r") as f:
        open_topography(f)
        assert (
            not f.closed
        ), f"Text file {fn} was opened as text file and is closed, but should not"


@pytest.mark.parametrize("fn", binary_example_file_list)
def test_keep_binary_file_open(fn):
    with open(fn, "rb") as f:
        open_topography(f)
        assert (
            not f.closed
        ), f"Binary file {fn} was opened as binary file and is closed, but should not"


@pytest.mark.parametrize("datastr", text_example_memory_list)
def test_keep_stream_from_memory_open(datastr):
    with io.StringIO(datastr) as f:
        open_topography(f)
        assert not f.closed, "text memory stream for '{}' was closed".format(datastr)

    # Doing the same when but only giving a binary stream
    with io.BytesIO(datastr.encode(encoding="utf-8")) as f:
        open_topography(f)
        assert not f.closed, "binary memory stream for '{}' was closed".format(datastr)


def test_is_binary_stream():
    # just grep a random existing file here
    fn = text_example_file_list[0]

    assert is_binary_stream(open(fn, mode="rb"))
    assert not is_binary_stream(open(fn, mode="r"))  # opened as text file

    # should also work with streams in memory
    assert is_binary_stream(io.BytesIO(b"11111"))  # some bytes in memory
    assert not is_binary_stream(io.StringIO("11111"))  # some bytes in memory


@pytest.mark.parametrize("fn", text_example_file_list + binary_example_file_list)
def test_can_be_pickled(fn):
    reader = open_topography(fn)
    physical_sizes = None
    if reader.default_channel.physical_sizes is None:
        physical_sizes = (1.0,) * reader.default_channel.dim

    topography = reader.topography(physical_sizes=physical_sizes)
    topographies = [topography]
    if hasattr(topography, "to_uniform"):
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
            for attr in ["dim", "physical_sizes", "is_periodic"]:
                assert getattr(x, attr) == getattr(y, attr)
            if x.physical_sizes is not None:
                assert_array_equal(x.positions(), y.positions())
                assert_array_equal(x.heights(), y.heights())


@pytest.mark.parametrize(
    "fn",
    text_example_file_list
    + binary_example_file_list
    + binary_without_stream_support_example_file_list,
)
def test_reader_arguments(fn):
    """Check whether all readers have channel, physical_sizes, height_scale_factor
    and unit arguments. Also check whether we can execute `topography` multiple times
    for all readers"""
    physical_sizes0 = (1.2, 1.3)
    unit0 = "mm"
    height_scale_factor0 = 1

    # Test open -> topography
    r = open_topography(fn)
    physical_sizes = (
        None if r.channels[0].physical_sizes is not None else physical_sizes0
    )
    unit = None if r.channels[0].unit is not None else unit0
    height_scale_factor = (
        None if r.channels[0].height_scale_factor is not None else height_scale_factor0
    )

    info = r.channels[0].info.copy()

    t = r.topography(
        channel_index=0,
        physical_sizes=physical_sizes,
        height_scale_factor=height_scale_factor,
        unit=unit,
    )
    if physical_sizes is not None:
        assert t.physical_sizes == physical_sizes
    if unit is not None:
        assert t.unit == unit
    if height_scale_factor is not None:
        assert t.height_scale_factor == height_scale_factor
    assert t.info == info
    # Second call to topography
    t2 = r.topography(
        channel_index=0,
        physical_sizes=physical_sizes,
        height_scale_factor=height_scale_factor,
        unit=unit,
    )
    if physical_sizes is not None:
        assert t2.physical_sizes == physical_sizes
    if unit is not None:
        assert t2.unit == unit
    if height_scale_factor is not None:
        assert t.height_scale_factor == height_scale_factor
    assert_array_equal(t.heights(), t2.heights())
    assert t2.info == info
    # Test read_topography
    t = read_topography(
        fn,
        channel_index=0,
        physical_sizes=physical_sizes,
        height_scale_factor=height_scale_factor,
        unit=unit,
    )
    if physical_sizes is not None:
        assert t.physical_sizes == physical_sizes
    if unit is not None:
        assert t.unit == unit
    if height_scale_factor is not None:
        assert t.height_scale_factor == height_scale_factor
    assert t.info == info

    print(type(t.has_undefined_data))
    assert isinstance(t.has_undefined_data, bool) or isinstance(
        t.has_undefined_data, np.bool_
    )


@pytest.mark.parametrize("fn", text_example_file_list + binary_example_file_list)
def test_readers_with_binary_file_object(fn):
    """Check whether all readers have channel, physical_sizes and
    height_scale_factor arguments. Also check whether we can execute
    `topography` multiple times for all readers"""
    physical_sizes0 = (1.2, 1.3)

    # Test open -> topography
    r = open_topography(open(fn, mode="rb"))
    physical_sizes = (
        None if r.channels[0].physical_sizes is not None else physical_sizes0
    )
    t = r.topography(
        channel_index=0, physical_sizes=physical_sizes, height_scale_factor=None
    )
    assert t.dim == len(t.physical_sizes)
    if physical_sizes is not None:
        assert t.physical_sizes == physical_sizes
    if t.dim == 2:
        sx, sy = t.physical_sizes
        assert isinstance(sx, float) or isinstance(sx, np.float64)
        assert isinstance(sy, float) or isinstance(sy, np.float64)
    else:
        (sx,) = t.physical_sizes
        assert isinstance(sx, float) or isinstance(sx, np.float64)
    # Second call to topography
    t2 = r.topography(
        channel_index=0, physical_sizes=physical_sizes, height_scale_factor=None
    )
    if physical_sizes is not None:
        assert t2.physical_sizes == physical_sizes
    assert_array_equal(t.heights(), t2.heights(), err_msg=fn)


@pytest.mark.parametrize("fn", text_example_file_list + binary_example_file_list)
def test_nb_grid_pts_and_physical_sizes_are_tuples_or_none(fn):
    r = open_topography(fn)
    assert isinstance(
        r.default_channel.nb_grid_pts, tuple
    ), f"{fn} - {r.__class__}: {r.default_channel.nb_grid_pts}"
    if r.default_channel.physical_sizes is not None:
        assert isinstance(
            r.default_channel.physical_sizes, tuple
        ), f"{fn} - {r.__class__}: {r.default_channel.physical_sizes}"
        # If it is a tuple, it cannot contains None's
        assert np.all(
            [p is not None for p in r.default_channel.physical_sizes]
        ), f"{fn} - {r.__class__}: {r.default_channel.physical_sizes}"


@pytest.mark.parametrize(
    "fn",
    text_example_file_list
    + binary_example_file_list
    + binary_without_stream_support_example_file_list,
)
def test_channel_info_and_topography_have_same_metadata(fn):
    """
    Tests that properties like physical sizes, units and nb_grid_pts are
    the same in the ChannelInfo and the loaded topography.
    """

    reader = open_topography(fn)

    for index, channel in enumerate(reader.channels):
        # some basic consistency checks
        assert isinstance(channel, ChannelInfo)
        assert channel.index == index
        assert reader.channels[index] == channel

        # check number of grid points
        foo_str = reader.format() + "-%d" % (channel.index,)  # unique for each channel
        topography = channel.topography(
            physical_sizes=(1, 1) if channel.physical_sizes is None else None,
            info=dict(datafile=dict(foo=foo_str)),
        )
        assert channel.nb_grid_pts == topography.nb_grid_pts
        assert topography.nb_grid_pts == topography.heights().shape

        # some checks on info dict in channel and topography
        assert topography.info["datafile"] == dict(foo=foo_str)
        if channel.unit is not None or topography.unit is not None:
            assert channel.unit == topography.unit

        if channel.physical_sizes is not None:
            assert channel.physical_sizes == topography.physical_sizes

        if channel.height_scale_factor is not None:
            assert channel.height_scale_factor == topography.height_scale_factor
        else:
            assert not hasattr(topography, "height_scale_factor")

        if channel.is_periodic is not None:
            assert isinstance(channel.is_periodic, (bool, np.bool_))
            assert channel.is_periodic == topography.is_periodic

        assert isinstance(channel.is_uniform, (bool, np.bool_))
        assert channel.is_uniform == topography.is_uniform

        if channel.has_undefined_data is not None:
            assert isinstance(channel.has_undefined_data, (bool, np.bool_))
            assert channel.has_undefined_data == topography.has_undefined_data


@pytest.mark.parametrize("fn", text_example_file_list + binary_example_file_list)
def test_reader_args_doesnt_overwrite_data_from_file(fn):
    """
    Tests that if some properties like `physical_sizes and `height_scale_factor`
    are given in the file, they cannot be overridden by given arguments to
    the .topography() method.
    """
    reader = open_topography(fn)
    ch = reader.default_channel
    physical_sizes_arg_if_missing_in_file = (1.0,) * ch.dim
    physical_sizes_arg = (
        physical_sizes_arg_if_missing_in_file if ch.physical_sizes is None else None
    )

    if ch.physical_sizes is not None:
        with pytest.raises(MetadataAlreadyFixedByFile):
            reader.topography(physical_sizes=physical_sizes_arg_if_missing_in_file)

    if ch.height_scale_factor is not None:
        with pytest.raises(MetadataAlreadyFixedByFile):
            if ch.physical_sizes is None:
                reader.topography(
                    physical_sizes=physical_sizes_arg, height_scale_factor=10
                )
            else:
                # if an exception happens, we want it because of height scale factor
                reader.topography(height_scale_factor=10)

    # A small problem with this test is maybe that there are a few input
    # files which pass this test without any assert, so it looks like
    # a passed test, but it has no meaning. Since this are only three files
    # by now, I think this is okay, sorting this out would be difficult.


@pytest.mark.parametrize("fn", text_example_file_list + binary_example_file_list)
def test_periodic_flag(fn):
    reader = open_topography(fn)
    ch = reader.default_channel
    physical_sizes_arg_if_missing_in_file = (1.0,) * ch.dim
    physical_sizes_arg = (
        physical_sizes_arg_if_missing_in_file if ch.physical_sizes is None else None
    )

    value_error_thrown = False
    try:
        t = reader.topography(physical_sizes=physical_sizes_arg, periodic=True)
        assert t.is_periodic, fn
    except ValueError:
        value_error_thrown = True

    try:
        t = reader.topography(physical_sizes=physical_sizes_arg, periodic=False)
        assert not t.is_periodic, fn
    except ValueError:
        assert not value_error_thrown


@pytest.mark.parametrize("fn", text_example_file_list + binary_example_file_list)
def test_reader_height_scale_factor_arg_for_topography(fn):
    """Test whether height_scale_factor can be given to .topography() and is effective.

    Also checking whether the reader channels have .height_scale_factor attribute and
    whether it is equal to the scaling factor known from topography.

    Also tests that info dict of channel and topography have no height_scale_factor,
    because this should be a channel property now.
    """
    reader = open_topography(fn)
    ch = reader.default_channel

    assert hasattr(ch, "height_scale_factor")
    assert "height_scale_factor" not in ch.info

    height_scale_factor_if_missing_in_file = 2  # just some number

    # calculate argument for .topography()
    height_scale_factor_arg = (
        height_scale_factor_if_missing_in_file
        if ch.height_scale_factor is None
        else None
    )

    # which factor we expect at the end
    exp_height_scale_factor = (
        height_scale_factor_if_missing_in_file
        if ch.height_scale_factor is None
        else ch.height_scale_factor
    )

    # in order to call .topography(), we also need valid physical_sizes
    physical_sizes_arg_if_missing_in_file = (1.0,) * ch.dim
    physical_sizes_arg = (
        physical_sizes_arg_if_missing_in_file if ch.physical_sizes is None else None
    )

    # The check whether an exception is raised if meta data like `physical_sizes`
    # and `height_scale_factor` has already been defined in the file and
    # one tries to override it, is done in another test.
    #
    # (only do this if not an NC file with units, since this is special: height_scale_factor
    # should only be choosable then.)
    if reader.format != "nc" and not ch.unit:
        topography = reader.topography(
            physical_sizes=physical_sizes_arg,
            height_scale_factor=height_scale_factor_arg,
        )
        if hasattr(topography, "scale_factor"):
            # sometimes we use height_scale_factor = 1 in the channel info in order
            # to denote that the height scale factor cannot be changed later.
            # This does not mean that the topography also really has been scaled, so
            # we compare the scale factor here only of it is available
            assert (
                pytest.approx(exp_height_scale_factor) == topography.scale_factor
            ), "Difference in height scale factor between channel/argument and resulting topography"


@pytest.mark.parametrize("fn", text_example_file_list + binary_example_file_list)
def test_info_dict_has_no_nans_and_no_tuples(fn):
    """Check that readers don't return NaNs and no tuples as those don't serialize correctly to JSON"""
    if fn in explicit_physical_sizes:
        t = read_topography(fn, physical_sizes=(1, 1))
    else:
        t = read_topography(fn)

    def assert_no_nans_and_no_tuples(d):
        if isinstance(d, dict):
            for key, value in d.items():
                assert_no_nans_and_no_tuples(value)
        elif isinstance(d, list):
            for value in d:
                assert_no_nans_and_no_tuples(value)
        elif isinstance(d, numbers.Number):
            assert d == d
        else:
            assert not isinstance(d, tuple)

    assert_no_nans_and_no_tuples(t.info)


@pytest.mark.parametrize("fn", text_example_file_list + binary_example_file_list)
def test_to_netcdf(fn):
    """Test that files can be stored as NetCDF and that reading then gives
    an identical topography object"""
    if fn in explicit_physical_sizes:
        t = read_topography(fn, physical_sizes=(1, 1))
    else:
        t = read_topography(fn)
    with tempfile.TemporaryDirectory() as d:
        tmpfn = f"{d}/netcdf_representation.nc"
        t.to_netcdf(tmpfn)

        # Try reading with surface topography
        t2 = read_topography(tmpfn)
        assert (
            t.info == t2.info
        )  # We check the info dictionary separately as this is often a source of issues
        assert t == t2

        # Try reading directly
        nc = netcdf_file(tmpfn)
        nc_h = nc.variables["heights"][...]
        nc_x = nc.variables["x"][...]
        if "y" in nc.variables:
            nc_y = nc.variables["y"][...]
            x, y, h = t.positions_and_heights()

            nc_x = np.repeat(nc_x.reshape((-1, 1)), x.shape[1], axis=1)
            nc_y = np.repeat(nc_y.reshape((1, -1)), y.shape[0], axis=0)

            np.testing.assert_allclose(nc_h, h)
            np.testing.assert_allclose(nc_x, x)
            np.testing.assert_allclose(nc_y, y)
        else:
            x, h = t.positions_and_heights()

            np.testing.assert_allclose(nc_h, h)
            np.testing.assert_allclose(nc_x, x)


def test_read_unknown_file_format(file_format_examples):
    with pytest.raises(SurfaceTopography.IO.UnknownFileFormat):
        SurfaceTopography.IO.open_topography(
            os.path.join(file_format_examples, "surface.2048x2048.h5"),
            format="Nonexistentfileformat",
        )


def test_detect_format_unknown_file_format(file_format_examples):
    with pytest.raises(SurfaceTopography.Exceptions.UnknownFileFormat):
        SurfaceTopography.IO.open_topography(
            os.path.join(file_format_examples, "surface.2048x2048.h5"),
            format="Nonexistentfileformat",
        )


def test_file_format_mismatch(file_format_examples):
    with pytest.raises(SurfaceTopography.Exceptions.FileFormatMismatch):
        SurfaceTopography.IO.open_topography(
            os.path.join(file_format_examples, "surface.2048x2048.h5"), format="npy"
        )


def test_line_scan_detect_format_then_read(file_format_examples):
    assert detect_format(os.path.join(file_format_examples, "xy-3.txt")) == "xyz"


def test_line_scan_read(file_format_examples):
    surface = XYZReader(os.path.join(file_format_examples, "xy-3.txt")).topography()

    assert not surface.is_uniform
    assert surface.dim == 1

    x, y = surface.positions_and_heights()
    assert len(x) > 0
    assert len(x) == len(y)


@pytest.mark.parametrize("reader", readers)
def test_readers_have_name(reader):
    reader.name()


# yes, the German version still has "Value units"
@pytest.mark.parametrize("lang_filename_infix", ["english", "german"])
def test_gwyddion_txt_import(lang_filename_infix, file_format_examples):
    fname = os.path.join(
        file_format_examples, "gwyddion-export-{}.txt".format(lang_filename_infix)
    )

    #
    # test channel infos
    #
    reader = open_topography(fname)

    assert len(reader.channels) == 1
    channel = reader.default_channel

    assert channel.name == "My Channel Name"
    assert channel.unit == "m"
    assert pytest.approx(channel.physical_sizes[0]) == 12.34 * 1e-6  # was given as µm
    assert pytest.approx(channel.physical_sizes[1]) == 5678.9 * 1e-9  # was given as nm

    #
    # test metadata of topography
    #
    topo = reader.topography()
    assert topo.unit == "m"
    assert pytest.approx(topo.physical_sizes[0]) == 12.34 * 1e-6  # was given as µm
    assert pytest.approx(topo.physical_sizes[1]) == 5678.9 * 1e-9  # was given as nm

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
    heights_in_file = [[1, 1.5, 3], [-2, -3, -6], [0, 0, 0], [9, 9, 9]]

    expected_heights = np.array(heights_in_file).T

    np.testing.assert_allclose(topo.heights(), expected_heights)


def test_detect_format(file_format_examples):
    assert detect_format(os.path.join(file_format_examples, "di-1.di")) == "di"
    assert detect_format(os.path.join(file_format_examples, "di-2.di")) == "di"
    with pytest.raises(CannotDetectFileFormat):
        detect_format(os.path.join(file_format_examples, "di_corrupted.di"))
    assert detect_format(os.path.join(file_format_examples, "ibw-1.ibw")) == "ibw"
    assert detect_format(os.path.join(file_format_examples, "opd-1.opd")) == "opd"
    assert detect_format(os.path.join(file_format_examples, "x3p-1.x3p")) == "x3p"
    assert detect_format(os.path.join(file_format_examples, "x3p-2.x3p")) == "x3p"
    assert detect_format(os.path.join(file_format_examples, "x3p-3.x3p")) == "x3p"
    assert detect_format(os.path.join(file_format_examples, "x3p-4.x3p")) == "x3p"
    assert detect_format(os.path.join(file_format_examples, "mat-1.mat")) == "mat"
    assert detect_format(os.path.join(file_format_examples, "xy-1.txt")) == "xyz"
    assert detect_format(os.path.join(file_format_examples, "xy-2.txt")) == "xyz"
    assert detect_format(os.path.join(file_format_examples, "xyz-1.txt")) == "xyz"
    assert detect_format(os.path.join(file_format_examples, "xy-3.txt")) == "xyz"
    assert detect_format(os.path.join(file_format_examples, "example-2d.npy")) == "npy"
    assert (
        detect_format(os.path.join(file_format_examples, "surface.2048x2048.h5"))
        == "h5"
    )
    assert detect_format(os.path.join(file_format_examples, "zon-1.zon")) == "zon"
    assert detect_format(os.path.join(file_format_examples, "vk3-1.vk3")) == "vk"
    assert detect_format(os.path.join(file_format_examples, "vk4-1.vk4")) == "vk"
    assert detect_format(os.path.join(file_format_examples, "vk6-1.vk6")) == "vk"
    assert (
        detect_format(os.path.join(file_format_examples, "mitutoyo_mock.xlsx"))
        == "mitutoyo"
    )
    assert (
        detect_format(
            os.path.join(file_format_examples, "mitutoyo_nonuniform_mock.xlsx")
        )
        == "mitutoyo"
    )
    assert detect_format(os.path.join(file_format_examples, "al3d-1.al3d")) == "al3d"
    assert detect_format(os.path.join(file_format_examples, "sur-1.sur")) == "sur"
    assert detect_format(os.path.join(file_format_examples, "example_ps.tiff")) == "ps"
    assert (
        detect_format(os.path.join(file_format_examples, "metropro-1.dat"))
        == "metropro"
    )
    assert detect_format(os.path.join(file_format_examples, "gwy-1.gwy")) == "gwy"
    assert detect_format(os.path.join(file_format_examples, "plu-1.plu")) == "plu"
    assert detect_format(os.path.join(file_format_examples, "frt-1.frt")) == "frt"
    assert detect_format(os.path.join(file_format_examples, "frt-2.frt")) == "frt"
    assert detect_format(os.path.join(file_format_examples, "hfm-1.hfm")) == "xyz"
    assert detect_format(os.path.join(file_format_examples, "lext-1.lext")) == "lext"
    assert detect_format(os.path.join(file_format_examples, "datx-1.datx")) == "datx"
    assert detect_format(os.path.join(file_format_examples, "oir-1.oir")) == "oir"
    assert detect_format(os.path.join(file_format_examples, "poir-1.poir")) == "poir"
    assert detect_format(os.path.join(file_format_examples, "stp-1.stp")) == "wsxm"
    assert detect_format(os.path.join(file_format_examples, "top-1.top")) == "wsxm"
    assert detect_format(os.path.join(file_format_examples, "plux-1.plux")) == "plux"
    assert detect_format(os.path.join(file_format_examples, "jpk-1.jpk")) == "jpk"
    assert detect_format(os.path.join(file_format_examples, "dektak-1.csv")) == "xyz"


def test_to_matrix():
    y = np.arange(10).reshape((1, -1))
    x = np.arange(5).reshape((-1, 1))
    arr = -2 * y + 0 * x
    t = Topography(arr, (5, 10), unit="nm")
    # Check that we can export downstream the pipeline
    with tempfile.TemporaryDirectory() as d:
        t.to_matrix(f"{d}/topo.txt")
        t.detrend("center").to_matrix(f"{d}/topo.txt")
        t2 = read_topography(f"{d}/topo.txt")
        np.testing.assert_allclose(t.detrend("center").heights(), t2.heights())


@pytest.mark.parametrize("unit", [None, "µm"])
def test_to_matrix_round_trip(unit):
    # The text reader reproduces heights, sizes, unit and undefined points
    from SurfaceTopography import UniformLineScan
    heights = np.ma.masked_array(np.arange(20.).reshape(5, 4) ** 1.5, mask=np.arange(20).reshape(5, 4) == 7)
    # A fully undefined x-profile is written as a column of NaNs
    profile_masked = np.ma.masked_array(np.arange(12.).reshape(4, 3), mask=np.arange(12).reshape(4, 3) // 3 == 1)
    for t in [Topography(heights, (2.5, 3.), unit=unit), UniformLineScan(np.arange(7.) ** 2, 3.5, unit=unit),
              # Physical sizes that are NumPy scalars
              Topography(heights, (np.float64(2.5), np.float32(3.)), unit=unit),
              # Small maps whose files have two or three columns, which are
              # also valid coordinate (XYZ) data
              Topography(np.arange(8.).reshape(2, 4) ** 2, (2.5, 3.), unit=unit),
              Topography(np.arange(8.).reshape(4, 2) ** 2, (2.5, 3.), unit=unit),
              Topography(np.arange(6.).reshape(3, 2) ** 2, (2.5, 3.), unit=unit),
              Topography(np.arange(6.).reshape(2, 3) ** 2, (2.5, 3.), unit=unit),
              Topography(profile_masked, (2.5, 3.), unit=unit)]:
        with tempfile.TemporaryDirectory() as d:
            t.to_matrix(f"{d}/topo.txt")
            reader = open_topography(f"{d}/topo.txt")
            assert reader.format() == "asc"
            t2 = reader.topography()
            assert t2.nb_grid_pts == t.nb_grid_pts
            np.testing.assert_allclose(t2.physical_sizes, t.physical_sizes)
            assert t2.unit == unit
            np.testing.assert_array_equal(np.ma.getmaskarray(t2.heights()), np.ma.getmaskarray(t.heights()))
            np.testing.assert_allclose(np.ma.asarray(t2.heights()).compressed(),
                                       np.ma.asarray(t.heights()).compressed())


@pytest.mark.parametrize(
    "fn",
    text_example_file_list
    + binary_example_file_list
    + binary_without_stream_support_example_file_list,
)
def test_json_encode_info(fn):
    """Check that info dictionary can be serialized to JSON"""
    physical_sizes0 = (1.2, 1.3)
    unit0 = "mm"
    height_scale_factor0 = 1

    r = open_topography(fn)
    physical_sizes = (
        None if r.channels[0].physical_sizes is not None else physical_sizes0
    )
    unit = None if r.channels[0].unit is not None else unit0
    height_scale_factor = (
        None if r.channels[0].height_scale_factor is not None else height_scale_factor0
    )

    t = r.topography(
        channel_index=0,
        physical_sizes=physical_sizes,
        height_scale_factor=height_scale_factor,
        unit=unit,
    )

    # This should simply pass without an exception
    json.dumps(t.info, cls=ExtendedJSONEncoder)


@pytest.mark.parametrize("fn", text_example_file_list + binary_example_file_list)
def test_channel_ids_and_height_indices(fn):
    # Every channel has a unique, stable identifier by which it can be
    # selected; height channels are numbered consecutively
    reader = open_topography(fn)
    channels = reader.channels
    ids = [ch.channel_id for ch in channels]
    assert None not in ids, fn
    assert len(set(ids)) == len(ids), fn
    for ch in channels:
        assert reader._get_channel_by_id(ch.channel_id).index == ch.index, fn
    height_indices = [ch.height_index for ch in channels if ch.is_height_channel]
    assert height_indices == list(range(len(height_indices))), fn
    assert all(ch.height_index is None for ch in channels if not ch.is_height_channel), fn


@pytest.mark.parametrize("fn", text_example_file_list + binary_example_file_list)
def test_channel_topography_has_periodicity_of_reader(fn):
    # Without `periodic`, the channel and the reader return the periodicity
    # stored in the file (or False); it is always a bool
    reader = open_topography(fn)
    ch = reader.default_channel
    physical_sizes = (1.0,) * ch.dim if ch.physical_sizes is None else None
    t_reader = reader.topography(physical_sizes=physical_sizes)
    t_channel = ch.topography(physical_sizes=physical_sizes)
    assert isinstance(t_reader.is_periodic, bool), fn
    assert t_channel.is_periodic == t_reader.is_periodic, fn


def test_periodicity_from_netcdf_file():
    from SurfaceTopography import Topography
    with tempfile.TemporaryDirectory() as d:
        fn = os.path.join(d, "periodic.nc")
        Topography(np.arange(48.).reshape(8, 6), (1., 1.), periodic=True, unit="m").to_netcdf(fn)
        reader = open_topography(fn)
        assert reader.channels[0].is_periodic
        assert reader.topography().is_periodic
        assert reader.channels[0].topography().is_periodic
        assert not reader.channels[0].topography(periodic=False).is_periodic


@pytest.mark.parametrize("fn", text_example_file_list + binary_example_file_list)
def test_data_kind_matches_data_unit(fn):
    # Channels whose values are not lengths are not height channels, and
    # their lateral unit is a length unit
    from SurfaceTopography.Support.UnitConversion import is_length_unit
    for ch in open_topography(fn).channels:
        if isinstance(ch.data_unit, str) and not is_length_unit(ch.data_unit):
            assert not ch.is_height_channel, f"{fn}: {ch.name}"
            assert ch.unit is None or is_length_unit(ch.unit), f"{fn}: {ch.name}"


@pytest.mark.parametrize("fn", text_example_file_list + binary_example_file_list)
def test_detect_format_with_communicator(fn):
    # Detection with a communicator gives the same result as without
    from NuMPI import MPI
    assert detect_format(fn, comm=MPI.COMM_WORLD) == detect_format(fn)
    assert open_topography(fn, communicator=MPI.COMM_WORLD).format() == detect_format(fn)


def test_open_non_parallel_format_with_parallel_communicator(file_format_examples):
    # A file whose reader does not support parallel I/O raises an error that
    # names the format, during detection as well as with explicit format
    class FakeCommunicator:
        size = 2
    fn = os.path.join(file_format_examples, "di-1.di")
    with pytest.raises(ValueError, match="'di'.*does not support parallel I/O"):
        open_topography(fn, communicator=FakeCommunicator())
    with pytest.raises(ValueError, match="does not support parallel I/O"):
        open_topography(fn, format="di", communicator=FakeCommunicator())


def _truncation_examples():
    # All example files whose format is detected
    examples = []
    for fn in text_example_file_list + binary_example_file_list:
        try:
            detect_format(fn)
        except Exception:
            continue
        examples += [fn]
    return examples


@pytest.mark.parametrize("nb_bytes", range(8, 64))
def test_hdf5_truncated_in_superblock(file_format_examples, nb_bytes):
    # Files that end right after the signature or within the superblock
    from SurfaceTopography.Exceptions import CorruptFile
    from SurfaceTopography.IO.common import check_hdf5_size
    data = open(os.path.join(file_format_examples, "multiple_data_sets.h5"), "rb").read()
    with pytest.raises(CorruptFile):
        check_hdf5_size(io.BytesIO(data[:nb_bytes]))


@pytest.mark.parametrize("fn", [fn for fn in text_example_file_list if detect_format(fn) == "xyz"])
def test_text_file_without_final_line_break(fn):
    # Complete text files whose last line has no line break (and e.g.
    # trailing empty fields, as in Dektak CSV files) are not truncated
    data = open(fn, "rb").read()
    complete = open_topography(fn).topography()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):  # UTF-16 with byte order mark
        data = data.decode("utf-16").rstrip("\r\n").encode("utf-16")
    else:
        data = data.rstrip(b"\r\n")
    reader = open_topography(io.BytesIO(data), format="xyz")
    np.testing.assert_allclose(reader.topography().heights(), complete.heights())


@pytest.mark.parametrize("fn", _truncation_examples())
@pytest.mark.parametrize("fraction", [0.5, 0.9])
def test_truncated_file_raises_corrupt_file(fn, fraction):
    # A truncated file either still yields a topography or raises a
    # CorruptFile (or another error of the library), never a raw error of
    # numpy or struct
    from SurfaceTopography.Exceptions import CannotPerformAnalysisError, ReadFileError
    data = open(fn, "rb").read()
    fmt = detect_format(fn)
    stream = io.BytesIO(data[: int(len(data) * fraction)])
    try:
        reader = open_topography(stream, format=fmt)
        for ch in reader.channels:
            reader.topography(channel_index=ch.index,
                              physical_sizes=None if ch.physical_sizes is not None else (1.0,) * ch.dim)
    except (ReadFileError, CannotPerformAnalysisError):
        pass
