#
# Copyright 2020-2021, 2023 Lars Pastewka
#           2019-2020 Antoine Sanner
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
import os

import numpy as np
import pytest
from NuMPI import MPI

from SurfaceTopography import read_topography
from SurfaceTopography.Exceptions import CorruptFile, UnsupportedFormatFeature
from SurfaceTopography.IO import DIReader

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")


def test_di_date(file_format_examples):
    t = read_topography(os.path.join(file_format_examples, 'di-1.di'))
    assert t.info['acquisition_time'] == datetime.datetime(2016, 1, 12, 9, 57, 48)
    assert t.info['instrument']['name'] == 'Dimension V'
    assert t.info['instrument']['vendor'] == 'Bruker'


def test_4byte_data(file_format_examples):
    r = DIReader(os.path.join(file_format_examples, 'di-5.di'))
    t = r.topography()
    np.testing.assert_allclose(t.rms_height_from_area(), 5.831926)
    assert t.info['instrument']['name'] == 'Dimension Icon'
    assert t.info['instrument']['vendor'] == 'Bruker'


def test_corrupted_file(file_format_examples):
    # Corruption should be detected when opening file; subsequent calls to `topography` must succeed
    with pytest.raises(CorruptFile):
        DIReader(os.path.join(file_format_examples, 'di_corrupted.di'))


def test_di7(file_format_examples):
    r = DIReader(os.path.join(file_format_examples, 'di-7.di'))
    r.topography()


###
# Synthetic files exercising format variants (cf. Gwyddion's nanoscope.c)
###

_HEADER_SIZE = 4096


def _write_di(path, sections, blocks, first_line=None):
    """
    Write a minimal Nanoscope file.

    Parameters
    ----------
    path : str
        Output file name
    sections : list of (str, list of (str, str))
        Header sections; each image section must have a 'Data offset'
        entry of `None`, which is filled in automatically
    blocks : list of bytes
        Binary data blocks, one per image section (in order)
    first_line : str, optional
        Replace the first header line (for testing magic variants)
    """
    lines = []
    offset = _HEADER_SIZE
    block_iter = iter(blocks)
    for name, entries in sections:
        lines.append(f"\\*{name}")
        for key, value in entries:
            if key == "Data offset" and value is None:
                block = next(block_iter)
                value = str(offset)
                offset += len(block)
            lines.append(f"\\{key}: {value}")
    lines.append("\\*File list end")
    if first_line is not None:
        lines[0] = first_line
    header = "\r\n".join(lines).encode("latin-1") + b"\r\n"
    assert len(header) <= _HEADER_SIZE
    header += b"\0" * (_HEADER_SIZE - len(header))
    with open(path, "wb") as f:
        f.write(header)
        for block in blocks:
            f.write(block)


def _file_list(version="0x09100000", file_list_name="File list"):
    return (file_list_name, [
        ("Version", version),
        ("Date", "09:57:48 AM Tue Jan 12 2016"),
        ("Start context", "OL2"),
        ("Data length", str(_HEADER_SIZE)),
    ])


def _scanner_list(name="Scanner list"):
    return (name, [("@Sens. Zsens", "V 20.0 nm/V")])


def _scan_list(nx, ny, extra=(), name="Ciao scan list"):
    return (name, [
        ("Operating mode", "Image"),
        ("Scan Size", "500 nm"),
        ("Samps/line", str(nx)),
        ("Lines", str(ny)),
        ("Aspect Ratio", "1:1"),
    ] + list(extra))


def _image_list(nx, ny, length, bytes_per_pixel=2, scan_size="500 500 nm",
                image_data='S [Height] "Height"',
                z_scale="V [Sens. Zsens] (0.005035400 V/LSB) 3.0 V",
                z_scale_key="@2:Z scale", extra=(), name="Ciao image list"):
    return (name, [
        ("Data offset", None),
        ("Data length", str(length)),
        ("Bytes/pixel", str(bytes_per_pixel)),
        ("Start context", "OL"),
        ("Samps/line", str(nx)),
        ("Number of lines", str(ny)),
        ("Aspect Ratio", "1:1"),
        ("Scan Size", scan_size),
        ("@2:Image Data", image_data),
        (z_scale_key, z_scale),
    ] + list(extra))


def _raw(nx, ny):
    # Rows of the raster are stored one after another
    return (np.arange(nx * ny).reshape(ny, nx) * 101 % 2001 - 1000)


def _expected(raw, scale):
    # First stored line is the bottom of the image (like Gwyddion)
    return np.fliplr(raw.T) * scale


def test_synthetic_16bit(tmp_path):
    nx, ny = 6, 4
    raw = _raw(nx, ny)
    fn = str(tmp_path / "test.spm")
    _write_di(fn, [_file_list(), _scanner_list(), _scan_list(nx, ny),
                   _image_list(nx, ny, 2 * nx * ny)],
              [raw.astype("<i2").tobytes()])
    r = DIReader(fn)
    (ch,) = r.channels
    assert ch.name == "Height"
    assert ch.nb_grid_pts == (nx, ny)
    assert ch.unit == "nm"
    t = r.topography()
    np.testing.assert_allclose(t.heights(), _expected(raw, 3.0 / 65536 * 20.0))
    np.testing.assert_allclose(t.physical_sizes, (500, 500))


@pytest.mark.parametrize("bytes_per_pixel", [2, 4])
def test_32bit_storage_from_version_9_2(tmp_path, bytes_per_pixel):
    # Starting with version 9.2, data is always stored as 32-bit integers;
    # 'Bytes/pixel' only determines the scaling of the integers
    nx, ny = 6, 4
    raw = _raw(nx, ny)
    fn = str(tmp_path / "test.spm")
    _write_di(fn, [_file_list("0x09200000"), _scanner_list(), _scan_list(nx, ny),
                   _image_list(nx, ny, 4 * nx * ny, bytes_per_pixel=bytes_per_pixel)],
              [raw.astype("<i4").tobytes()])
    t = DIReader(fn).topography()
    np.testing.assert_allclose(
        t.heights(), _expected(raw, 3.0 / 256**bytes_per_pixel * 20.0))


def test_single_number_scan_size(tmp_path):
    # Old files have a single number in 'Scan size' for square scans
    nx, ny = 6, 4
    raw = _raw(nx, ny)
    fn = str(tmp_path / "test.spm")
    _write_di(fn, [_file_list(), _scanner_list(), _scan_list(nx, ny),
                   _image_list(nx, ny, 2 * nx * ny, scan_size="2 ~m")],
              [raw.astype("<i2").tobytes()])
    (ch,) = DIReader(fn).channels
    # Lateral sizes are converted to the height unit
    np.testing.assert_allclose(ch.physical_sizes, (2000, 2000))
    assert ch.unit == "nm"


def test_slow_axis_size(tmp_path):
    # The second number in 'Scan size' of the image section can be wrong;
    # 'Slow Axis Size' in the scan list holds the correct value
    nx, ny = 8, 4
    raw = _raw(nx, ny)
    fn = str(tmp_path / "test.spm")
    _write_di(fn, [_file_list("0x10000102"), _scanner_list(),
                   _scan_list(nx, ny, extra=[("Slow Axis Size", "250 nm")]),
                   _image_list(nx, ny, 4 * nx * ny, bytes_per_pixel=4, scan_size="0.5 0.5 ~m")],
              [raw.astype("<i4").tobytes()])
    (ch,) = DIReader(fn).channels
    np.testing.assert_allclose(ch.physical_sizes, (500, 250))


def test_slow_axis_size_capture_now(tmp_path):
    # Non-square scan that was stopped before all lines were recorded
    nx, ny = 8, 2
    raw = _raw(nx, ny)
    fn = str(tmp_path / "test.spm")
    image = _image_list(nx, ny, 4 * nx * ny, bytes_per_pixel=4, scan_size="0.5 0.5 ~m")
    image[1][6] = ("Aspect Ratio", "2:1")
    _write_di(fn, [_file_list("0x10000102"), _scanner_list(),
                   _scan_list(nx, 4, extra=[("Slow Axis Size", "250 nm")]),
                   image],
              [raw.astype("<i4").tobytes()])
    (ch,) = DIReader(fn).channels
    assert ch.nb_grid_pts == (nx, ny)
    np.testing.assert_allclose(ch.physical_sizes, (500, 125))


def test_global_resolution(tmp_path):
    # Some files report bogus resolutions in the image sections; the size
    # of the data block then matches the global resolution of the scan list
    nx, ny = 4, 4
    raw = _raw(nx, ny)
    fn = str(tmp_path / "test.spm")
    _write_di(fn, [_file_list("0x05300001"), _scanner_list(), _scan_list(nx, ny),
                   _image_list(8, 8, 2 * nx * ny)],
              [raw.astype("<i2").tobytes()])
    r = DIReader(fn)
    (ch,) = r.channels
    assert ch.nb_grid_pts == (nx, ny)
    np.testing.assert_allclose(ch.physical_sizes, (250, 250))
    np.testing.assert_allclose(r.topography().heights(), _expected(raw, 3.0 / 65536 * 20.0))


def test_alternative_section_names(tmp_path):
    # Section names used by older software versions
    nx, ny = 6, 4
    raw = _raw(nx, ny)
    fn = str(tmp_path / "test.spm")
    _write_di(fn, [_file_list(file_list_name="EC File list"),
                   _scanner_list("Microscope list"),
                   _scan_list(nx, ny, name="Afm list"),
                   _image_list(nx, ny, 2 * nx * ny, name="AFM image list",
                               image_data='S "Height"')],
              [raw.astype("<i2").tobytes()])
    r = DIReader(fn)
    (ch,) = r.channels
    # Without soft-scale name, the channel is named by the quoted value
    assert ch.name == "Height"
    assert ch.unit == "nm"
    np.testing.assert_allclose(r.topography().heights(), _expected(raw, 3.0 / 65536 * 20.0))


def test_z_scale_at_4(tmp_path):
    # '@4:Z scale' takes precedence over '@2:Z scale'
    nx, ny = 6, 4
    raw = _raw(nx, ny)
    fn = str(tmp_path / "test.spm")
    image = _image_list(nx, ny, 2 * nx * ny,
                        extra=[("@4:Z scale", "V [Sens. Zsens] (0.005035400 V/LSB) 6.0 V")])
    _write_di(fn, [_file_list(), _scanner_list(), _scan_list(nx, ny), image],
              [raw.astype("<i2").tobytes()])
    t = DIReader(fn).topography()
    np.testing.assert_allclose(t.heights(), _expected(raw, 6.0 / 65536 * 20.0))


def test_non_height_channel_with_nested_parentheses(tmp_path):
    nx, ny = 6, 4
    raw = _raw(nx, ny)
    fn = str(tmp_path / "test.spm")
    scan_list = _scan_list(nx, ny, extra=[("@Sens. LogStiffnessSens", "V 1.000000 log(Arb)/log(Arb)")])
    image = _image_list(
        nx, ny, 2 * nx * ny, image_data='S [LogStiffness] "Log Stiffness"',
        z_scale="V [Sens. LogStiffnessSens] (0.0004882813 log(Arb)/LSB) 32.0 log(Arb)")
    _write_di(fn, [_file_list(), _scanner_list(), scan_list, image],
              [raw.astype("<i2").tobytes()])
    (ch,) = DIReader(fn).channels
    assert ch.name == "LogStiffness"
    assert ch.height_scale_factor is None
    assert ch.unit == ("nm", "log(Arb)")


@pytest.mark.parametrize("operating_mode,start_context", [
    ("Force", "FOL"),
    ("Image", "FOL"),
    ("Image", "OLVAR"),
])
def test_unsupported_data_types(tmp_path, operating_mode, start_context):
    # Force curves and sets of profiles are not topography maps
    nx, ny = 6, 4
    raw = _raw(nx, ny)
    fn = str(tmp_path / "test.spm")
    file_list = _file_list()
    file_list[1][2] = ("Start context", start_context)
    scan_list = _scan_list(nx, ny)
    scan_list[1][0] = ("Operating mode", operating_mode)
    _write_di(fn, [file_list, _scanner_list(), scan_list,
                   _image_list(nx, ny, 2 * nx * ny)],
              [raw.astype("<i2").tobytes()])
    with pytest.raises(UnsupportedFormatFeature):
        DIReader(fn)


def test_text_data_unsupported(tmp_path):
    nx, ny = 6, 4
    raw = _raw(nx, ny)
    fn = str(tmp_path / "test.spm")
    _write_di(fn, [_file_list(), _scanner_list(), _scan_list(nx, ny),
                   _image_list(nx, ny, 2 * nx * ny)],
              [raw.astype("<i2").tobytes()], first_line="?*File list")
    with pytest.raises(UnsupportedFormatFeature):
        DIReader(fn)
