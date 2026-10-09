#
# Copyright 2020-2023 Lars Pastewka
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
import zipfile

import numpy as np
import pytest
from NuMPI import MPI

from SurfaceTopography import read_topography
from SurfaceTopography.Exceptions import FileFormatMismatch
from SurfaceTopography.IO import OIRReader, POIRReader

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial funcionalities, please execute with pytest")


def test_oir_read_filestream(file_format_examples):
    """
    The reader has to work when the file was already opened as binary for
    it to work in topobank.
    """
    file_path = os.path.join(file_format_examples, 'oir-1.oir')

    read_topography(file_path)

    with open(file_path, 'r') as f:
        read_topography(f)

    # This test just needs to arrive here without raising an exception


def test_poir_read_filestream(file_format_examples):
    """
    The reader has to work when the file was already opened as binary for
    it to work in topobank.
    """
    file_path = os.path.join(file_format_examples, 'poir-1.poir')

    read_topography(file_path)

    with open(file_path, 'r') as f:
        read_topography(f)

    # This test just needs to arrive here without raising an exception


def test_oir_metadata(file_format_examples):
    file_path = os.path.join(file_format_examples, 'oir-1.oir')

    r = OIRReader(file_path)
    t = r.topography()

    nx, ny = t.nb_grid_pts
    assert nx == 1024
    assert ny == 1024

    sx, sy = t.physical_sizes
    np.testing.assert_allclose(sx, 2565.801408, rtol=1e-6)
    np.testing.assert_allclose(sy, 2565.177801, rtol=1e-6)

    assert t.unit == 'µm'
    assert t.info['instrument']['vendor'] == 'Olympus'

    np.testing.assert_allclose(t.rms_height_from_area(), 2.048709, rtol=1e-6)


def test_poir_metadata(file_format_examples):
    file_path = os.path.join(file_format_examples, 'poir-1.poir')

    r = POIRReader(file_path)
    t = r.topography()

    nx, ny = t.nb_grid_pts
    assert nx == 1024
    assert ny == 1024

    sx, sy = t.physical_sizes
    np.testing.assert_allclose(sx, 2565.801408, rtol=1e-6)
    np.testing.assert_allclose(sy, 2565.177801, rtol=1e-6)

    assert t.unit == 'µm'

    np.testing.assert_allclose(t.rms_height_from_area(), 2.048709, rtol=1e-6)


def _patched_oir(file_format_examples, tmp_path, old, new):
    """Copy of the example OIR file with an equal-length XML replacement."""
    assert len(old) == len(new)
    with open(os.path.join(file_format_examples, 'oir-1.oir'), 'rb') as f:
        buffer = f.read()
    assert old in buffer
    file_path = tmp_path / 'patched.oir'
    file_path.write_bytes(buffer.replace(old, new))
    return str(file_path)


def test_oir_z_calibration(file_format_examples, tmp_path):
    # The z pixel calibration (1.0 in the example) scales the heights
    reference = OIRReader(os.path.join(file_format_examples, 'oir-1.oir')).topography()
    file_path = _patched_oir(
        file_format_examples, tmp_path,
        b'<commonparam:z>1.0</commonparam:z>\r\n      </commonimage:pixelCalibration>',
        b'<commonparam:z>2.5</commonparam:z>\r\n      </commonimage:pixelCalibration>')
    t = OIRReader(file_path).topography()
    np.testing.assert_allclose(t.physical_sizes, reference.physical_sizes)
    np.testing.assert_allclose(t.heights(), 2.5 * reference.heights())


def test_oir_significant_bits(file_format_examples, tmp_path):
    # Height data with fewer significant bits than its 16-bit storage size
    reference = OIRReader(os.path.join(file_format_examples, 'oir-1.oir')).topography()
    file_path = _patched_oir(
        file_format_examples, tmp_path,
        b'<commonphase:bitCounts>16</commonphase:bitCounts>\r\n'
        b'            <commonphase:colorType>GlayScale</commonphase:colorType>\r\n'
        b'            <commonphase:imageType>HEIGHT</commonphase:imageType>',
        b'<commonphase:bitCounts>14</commonphase:bitCounts>\r\n'
        b'            <commonphase:colorType>GlayScale</commonphase:colorType>\r\n'
        b'            <commonphase:imageType>HEIGHT</commonphase:imageType>')
    t = OIRReader(file_path).topography()
    np.testing.assert_allclose(t.heights(), reference.heights())


def test_poir_ignores_other_members(file_format_examples, tmp_path):
    # Packed OIR files may contain members that are not OIR files
    file_path = str(tmp_path / 'other_members.poir')
    with zipfile.ZipFile(os.path.join(file_format_examples, 'poir-1.poir')) as source, \
            zipfile.ZipFile(file_path, 'w') as z:
        z.writestr('readme.txt', 'This is not an OIR file.')
        for name in source.namelist():
            z.writestr(name, source.read(name))
    r = POIRReader(file_path)
    assert len(r.channels) == 1
    np.testing.assert_allclose(r.topography().rms_height_from_area(), 2.048709, rtol=1e-6)


def test_poir_without_oir_members(tmp_path):
    file_path = str(tmp_path / 'no_oir.zip')
    with zipfile.ZipFile(file_path, 'w') as z:
        z.writestr('readme.txt', 'This is not an OIR file.')
    with pytest.raises(FileFormatMismatch):
        POIRReader(file_path)
