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

import datetime
import os
import struct

import numpy as np
import pytest

from NuMPI import MPI

from SurfaceTopography import read_topography
from SurfaceTopography.IO import PLUReader

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial funcionalities, please execute with pytest")


def test_read_filestream(file_format_examples):
    """
    The reader has to work when the file was already opened as binary for
    it to work in topobank.
    """
    file_path = os.path.join(file_format_examples, 'plu-1.plu')

    read_topography(file_path)

    with open(file_path, 'r') as f:
        read_topography(f)

    # This test just needs to arrive here without raising an exception


def test_plu_metadata(file_format_examples):
    file_path = os.path.join(file_format_examples, 'plu-1.plu')

    r = PLUReader(file_path)
    t = r.topography()

    nx, ny = t.nb_grid_pts
    assert nx == 768
    assert ny == 576

    sx, sy = t.physical_sizes
    np.testing.assert_allclose(sx, 1274.880066, rtol=1e-6)
    np.testing.assert_allclose(sy, 956.160049, rtol=1e-6)

    assert t.unit == 'µm'

    np.testing.assert_allclose(t.rms_height_from_area(), 2.834391, rtol=1e-6)
    np.testing.assert_allclose(t.rms_height_from_profile(), 2.833895, rtol=1e-6)

    t = t.detrend('curvature')
    np.testing.assert_allclose(t.rms_height_from_area(), 0.01818, rtol=1e-4)
    np.testing.assert_allclose(t.rms_height_from_profile(), 0.018167, rtol=1e-4)


def _synthetic_plu(layers, objective=0, version=0xFB, hardware=9, num_images=0, mpp=(0.5, 0.25)):
    """
    Build a minimal PLU file holding the given topography layers (arrays
    of shape (ny, nx), heights in micrometers).
    """
    ny, nx = layers[0].shape
    header = struct.pack('<128sI256s', b'Fri Mar 10 16:35:59 2023', 1678462559, b'synthetic')
    calibration = struct.pack('<3I7f', ny, nx, nx, 1.0, mpp[0], mpp[1], 0.0, 0.0, 1.0, 0.0)
    # Topography measurement, confocal intensity algorithm, field-of-view area
    configuration1 = struct.pack('<5I', 3, 0, 0, objective, 3)
    scan_settings = struct.pack('<5IdfII', nx, ny, nx, ny, 1, 0.1, 10.0, 100, 0)
    configuration2 = struct.pack('<8BI', 0, len(layers), version, hardware, num_images, 0, 0, 0, 1)
    buffer = header + calibration + configuration1 + scan_settings + configuration2
    for i, layer in enumerate(layers):
        buffer += struct.pack('<2I', ny, nx)
        buffer += np.asarray(layer, dtype='<f4').tobytes()
        buffer += struct.pack('<2f', np.min(layer), np.max(layer))
        # RGB images following the layer; fill with a value that would
        # produce garbage heights if interpreted as part of the next layer
        buffer += bytes([0x55 + i]) * (3 * nx * ny * num_images)
    return buffer


def test_plu_metadata_info(file_format_examples):
    r = PLUReader(os.path.join(file_format_examples, 'plu-1.plu'))
    info = r.channels[0].info
    assert info['acquisition_time'] == datetime.datetime(2023, 3, 10, 16, 35, 59)
    assert info['instrument']['vendor'] == 'Sensofar'
    # Format version 2012, hardware configuration 6
    assert info['instrument']['name'] == 'PLu neox'
    assert info['raw_metadata']['measurement_configuration']['objective'] == 'Nikon CFI Plan Interferential 10X'


@pytest.mark.parametrize('objective,name', [
    (0, 'Unknown'),
    (64, 'Leica Interferential Mirau SR 100X'),
    # Objective IDs 65 to 71 are not assigned, numbering continues at 72
    (72, 'Leica HC PL Fluotar EPI 50X 0.8'),
    (90, 'Nikon CFI TU Plan Fluor EPI 50X'),
    (92, 'Nikon CFI TU Plan Apo EPI 150X'),
    (68, None),
    (1000, None),
])
def test_plu_objective_ids(tmp_path, objective, name):
    layer = np.arange(12, dtype=float).reshape(3, 4)
    file_path = tmp_path / 'objective.plu'
    file_path.write_bytes(_synthetic_plu([layer], objective=objective))
    r = PLUReader(str(file_path))
    assert r.metadata['measurement_configuration1']['objective'] == name


@pytest.mark.parametrize('version,hardware,name', [
    (0xFB, 9, 'S neox'),  # 2012
    (0xFA, 10, 'DCM8'),  # 2013
    # The hardware configuration is not reliable in older files
    (0xFC, 9, None),  # 2011B
    (0x00, 9, None),  # 2000
])
def test_plu_hardware_configuration(tmp_path, version, hardware, name):
    layer = np.arange(12, dtype=float).reshape(3, 4)
    file_path = tmp_path / 'hardware.plu'
    file_path.write_bytes(_synthetic_plu([layer], version=version, hardware=hardware))
    instrument = PLUReader(str(file_path)).channels[0].info['instrument']
    assert instrument['vendor'] == 'Sensofar'
    assert instrument.get('name') == name


def test_plu_multiple_layers_with_rgb_images(tmp_path):
    # Each topography layer is followed by `num_images` RGB images
    ny, nx = 3, 4
    layer0 = np.arange(nx * ny, dtype=float).reshape(ny, nx)
    layer1 = -2 * layer0
    layer1[1, 2] = 1000001  # Undefined data marker
    file_path = tmp_path / 'layers.plu'
    file_path.write_bytes(_synthetic_plu([layer0, layer1], num_images=2))

    r = PLUReader(str(file_path))
    assert len(r.channels) == 2
    for c in r.channels:
        assert c.nb_grid_pts == (nx, ny)
        np.testing.assert_allclose(c.physical_sizes, (nx * 0.5, ny * 0.25))

    t0 = r.topography(channel_index=0)
    np.testing.assert_allclose(t0.heights(), layer0.T)
    assert not t0.has_undefined_data

    t1 = r.topography(channel_index=1)
    assert t1.has_undefined_data
    heights = t1.heights()
    assert heights.mask[2, 1]
    assert np.ma.count_masked(heights) == 1
    np.testing.assert_allclose(heights[~heights.mask], layer1.T[~heights.mask])
