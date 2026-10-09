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
from SurfaceTopography.IO import PLUXReader

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial funcionalities, please execute with pytest")


def test_read_filestream(file_format_examples):
    """
    The reader has to work when the file was already opened as binary for
    it to work in topobank.
    """
    file_path = os.path.join(file_format_examples, 'plux-1.plux')

    read_topography(file_path)

    with open(file_path, 'r') as f:
        read_topography(f)

    # This test just needs to arrive here without raising an exception


def test_plux_metadata(file_format_examples):
    file_path = os.path.join(file_format_examples, 'plux-1.plux')

    r = PLUXReader(file_path)
    t = r.topography()

    assert t.has_undefined_data

    nx, ny = t.nb_grid_pts
    assert nx == 2583
    assert ny == 1023

    sx, sy = t.physical_sizes
    np.testing.assert_allclose(sx, 1666.034951, rtol=1e-6)
    np.testing.assert_allclose(sy, 659.83498, rtol=1e-6)

    assert t.unit == 'µm'
    assert t.info['instrument']['vendor'] == 'Sensofar'
    assert t.info['instrument']['name'] == 'S neox'

    np.testing.assert_allclose(t.rms_height_from_area(), 2.162500740959119, rtol=1e-6)
    np.testing.assert_allclose(t.rms_height_from_profile(), 1.589988838291435, rtol=1e-6)

    t = t.detrend('curvature')
    np.testing.assert_allclose(t.rms_height_from_area(), 1.4420593851629364, rtol=1e-4)
    np.testing.assert_allclose(t.rms_height_from_profile(), 1.2085860010079734, rtol=1e-4)


_INDEX_XML = """<?xml version="1.0" encoding="utf-8"?>
<xml>
\t<GENERAL>
\t\t<AUTHOR>Someone</AUTHOR>
\t\t<DATE>2024-01-02 03:04:05</DATE>
\t\t<FOV_X>0.5</FOV_X>
\t\t<FOV_Y>0.25</FOV_Y>
\t\t<IMAGE_SIZE_X>4</IMAGE_SIZE_X>
\t\t<IMAGE_SIZE_Y>3</IMAGE_SIZE_Y>
\t</GENERAL>
\t<INFO>
\t\t<SIZE>1</SIZE>
\t\t<ITEM_0>
\t\t\t<NAME>Device</NAME>
\t\t\t<VALUE>S lynx</VALUE>
\t\t</ITEM_0>
\t</INFO>
\t<LAYER_0>
\t\t<FILENAME_Z>LAYER_0.raw</FILENAME_Z>
\t</LAYER_0>
</xml>
"""

_RECIPE_XML = """<?xml version="1.0" encoding="utf-8"?>
<xml><MEASUREMENT_CONFIG><TYPE>3</TYPE></MEASUREMENT_CONFIG></xml>
"""


def _write_plux(file_path, recipe_name, index_xml=_INDEX_XML):
    heights = np.arange(12, dtype='<f4').reshape(3, 4)
    heights[2, 1] = np.nan
    with zipfile.ZipFile(file_path, 'w') as z:
        z.writestr('LAYER_0.raw', heights.tobytes())
        z.writestr('index.xml', index_xml)
        if recipe_name is not None:
            z.writestr(recipe_name, _RECIPE_XML)
    return heights


@pytest.mark.parametrize('recipe_name', ['recipe.txt', './recipe.txt', None])
def test_plux_recipe_location(tmp_path, recipe_name):
    # The recipe is optional and may be stored as `./recipe.txt`
    file_path = str(tmp_path / 'synthetic.plux')
    heights = _write_plux(file_path, recipe_name)

    r = PLUXReader(file_path)
    assert len(r.channels) == 1
    c = r.channels[0]
    assert c.nb_grid_pts == (4, 3)
    np.testing.assert_allclose(c.physical_sizes, (2.0, 0.75))
    assert c.info['instrument']['name'] == 'S lynx'
    if recipe_name is None:
        assert c.info['raw_metadata']['recipe'] is None
    else:
        assert c.info['raw_metadata']['recipe']['MEASUREMENT_CONFIG']['TYPE'] == '3'

    t = r.topography()
    assert t.has_undefined_data
    h = t.heights()
    assert h.mask[1, 2]
    assert np.ma.count_masked(h) == 1
    np.testing.assert_allclose(h[~h.mask], heights.T[~h.mask])


def test_plux_index_without_image_size(tmp_path):
    # An `index.xml` without image size is not a PLUX index
    file_path = str(tmp_path / 'other.zip')
    _write_plux(file_path, 'recipe.txt', index_xml='<?xml version="1.0"?><xml><Version>1</Version></xml>')
    with pytest.raises(FileFormatMismatch):
        PLUXReader(file_path)
