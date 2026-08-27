#
# Copyright 2023-2026 Lars Pastewka
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

from ..Exceptions import FileFormatMismatch
from .binary import BinaryArray, XMLStructure, ZipContainer, ZipMemberLoop
from .expr import C, F, Tup, V
from .Reader import DeclarativeReaderBase

# The measurement layers are described by entries `LAYER_0`, `LAYER_1`,
# ... of the XML index; the name of each layer's height-data member is
# given by its `FILENAME_Z` entry
_layers = F.values_with_prefix(C.index, "LAYER_")

# The `INFO` section is a list of NAME/VALUE items; the `Device` item
# holds the instrument name
_device = F.get(
    F.to_map(F.values_with_prefix(C.index.INFO, "ITEM_"), "NAME", "VALUE"),
    "Device",
    None,
)


class PLUXReader(DeclarativeReaderBase):
    _format = "plux"
    _mime_types = ["application/x-sensofarx-spm"]
    _file_extensions = ["plux"]

    _name = "Sensofar PLUX"
    _description = """
PLUX files of Sensofar 3D optical profilometers: ZIP containers holding XML
metadata and binary data layers.
"""

    # PLUX files are ZIP archives without a distinctive magic; detection
    # needs to try parsing
    _magic = None

    _file_layout = ZipContainer(
        [
            ("index.xml", XMLStructure(name="index")),
            ("recipe.txt", XMLStructure(name="recipe")),
            ZipMemberLoop(
                _layers,
                C.item.FILENAME_Z,
                BinaryArray(
                    "data",
                    Tup(
                        F.int(C.index.GENERAL.IMAGE_SIZE_Y),
                        F.int(C.index.GENERAL.IMAGE_SIZE_X),
                    ),
                    F.dtype("<f4"),
                    conversion_fun=F.transpose(V),  # Transpose to (nx, ny)
                ),
                name="layers",
            ),
        ],
        # `index.xml` and `recipe.txt` identify the format
        mismatch_error=FileFormatMismatch,
    )

    _channel_bindings = [
        {
            # One channel per measurement layer
            "foreach": _layers,
            "name": "Layer " + F.str(C.item_index),
            "dim": 2,
            "nb_grid_pts": Tup(
                F.int(C.index.GENERAL.IMAGE_SIZE_X),
                F.int(C.index.GENERAL.IMAGE_SIZE_Y),
            ),
            "physical_sizes": Tup(
                F.float(C.index.GENERAL.FOV_X)
                * F.int(C.index.GENERAL.IMAGE_SIZE_X),
                F.float(C.index.GENERAL.FOV_Y)
                * F.int(C.index.GENERAL.IMAGE_SIZE_Y),
            ),
            "height_scale_factor": 1,
            "uniform": True,
            "periodic": False,
            "unit": "µm",  # Everything seems to be in micrometers
            "info": {
                "acquisition_time": F.parse_datetime(C.index.GENERAL.DATE),
                "instrument": {
                    "name": _device,
                    "vendor": "Sensofar",
                },
                "raw_metadata": {
                    "index": C.index,
                    "recipe": C.recipe,
                },
            },
            "data": C.layers[C.item_index].data,
            # Undefined points are stored as NaN
            "mask": {
                "source": C.layers[C.item_index].data,
                "rule": F.isnan(V),
            },
        }
    ]
