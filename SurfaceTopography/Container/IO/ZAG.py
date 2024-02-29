#
# Copyright 2020, 2023 Lars Pastewka
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

#
# Copyright 2021 Lars Pastewka
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

import logging
import os
from zipfile import ZipFile

import defusedxml.ElementTree as ElementTree

from ...Exceptions import CorruptFile, FileFormatMismatch
from ...IO import ZONReader
from ...IO.binary import decode
from ...IO.common import OpenFromAny

from ..SurfaceContainer import LazySurfaceContainer

from .Reader import ContainerReaderBase

_log = logging.Logger(__name__)


class ZAGReader(ContainerReaderBase):
    _format = "zag"
    _mime_types = ["application/zip"]
    _file_extensions = ["zag"]

    _name = "Keyence ZAG"
    _description = """
    This reader imports Keyence containers stored in the ZAG format.
    """

    _MAGIC = "KPK0"

    # The files within ZON (zip) files are named using UUIDs. Some of these
    # UUIDs are fixed and contain the same information in each of these files.

    # This file contains the inventory
    _INVENTORY_UUID = "f1724dc6-686c-4502-9227-2a594bc8ed33"

    # This file contains the actual topography data
    _ZON_UUID = "84b648d7-e44f-4909-ac11-0476720a67ff"

    # XML tags
    _INVENTORY_TAG = "DeserializeDataMap"  # This is a toplevel tag corresponding to _INVENTORY_UUID. Is it unqiue?
    _ITEM_TAG = "Item"
    _PATH_TAG = "Path"
    _MEASUREMENT_TAG = (
        "MeasurementDataMap"  # This is a toplevel tag. Is the file UUID unique?
    )
    _DATA_TAG = "MeasurementData"

    _header_structure = [("magic", "4s"), ("bmp_size", "L")]

    def __init__(self, fobj):
        """
        Read all surfaces in a contact.engineering container file and associated
        metadata. The container is a ZIP file with raw data files and a YAML file
        meta.yml that contains all metadata not contained in the raw data files.

        Parameters
        ----------
        fn : str or stream
            File or stream that contains the ZIP-container.

        Returns
        -------
        surface_containers : list of :obj:`SurfaceContainer`s
            List of all surfaces contained in this container file.
        """
        # Open if a file name is given
        if not hasattr(fobj, "read"):
            # This is a string
            self._fstream = open(fobj, "rb")
            self._do_close = True
        else:
            self._fstream = fobj
            self._do_close = False

        self._containers = []

        with OpenFromAny(self._fstream, "rb") as f:
            # There is a header with a file magic and size information
            header = decode(f, self._header_structure, "<")
            if header["magic"] != self._MAGIC:
                raise FileFormatMismatch("This is not a Keyence ZAG file.")

            # The beginning of the file contains a BMP thumbnail, we skip it
            f.seek(header["bmp_size"], os.SEEK_CUR)

            readers = []
            with ZipFile(f, "r") as z:
                # Parse inventory
                root = ElementTree.parse(z.open(self._INVENTORY_UUID)).getroot()
                if root.tag != self._INVENTORY_TAG:
                    raise CorruptFile(
                        f"Found {root.tag} for toplevel inventory XML item, but expected "
                        f"{self._INVENTORY_TAG}."
                    )

                # Get all items
                for item in root.findall(self._ITEM_TAG):
                    item_path = item.find(self._PATH_TAG).text
                    data_uuid = os.path.split(item_path)[0]

                    # Parse per item inventory
                    item_root = ElementTree.parse(z.open(item_path)).getroot()
                    if item_root.tag == self._MEASUREMENT_TAG:
                        for data in item_root.findall(self._DATA_TAG):
                            # Construct reader - we currently assume that all are ZON files
                            data_path = data.find(self._PATH_TAG).text
                            readers += [
                                ZONReader(
                                    z.open(
                                        f"{data_uuid}/{data_path}/{self._ZON_UUID}", "r"
                                    )
                                ).topography
                            ]
                    else:
                        _log.info(f"ZAG reader: Ignoring tag {item_root.tag}")

        self._containers = [LazySurfaceContainer(readers)]

    def __del__(self):
        self.close()

    def close(self):
        if self._do_close:
            self._fstream.close()

    def container(self, index=0):
        """
        Returns an instance of a subclass of :obj:`SurfaceContainer` that
        contains a list of topographies.

        Arguments
        ---------
        index : int
            Index of the container to load.
            (Default: 0, which loads the first container)

        Returns
        -------
        surface_container : subclass of :obj:`SurfaceContainer`
            The object containing a list with actual topography data.
        """
        return self._containers[index]
