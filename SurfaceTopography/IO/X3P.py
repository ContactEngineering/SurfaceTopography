#
# Copyright 2022-2023 Lars Pastewka
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
# Reference information and implementations:
# https://sourceforge.net/p/open-gps/mwiki
#

import numpy as np
import xml.etree.ElementTree as ElementTree
from zipfile import ZipFile, BadZipFile

import dateutil.parser

from .common import OpenFromAny
from .Reader import ReaderBase, ChannelInfo
from ..Exceptions import CorruptFile, FileFormatMismatch, MetadataAlreadyFixedByFile
from ..UniformLineScanAndTopography import Topography


class X3PReader(ReaderBase):
    _format = 'x3p'
    _mime_types = ['application/x-iso5436-2-spm']
    _file_extensions = ['x3p']

    _name = 'XML 3D surface profile (X3P)'
    _description = '''
X3P is a container format conforming to the ISO 5436-2 (Geometrical Product
Specifications — Surface texture) standard. The format is defined in ISO
25178 and is a standardized format for the exchange of surface topography
data. The full specification of the format can be found
[here](http://www.opengps.eu/).
'''

    _REVISION = 'ISO5436 - 2000'
    _FEATURE_TYPE = 'SUR'

    # Data types of binary container
    # See: https://sourceforge.net/p/open-gps/mwiki/X3p/
    _DTYPE_MAP = {'I': np.dtype('<u2'),
                  'L': np.dtype('<u4'),
                  'F': np.dtype('f4'),
                  'D': np.dtype('f8')}

    # Reads in the positions of all the data and metadata
    def __init__(self, file_path):
        self.file_path = file_path
        with OpenFromAny(file_path, 'rb') as f:
            try:
                with ZipFile(f, 'r') as x3p:
                    try:
                        main_xml = x3p.open('main.xml')
                    except OSError:
                        # This appears not to be an X3P
                        raise FileFormatMismatch("ZIP file does not have 'main.xml'.")

                    xmlroot = ElementTree.parse(main_xml).getroot()
                    # Information on the measurement (grid points, size, etc.)
                    record1 = xmlroot.find('Record1')
                    # Information on the data file
                    record3 = xmlroot.find('Record3')

                    if record1 is None:
                        raise FileFormatMismatch("'Record1' not found in 'main.xml'.")
                    if record3 is None:
                        raise FileFormatMismatch("'Record3' not found in 'main.xml'.")

                    # Parse record1
                    revision = record1.find('Revision').text
                    if revision != self._REVISION:
                        raise CorruptFile(f"Revision should be '{self._REVISION}' but is '{revision}.")
                    feature_type = record1.find('FeatureType').text
                    if feature_type != self._FEATURE_TYPE:
                        raise CorruptFile(f"FeatureType should be '{self._FEATURE_TYPE}' but is {feature_type}.")
                    axes = record1.find('Axes')
                    cx = axes.find('CX')
                    cy = axes.find('CY')
                    cz = axes.find('CZ')

                    if cx.find('AxisType').text != 'I':
                        raise CorruptFile("CX AxisType is not 'I'. Don't know how to handle this.")
                    if cy.find('AxisType').text != 'I':
                        raise CorruptFile("CY AxisType is not 'I'. Don't know how to handle this.")
                    if cz.find('AxisType').text != 'A':
                        raise CorruptFile("CZ AxisType is not 'A'. Don't know how to handle this.")

                    grid_spacing_x = float(cx.find('Increment').text)
                    grid_spacing_y = float(cy.find('Increment').text)

                    datatype = cz.find('DataType').text
                    self._dtype = self._DTYPE_MAP[datatype]

                    increment = cz.find('Increment')
                    if increment is not None:
                        # We have no proper test for this, as in all files that
                        # we have, the z-increment is either missing or unity.
                        self._height_scale_factor = float(increment.text)
                    else:
                        self._height_scale_factor = None

                    # Parse record3
                    matrix_dimension = record3.find('MatrixDimension')
                    nb_grid_pts_x = int(matrix_dimension.find('SizeX').text)
                    nb_grid_pts_y = int(matrix_dimension.find('SizeY').text)
                    nb_grid_pts_z = int(matrix_dimension.find('SizeZ').text)

                    if nb_grid_pts_z != 1:
                        raise CorruptFile('Z dimension has extend != 1. Volumetric data is not supported.')

                    self._nb_grid_pts = (nb_grid_pts_x, nb_grid_pts_y)
                    self._physical_sizes = (grid_spacing_x * nb_grid_pts_x, grid_spacing_y * nb_grid_pts_y)

                    data_link = record3.find('DataLink')
                    self._name_of_binary_file = data_link.find('PointDataLink').text

                    # Check if binary file exists and has a reasonable size
                    binary_info = x3p.getinfo(self._name_of_binary_file)
                    expected_file_size = np.prod(self._nb_grid_pts) * self._dtype.itemsize
                    if binary_info.file_size < expected_file_size:
                        raise CorruptFile(f'Binary file is too small. It has size if {binary_info.file_size} bytes, '
                                          f'but I expected a size of {expected_file_size} bytes.')

                    # Unit is always meters
                    self._unit = 'm'
                    self._info = {}

                    # Metadata; if this record is missing, we just don't extract metadata
                    record2 = xmlroot.find('Record2')
                    if record2 is not None:
                        raw_metadata = {}

                        date = record2.find('Date')
                        if date is not None:
                            raw_metadata['Date'] = date.text
                            self._info['acquisition_time'] = str(dateutil.parser.parse(date.text))

                        calibration_date = record2.find('CalibrationDate')
                        if calibration_date is not None:
                            raw_metadata['CalibrationDate'] = calibration_date.text

                        instrument = record2.find('Instrument')
                        if instrument is not None:
                            instrument_metadata = {child.tag: child.text for child in instrument}
                            raw_metadata['Instrument'] = instrument_metadata
                            model = manufacturer = version = None
                            if 'Model' in instrument_metadata:
                                model = instrument_metadata['Model']
                            if 'Manufacturer' in instrument_metadata:
                                manufacturer = instrument_metadata['Manufacturer']
                            if 'Version' in instrument_metadata:
                                version = instrument_metadata['Version']

                            instrument_info = {}
                            if model == 'unknown':
                                if manufacturer != 'unknown':
                                    if version != 'unknown':
                                        instrument_info['name'] = f'{manufacturer} (version {version})'
                                    else:
                                        instrument_info['name'] = manufacturer
                            else:
                                if manufacturer != 'unknown':
                                    if version != 'unknown':
                                        instrument_info['name'] = f'{model} ({manufacturer}, version {version})'
                                    else:
                                        instrument_info['name'] = f'{model} ({manufacturer})'
                                else:
                                    if version != 'unknown':
                                        instrument_info['name'] = f'{model} (version {version})'
                                    else:
                                        instrument_info['name'] = model
                            if instrument_info != {}:
                                self._info['instrument'] = instrument_info

            except BadZipFile:
                # This is not an X3P
                raise FileFormatMismatch('This is not a ZIP file.')

    @property
    def channels(self):
        return [ChannelInfo(self,
                            0,  # channel index
                            name='Default',
                            dim=2,
                            nb_grid_pts=self._nb_grid_pts,
                            physical_sizes=self._physical_sizes,
                            height_scale_factor=self._height_scale_factor,
                            uniform=True,
                            periodic=False,
                            unit=self._unit,
                            info=self._info)]

    def topography(self, channel_index=None, physical_sizes=None,
                   height_scale_factor=None, unit=None, info={},
                   periodic=None, subdomain_locations=None,
                   nb_subdomain_grid_pts=None):
        if subdomain_locations is not None or \
                nb_subdomain_grid_pts is not None:
            raise RuntimeError('This reader does not support MPI parallelization.')

        if channel_index is None:
            channel_index = self._default_channel_index

        if channel_index != self._default_channel_index:
            raise RuntimeError(f'There is only a single channel. Channel index must be {self._default_channel_index}.')

        if physical_sizes is not None:
            raise MetadataAlreadyFixedByFile('physical_sizes')

        if height_scale_factor is not None and self._height_scale_factor is not None:
            raise MetadataAlreadyFixedByFile('height_scale_factor')

        if unit is not None:
            raise MetadataAlreadyFixedByFile('unit')

        _info = self._info.copy()
        _info.update(info)

        with OpenFromAny(self.file_path, 'rb') as f:
            with ZipFile(f, 'r') as x3p:
                nx, ny = self._nb_grid_pts
                rawdata = x3p.open(self._name_of_binary_file).read(np.prod(self._nb_grid_pts) * self._dtype.itemsize)
                height_data = np.frombuffer(rawdata, count=np.prod(self._nb_grid_pts), dtype=self._dtype) \
                    .reshape(ny, nx).T

        topo = Topography(
            height_data,
            self._physical_sizes,
            unit=self._unit,
            info=_info,
            periodic=False if periodic is None else periodic)
        if self._height_scale_factor is not None:
            return topo.scale(self._height_scale_factor)
        elif height_scale_factor is not None:
            return topo.scale(height_scale_factor)
        else:
            return topo
