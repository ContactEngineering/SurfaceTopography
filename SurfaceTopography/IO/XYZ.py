#
# Copyright 2023 Lars Pastewka
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

import numpy as np
import pandas as pd

from ..Exceptions import MetadataAlreadyFixedByFile, UnknownFileFormat
from ..NonuniformLineScan import NonuniformLineScan
from ..UniformLineScanAndTopography import Topography, UniformLineScan
from ..Support.UnitConversion import mangle_length_unit_utf8, get_unit_conversion_factor, find_length_unit_in_string
from .common import OpenFromAny
from .Reader import ReaderBase, ChannelInfo

_log = logging.Logger(__name__)


def read_text_header_hfm(fobj, unit, height_scale_factor):
    """HFM files"""
    # HFM files have the following header
    # X;Y;valid
    # [mm];[mm];[1/0]
    first_line = fobj.readline()
    assert first_line.strip() == 'X;Y;valid'  # This is the file magic
    second_line = fobj.readline()
    xunit, zunit, _ = second_line.split(';')
    xunit = xunit.strip('[').strip(']')
    zunit = zunit.strip('[').strip(']')
    if unit is not None:
        raise MetadataAlreadyFixedByFile('unit')
    unit = xunit
    if height_scale_factor is not None:
        raise MetadataAlreadyFixedByFile('height_scale_factor')
    height_scale_factor = get_unit_conversion_factor(zunit, xunit)
    sep = ';'  # This file seems to use semicolons as separators
    usecols = (0, 1)

    return sep, usecols, unit, height_scale_factor, {}


def read_text_header_dektak(fobj, unit, height_scale_factor):
    """Dektak CSV files"""
    sep = ','  # This file seems to use coma as separators
    usecols = (0, 1)

    # We just parse everything before scan data in key-value pairs
    raw_metadata = {}
    line = fobj.readline()
    while line and not line.startswith('Scan Data'):
        s = line.strip().split(sep, maxsplit=1)
        if len(s) == 2:
            # Skip if this is not a key-value pair
            key, value = s
            raw_metadata[key] = value
        line = fobj.readline()

    # Next line should be empty
    line = fobj.readline()
    assert len(line.strip()) == 0

    # Next line contains a header, example: "Lateral um,Raw Micrometer,"
    line = fobj.readline()
    header = line.split(sep)
    xcol, ycol = usecols
    xheader = header[xcol]
    yheader = header[ycol]

    # Find units and conversion factors
    xunit = find_length_unit_in_string(xheader)
    yunit = find_length_unit_in_string(yheader)

    if xunit is not None:
        if unit is not None:
            raise MetadataAlreadyFixedByFile('unit')
        else:
            unit = xunit
    if yunit is not None:
        if height_scale_factor is not None:
            raise MetadataAlreadyFixedByFile('height_scale_factor')
        else:
            height_scale_factor = get_unit_conversion_factor(yunit, unit)

    # Prepare info dictionary
    info = {'raw_metadata': raw_metadata}

    # Some interpretation of metadata, if present
    try:
        stylus = raw_metadata['Stylus']
        if stylus.startswith('Radius:'):
            tip_radius_value, tip_radius_unit = stylus[7:].strip().split(' ')
            try:
                info['instrument'] = {
                    'parameters': {
                        'tip_radius': {
                            'value': float(tip_radius_value),
                            'unit': mangle_length_unit_utf8(tip_radius_unit),
                        }
                    }
                }
            except ValueError:
                # Cannot convert tip radius to floating point number
                pass
    except KeyError:
        # 'Stylus' key does not exist
        pass

    return sep, usecols, unit, height_scale_factor, info


# Dictionary of magic: parser pairs, where magic is first line of file
_text_header_parsers = {
    b'X;Y;valid': ('utf-8', read_text_header_hfm),
    b'Scan Parameters': ('latin-1', read_text_header_dektak)
}


class XYZReader(ReaderBase):
    _format = 'xyz'
    _mime_types = ['text/plain']
    _file_extensions = ['xyz', 'hfm', 'csv']

    _name = 'Plain text (x,y,z coordinates)'
    _description = '''
Surface topography information can be provided as coordinate data. This is a
text file that contains either two columns (for line scans) or three columns
(for two-dimensional topographies) of data.

Line scans can be provided on a non-uniform grid. The x-coordinates do not
need to be equally spaced and the surface specified in this format can be
reentrant. The code interprets such topographies as piecewise linear between
the points that are specified in the file.

Two-dimensional topography maps need to reside on a regular grid. The x- and
y-coordinates need to be equally spaced.

The reader supports parsing HFM and Dektak header information.
'''

    def __init__(self, file_path):
        self._file_path = file_path
        self._default_channel_index = 0

        # We read this thing once to get the metadata
        topography = self._read()

        self._channels = [ChannelInfo(
            self,
            0,  # channel index
            name="default",
            dim=topography.dim,
            nb_grid_pts=topography.nb_grid_pts,
            physical_sizes=topography.physical_sizes,
            height_scale_factor=topography.height_scale_factor if hasattr(topography, 'height_scale_factor') else None,
            uniform=topography.is_uniform,
            info=topography.info,
            unit=topography.unit
        )]

    def _read(self, physical_sizes=None, height_scale_factor=None, unit=None, info={}, periodic=None,
              max_header_rows=200, tol=1e-6):
        """
        Read the xyz-file. These files contain line scan information in terms of
        (x,y)-positions.

        Parameters
        ----------
        file : str or file object
             File name or stream.
        max_header_rows : int, optional
             Maximum number of rows to skip when testing for headers.
             (Default: 100)
        tol : float, optional
             Tolerance for detecting uniform grids. (Default: 1e-6)

        Returns
        -------
        topography : Topography or UniformLineScan or NonuniformLineScan
            SurfaceTopography object.
        """
        # Default is to autodetect separator and columns
        sep = r'[\s,;]+'  # white space, comma or semicolon
        usecols = None

        # Determine maximum length of magic
        maxmagic = 0
        for key in _text_header_parsers.keys():
            maxmagic = max(maxmagic, len(key))

        # Read header (if present) and guess file format
        encoding = 'utf-8'  # Open with UTF-8 encoding by default
        header_parser = None  # We don't parse any headers by default
        with OpenFromAny(self._file_path, mode='rb') as fobj:
            first_line = fobj.read(maxmagic)
            for magic, parser in _text_header_parsers.items():
                if first_line.startswith(magic):
                    encoding, header_parser = parser
                    break  # Don't check other file magics since we found one
        # Stream will automatically rewind to old position here

        # Reopen, but with correct encoding
        with OpenFromAny(self._file_path, mode='r', encoding=encoding) as fobj:
            if header_parser is not None:
                # Read header
                sep, usecols, unit, height_scale_factor, _info = header_parser(fobj, unit, height_scale_factor)
            else:
                _info = {}

            # Update info with user-specified dictionary
            _info.update(info)

            # Reading data, skipping first rows (which may contain headers)
            data_start = fobj.tell()
            skiprows = 0
            data = None
            while data is None and skiprows < max_header_rows:
                fobj.seek(data_start)
                try:
                    data = pd.read_csv(fobj, sep=sep, usecols=usecols, header=None, skiprows=skiprows, engine='python')
                    if (data.dtypes == 'O').any():
                        data = None
                except pd.errors.ParserError:
                    data = None
                skiprows += 1

            _log.debug(f'Skipping {skiprows} rows before attempting to read data from XYZ file.')

            # We cannot make sense of this data
            if data is None:
                raise UnknownFileFormat('Could not parse file as XYZ.')

            # Remove columns that contain no data
            drop_cols = []
            for col in data.columns:
                arr = data.values[:, col]
                if not np.isfinite(arr).any():
                    drop_cols += [col]
            data = data.drop(columns=drop_cols)

        # Check is this is a line scan in XY format or two-dimensional data in XYZ format
        if len(data.columns) == 2:
            # This is a line scan.
            x, z = np.array(data, dtype=float).T
            x -= np.min(x)

            d_uniform = (x[-1] - x[0]) / (len(x) - 1)
            if np.max(np.abs(np.diff(x) - d_uniform)) < tol:
                if physical_sizes is None:
                    physical_sizes = d_uniform * len(x)
                else:
                    raise MetadataAlreadyFixedByFile('physical_sizes')
                t = UniformLineScan(z, physical_sizes,
                                    periodic=False if periodic is None else periodic,
                                    unit=unit,
                                    info=_info)
            else:
                if periodic is not None and periodic:
                    raise ValueError('XYZ reader found nonuniform data, and the user specified that it is periodic. '
                                     'Nonuniform line scans cannot be periodic.')
                t = NonuniformLineScan(x, z, unit=unit, info=_info)
                if physical_sizes is not None:
                    raise MetadataAlreadyFixedByFile('physical_sizes')

        elif len(data.columns) == 3:
            # This is a topography map.
            x, y, z = np.array(data, dtype=float).T

            # Compute grid spacing in y-direction
            indices = np.lexsort((y, x))
            y0 = y[indices[0]]
            dy = y[indices[1]] - y0

            # Sort values, first x than y
            indices = np.lexsort((x, y))
            x = x[indices]
            y = y[indices]
            z = z[indices]

            # Compute grid spacing in x-direction
            x0 = x[0]
            dx = x[1] - x0

            # Sort x-values into bins. Assume points on surface are equally spaced.
            binx = np.array((x - x0) / dx + 0.5, dtype=int)
            n = np.bincount(binx)
            ny = n[0]
            assert np.all(n == ny)  # FIXME: Turn assert into exception

            # Sort y-values into bins.
            biny = np.array((y - y0) / dy + 0.5, dtype=int)
            n = np.bincount(biny)
            nx = n[0]
            assert np.all(n == nx)  # FIXME: Turn assert into exception

            # Sort data into bins.
            data = np.zeros((nx, ny))
            data[binx, biny] = z

            # Sanity check. Should be covered by above asserts.
            value_present = np.zeros((nx, ny), dtype=bool)
            value_present[binx, biny] = True
            assert np.all(value_present)  # FIXME: Turn assert into exception

            if physical_sizes is None:
                physical_sizes = (dx * nx, dy * ny)
            else:
                raise MetadataAlreadyFixedByFile('physical_sizes')
            t = Topography(data, physical_sizes, unit=unit, info=_info, periodic=periodic)
        else:
            raise UnknownFileFormat(
                'Expected two or three columns for topography that is a list of positions and heights.')

        if height_scale_factor is not None:
            t = t.scale(height_scale_factor)
        return t

    @property
    def channels(self):
        """
        Returns a list of :obj:`ChannelInfo`s describing the available data
        channels.
        """
        return self._channels

    def topography(self, channel_index=None, physical_sizes=None, height_scale_factor=None, unit=None, info={},
                   periodic=None, subdomain_locations=None, nb_subdomain_grid_pts=None):
        if subdomain_locations is not None or nb_subdomain_grid_pts is not None:
            raise RuntimeError("This reader does not support MPI parallelization.")

        if channel_index is None:
            channel_index = self._default_channel_index

        if channel_index != self._default_channel_index:
            raise RuntimeError(f"There is only a single channel. Channel index must be {self._default_channel_index}.")

        return self._read(physical_sizes=physical_sizes, height_scale_factor=height_scale_factor, unit=unit, info=info,
                          periodic=False if periodic is None else periodic)
