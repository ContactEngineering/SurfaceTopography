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
import re
from collections import defaultdict

import numpy as np

from ..Exceptions import MetadataAlreadyFixedByFile, UnknownFileFormat
from ..NonuniformLineScan import NonuniformLineScan
from ..Support.UnitConversion import (find_length_unit_in_string,
                                      get_unit_conversion_factor,
                                      mangle_length_unit_utf8)
from ..UniformLineScanAndTopography import Topography, UniformLineScan
from .common import OpenFromAny
from .Reader import ChannelInfo, ReaderBase

_log = logging.Logger(__name__)


def read_text_header_hfm(fobj, unit, height_scale_factor):
    """
    Read header HFM files.

    Parameters
    ----------
    fobj : file object
        File object to read from.
    unit : str
        Length unit of the data. Raises an error if not `None`because HFM
        files always give a unit.
    height_scale_factor : float
        Conversion factor for the height data. Raises an error if not `None`
        because HFM files always define a height scale.

    Returns
    -------
    sep : str
        Separator between columns.
    usecols : list of int
        List of column indices to read.
    skiprows : int
        Number of rows to skip before reading data.
    unit : str
        Length unit of the data.
    height_scale_factor : float
        Conversion factor for the height data.
    info : dict
        Additional information.
    """
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

    return sep, usecols, 0, unit, height_scale_factor, {}


def read_text_header_dektak(fobj, unit, height_scale_factor):
    """
    Read header of Dektak CSV files.

    Parameters
    ----------
    fobj : file object
        File object to read from.
    unit : str
        Length unit of the data, only if not present in the file.
    height_scale_factor : float
        Conversion factor for the height data, only if not present in the
        file.

    Returns
    -------
    sep : str
        Separator between columns.
    usecols : list of int
        List of column indices to read.
    skiprows : int
        Number of rows to skip before reading data.
    unit : str
        Length unit of the data.
    height_scale_factor : float
        Conversion factor for the height data.
    info : dict
        Additional information.
    """
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

    return sep, usecols, 0, unit, height_scale_factor, info


def read_csv(fobj, sep=None, usecols=None, skiprows=0):
    """
    Simple reader for tabular data in C(omma) S(eparated) V(alue) format. The
    comma should not be taken literal as the reader tries to be fuzzy (and
    flexible).

    Parameters
    ----------
    fobj : file object
        File object to read from.
    sep : str, optional
        Separator between columns. (Default: White space, comma or semicolon)
    usecols : list of int, optional
        List of column indices to read. (Default: All columns)
    skiprows : int, optional
        Number of rows to skip before reading data. (Default: 0)

    Returns
    -------
    data : list of np.ndarray
        List of arrays, one for each column in the file.
    """
    if sep is None:
        sep = r'[\s,;]+'  # white space, comma or semicolon
    for i in range(skiprows):
        fobj.readline()
    line = fobj.readline()
    data = defaultdict(list)
    min_cols = None
    nb_cols = None
    if usecols is not None:
        min_cols = max(usecols)  # We need at least this many columns
    nb_lines = 0
    while line:
        line = line.strip()
        if len(line) > 0:  # We ignore empty lines
            line = re.split(sep, line)
            # Idiot check columns
            if nb_cols is None:
                nb_cols = len(line)
            if min_cols is None:
                min_cols = nb_cols
            elif nb_cols < min_cols:
                raise ValueError(f'Too few columns in line {nb_lines}: expected {min_cols} but found {nb_cols}.')
            if len(line) != nb_cols:
                raise ValueError(f'Number of columns changed during parse in line {nb_lines}: expected {nb_cols} but '
                                 f'found {len(line)} (with data {line}).')

            if usecols is None:
                # If columns are given by the user, only collect the data from those colums
                for key, value in enumerate(line):
                    data[key] += [value]
            else:
                # If no columns are given, we return all columns
                for i, key in enumerate(usecols):
                    data[key] += [line[i]]
        line = fobj.readline()
        nb_lines += 1
    if usecols is None:
        return [np.array(values) for key, values in data.items()]
    else:
        return [np.array(data[key]) for key in usecols]


# Dictionary of magic: parser pairs, where magic is first line of file
_text_header_parsers = {
    b'X;Y;valid': (['utf-8'], read_text_header_hfm),
    b'Scan Parameters': (['latin-1'], read_text_header_dektak)
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

    def _read_file(self, header_parser, unit, height_scale_factor, info, max_header_rows, encoding):
        # Default is to autodetect separator and columns
        sep = r'[\s,;]+'  # white space, comma or semicolon
        usecols = None

        # Reopen, but with correct encoding
        with OpenFromAny(self._file_path, mode='r', encoding=encoding) as fobj:
            if header_parser is not None:
                # Read header
                sep, usecols, skiprows, unit, height_scale_factor, _info = header_parser(fobj, unit,
                                                                                         height_scale_factor)
            else:
                skiprows = 0
                _info = {}

            # Update info with user-specified dictionary
            _info.update(info)

            # Reading data, skipping first rows (which may contain headers)
            data_start = fobj.tell()
            data = None
            while data is None and skiprows < max_header_rows:
                try:
                    data = read_csv(fobj, sep=sep, usecols=usecols, skiprows=skiprows)
                except ValueError:
                    data = None
                if data is not None:
                    # Remove columns that contain no numerical data
                    numerical_data = []
                    for arr in data:
                        try:
                            arr = arr.astype(float)
                            if np.isfinite(arr).any():
                                numerical_data += [arr]
                        except ValueError:
                            pass

                    # Check if columns are left
                    if len(numerical_data) == 0:
                        data = None
                    else:
                        if usecols is not None and len(numerical_data) != len(usecols):
                            data = None
                        else:
                            data = numerical_data

                skiprows += 1
                fobj.seek(data_start)

            _log.debug(f'Skipping {skiprows} rows before attempting to read data from XYZ file.')

        return data, unit, height_scale_factor, _info

    def _read(self, physical_sizes=None, height_scale_factor=None, unit=None, info={}, periodic=None,
              max_header_rows=200, tol=1e-6):
        """
        Read the xyz-file. These files contain line scan information in terms of
        (x,y)-positions (for line scans) or (x,y,z)-positions (for topographies).

        Parameters
        ----------
        file : str or file object
             File name or stream.
        max_header_rows : int, optional
             Maximum number of rows to skip when testing for headers.
             (Default: 200)
        tol : float, optional
             Tolerance for detecting uniform grids. (Default: 1e-6)

        Returns
        -------
        topography : Topography or UniformLineScan or NonuniformLineScan
            SurfaceTopography object.
        """
        # Determine maximum length of magic
        maxmagic = 0
        for key in _text_header_parsers.keys():
            maxmagic = max(maxmagic, len(key))

        # Read header (if present) and guess file format
        encoding = ['utf-8', 'utf-16', 'latin-1']  # Open with UTF-8 encoding by default
        header_parser = None  # We don't parse any headers by default
        with OpenFromAny(self._file_path, mode='rb') as fobj:
            first_line = fobj.read(maxmagic)
            for magic, parser in _text_header_parsers.items():
                if first_line.startswith(magic):
                    encoding, header_parser = parser
                    break  # Don't check other file magics since we found one
        # Stream will automatically rewind to old position here

        data = None
        for e in encoding:
            try:
                data, unit, height_scale_factor, _info = self._read_file(header_parser, unit, height_scale_factor, info,
                                                                         max_header_rows, e)
                if data is not None:
                    break
            except UnicodeDecodeError:
                pass

        if data is None:
            raise UnknownFileFormat('Could not parse file as XYZ.')

        if len(data) == 2:
            # This is a line scan.
            kind = 'xz'
            x, z = data
        elif len(data) == 3:
            x, y, z = data
            if ((x[1:] - x[0]) / (y[1:] - y[0])).ptp() < tol:
                # This is a line scan and the first column is likely an index column.
                kind = 'xz'
                x = y
            else:
                # This is a topography map.
                kind = 'xyz'

        # Check is this is a line scan in XY format or two-dimensional data in XYZ format
        if kind == 'xz':
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

        elif kind == 'xyz':
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
