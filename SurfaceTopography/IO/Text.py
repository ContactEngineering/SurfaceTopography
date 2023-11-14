#
# Copyright 2018-2023 Lars Pastewka
#           2018-2021 Michael Röttger
#           2019-2020 Antoine Sanner
#           2019 Kai Haase
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

import re

import numpy as np

from ..Exceptions import MetadataAlreadyFixedByFile
from ..HeightContainer import UniformTopographyInterface
from ..UniformLineScanAndTopography import Topography, UniformLineScan
from ..Support.UnitConversion import length_units, mangle_length_unit_utf8
from .common import CHANNEL_NAME_INFO_KEY, text
from .FromFile import make_wrapped_reader


@text()
def read_matrix(fobj, physical_sizes=None, unit=None, height_scale_factor=None, periodic=False):
    """
    Reads a surface profile from a text file and presents in in a
    SurfaceTopography-conformant manner. No additional parsing of
    meta-information is carried out.

    Keyword Arguments:
    fobj -- filename or file object
    """
    arr = np.loadtxt(fobj)
    if physical_sizes is None:
        surface = Topography(arr, arr.shape, periodic=periodic, unit=unit)
    else:
        surface = Topography(arr, physical_sizes, periodic=periodic, unit=unit)
    if height_scale_factor is not None:
        surface = surface.scale(height_scale_factor)
    return surface


MatrixReader = make_wrapped_reader(
    read_matrix, class_name="MatrixReader", format='matrix', mime_types=['text/plain'],
    file_extensions=['txt', 'asc', 'dat'],
    name='Plain text (matrix)')


@text()
def read_asc(fobj, physical_sizes=None, height_scale_factor=None, x_factor=1.0,
             z_factor=None, unit=None, info={}, periodic=False):
    # pylint: disable=too-many-branches,too-many-statements,invalid-name
    """
    Reads a surface profile (topography) from an generic asc file and presents
    it in a surface-conformant manner. Applies some heuristic to extract
    meta-information for different file formats.

    The info dict of the topography returned is a copy from the given info dict
    and may have some extra keys added:
    - "unit": a common unit for the data, for dimensions and heights
    - "channel_name": the name of the channel (if unknown "Default" is used")

    Parameters
    ----------

    fobj: filename or file object
    physical_sizes: tuple or None
    height_scale_factor: float or None
    x_factor: float
        multiplication factor for physical_sizes
    z_factor: float or None
        multiplication factor for height
    info: dict
    periodic: bool
    """
    _float_regex = r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'

    checks = list()
    # Resolution keywords
    checks.append((re.compile(r"\b(?:x-pixels|h)\b\s*=\s*([0-9]+)"), int,
                   "yres"))
    checks.append((re.compile(r"\b(?:y-pixels|w)\b\s*=\s*([0-9]+)"), int,
                   "xres"))

    # Size keywords
    checks.append(
        (re.compile(r"\b(?:x-length|Width|Breite)\b\s*(?:=|\:)\s*(?P<value>" +
                    _float_regex + ")(?P<unit>.*)"), float, "xsiz"))
    checks.append(
        (re.compile(r"\b(?:y-length|Height|Höhe)\b\s*(?:=|\:)\s*(?P<value>" +
                    _float_regex + ")(?P<unit>.*)"), float, "ysiz"))

    # Unit keywords
    checks.append(
        (re.compile(r"\b(?:x-unit)\b\s*(?:=|\:)\s*(\w+)"), str, "xunit"))
    checks.append(
        (re.compile(r"\b(?:y-unit)\b\s*(?:=|\:)\s*(\w+)"), str, "yunit"))
    checks.append(
        (re.compile(r"\b(?:z-unit|Value units)\b\s*(?:=|\:)\s*(\w+)"),
         str, "zunit"))

    # Scale factor keywords
    checks.append(
        (re.compile(r"(?:pixel\s+size)\s*=\s*(?P<value>" + _float_regex +
                    ")(?P<unit>.*)"), float, "xfac"))
    checks.append((
        re.compile((
                r"(?:height\s+conversion\s+factor\s+\(->\s+(?P<unit>.*)\))\s*="
                r"\s*(?P<value>" + _float_regex + ")")),
        float, "zfac"))
    # Channel name keywords
    checks.append((
        re.compile(r"\b(?:Channel|Kanal)\b\s*(?:=|\:)\s*([\w|\s]+)"),
        str, "channel_name"))

    xres = yres = xsiz = ysiz = xunit = yunit = zunit = xfac = yfac = None
    zfac = None
    channel_name = "Default"

    def process_comment(line):
        """Find and interpret known comments in the header of the asc file"""
        nonlocal xres, yres, xsiz, ysiz, xunit, yunit, zunit, data, xfac, yfac
        nonlocal zfac
        nonlocal channel_name
        # TODO Is 'fun' needed here? No used so far..
        for reg, fun, key in checks:
            match = reg.search(line)
            if match is None:
                continue
            if key == 'xres':
                xres = int(match.group(1))
            elif key == 'yres':
                yres = int(match.group(1))
            elif key == 'xsiz':
                xsiz = float(match.group('value'))
                x = match.group('unit')
                if x:
                    xunit = mangle_length_unit_utf8(x)
            elif key == 'ysiz':
                ysiz = float(match.group('value'))
                y = match.group('unit')
                if y:
                    yunit = mangle_length_unit_utf8(y)
            elif key == 'xunit':
                xunit = mangle_length_unit_utf8(match.group(1))
            elif key == 'yunit':
                yunit = mangle_length_unit_utf8(match.group(1))
            elif key == 'zunit':
                zunit = mangle_length_unit_utf8(match.group(1))
            elif key == 'xfac':
                xfac = float(match.group('value'))
                xunit = mangle_length_unit_utf8(match.group('unit'))
            elif key == 'zfac':
                zfac = float(match.group('value'))
                zunit = mangle_length_unit_utf8(match.group('unit'))
            elif key == 'channel_name':
                channel_name = match.group(1).strip()

    data = []
    for line in fobj:
        line_elements = line.strip().split()
        if len(line_elements) > 0:
            try:
                data += [[float(strval) for strval in line_elements]]
            except ValueError:
                process_comment(line)

    if (height_scale_factor is not None) and (zfac is not None):
        # it should not be allowed to override a height scale factor if
        # there is one already in the file
        raise MetadataAlreadyFixedByFile('height_scale_factor')

    data = np.array(data).T
    nx, ny = data.shape
    if nx == 2 or ny == 2:
        raise Exception(
            "This file has just two rows or two columns and is more likely a "
            "line scan than a map.")
    if xres is not None and xres != nx:
        raise Exception(
            "The number of rows (={}) open_topography from the file '{}' "
            "does not match the nb_grid_pts in the file's metadata (={}).".format(nx, fobj, xres))
    if yres is not None and yres != ny:
        raise Exception(
            "The number of columns (={}) open_topography from the file '{}' "
            "does not match the nb_grid_pts in the file's metadata "
            "(={}).".format(ny, fobj, yres))

    # Handle scale factors
    if xfac is not None and yfac is None:
        yfac = xfac
    elif xfac is None and yfac is not None:
        xfac = yfac
    if xfac is not None:
        if xsiz is None:
            xsiz = xfac * nx
        else:
            xsiz *= xfac
    if yfac is not None:
        if ysiz is None:
            ysiz = yfac * ny
        else:
            ysiz *= yfac
    if z_factor is not None:
        zfac = z_factor if zfac is None else zfac * z_factor

    info = info.copy()

    # Handle units -> convert to target unit
    if xunit is None and zunit is not None:
        xunit = zunit
    if yunit is None and zunit is not None:
        yunit = zunit

    if unit is None:
        unit = zunit
    if unit is not None:
        if xunit is not None:
            xsiz *= length_units[xunit] / length_units[unit]
        if yunit is not None:
            ysiz *= length_units[yunit] / length_units[unit]
        if zunit is not None:
            if zfac is None:
                if length_units[zunit] != length_units[unit]:
                    zfac = length_units[zunit] / length_units[unit]
            else:
                zfac *= length_units[zunit] / length_units[unit]

    # handle channel name
    # we use the info dict here to transfer the channel name
    info[CHANNEL_NAME_INFO_KEY] = channel_name

    # calculate physical sizes and generate topography
    if xsiz is not None and ysiz is not None:
        if physical_sizes is None:
            physical_sizes = (x_factor * xsiz, x_factor * ysiz)
        else:
            raise MetadataAlreadyFixedByFile('physical_sizes')
    if data.shape[0] == 1:
        if physical_sizes is not None and len(physical_sizes) > 1:
            physical_sizes = physical_sizes[0]
        surface = UniformLineScan(data[0, :], physical_sizes, unit=unit, info=info, periodic=periodic)
    else:
        surface = Topography(data, physical_sizes, unit=unit, info=info, periodic=periodic)
    if height_scale_factor is not None:
        zfac = height_scale_factor
    if zfac is not None:
        surface = surface.scale(zfac)
    return surface


AscReader = make_wrapped_reader(read_asc, class_name="AscReader", format='asc', mime_types=['text/plain'],
                                file_extensions=['txt', 'asc', 'dat'],
                                name='Plain text (with headers)',
                                description='''
SurfaceTopography data stored in plain text (ASCII) format needs to be stored
in a matrix format. Each row contains the height information for subsequent
points in x-direction separated by a whitespace. The next row belong to the
following y-coordinate. Note that if the file has three or less columns, it
will be interpreted as a topography stored in a coordinate format (the three
columns contain the x, y and z coordinates of the same points). The smallest
topography that can be provided in this format is therefore 4 x 1.

The reader supports parsing file headers for additional metadata. This allows
to specify the physical size of the topography and the unit. In particular, it
supports reading ASCII files exported from SPIP and Gwyddion.

When writing your own ASCII files, we recommend to prepent the header with a
'#'. The following file is an example that contains 4 x 3 data points:
```
# Channel: Main
# Width: 10 µm
# Height: 10 µm
# Value units: m
 1.0  2.0  3.0  4.0
 5.0  6.0  7.0  8.0
 9.0 10.0 11.0 12.0
```
''')


def write_matrix(self, fname):
    """
    Saves the topography using `np.savetxt`. Warning: This only saves
    the heights; the physical_sizes is not contained in the file
    """
    np.savetxt(fname, self.heights())


# Register analysis functions from this module
UniformTopographyInterface.register_function('to_matrix', write_matrix)
