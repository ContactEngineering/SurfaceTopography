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

import numpy as np
import pandas as pd
import re

from datetime import datetime

from ..Exceptions import MetadataAlreadyFixedByFile

from ..NonuniformLineScan import NonuniformLineScan
from ..UniformLineScanAndTopography import UniformLineScan
from .Reader import ReaderBase, ChannelInfo

R_metric_regex = re.compile(r'(?P<key>[a-zA-Z]+)\s+(?P<value>[+-]?(?:[0-9]*[.])?[0-9]+)\s+(?P<unit>[^\s]+)')
cut_off_regex = re.compile(r'(?P<value>[+-]?(?:[0-9]*[.])?[0-9]+)\s*(?P<unit>[^\s]+)')


class MitutoyoReader(ReaderBase):
    """
    Mitutoyo SurfTest surface roughness testers produce specifically formatted
    Excel spread sheets.
    """

    _format = 'mitutoyo'
    _name = 'Mitutoyo SurfTest Excel spread sheet (xlsx)'
    _description = '''
Load topography information stored as Excel spread sheet by Mitutoyo SurfTest
surface roughness testers.
    '''

    def __init__(self, fobj):
        """
        Open Excel spread sheet produced by Mitutoyo SurfTest surface roughness
        testers.

        The reader expects a line scan by positions and heights in columns
        5 and 6 (E, F) and tries to extract standard roughness metrics from
        column 1 (A) on the sheet 'DATA'. The reader extracts expects the
        acquisition date on sheet in column 5 (E) row 2 on sheet 'Certificate'.

        Parameters
        ----------
        fobj : filename or file object
             File to read.
        """

        try:
            # Pandas exceptions not documented for pd.read_excel
            _dataset_df = pd.read_excel(fobj, sheet_name='DATA', header=None)
            _metadata_df = pd.read_excel(fobj, sheet_name='Certificate', header=None)
            # How should a reader properly fulfill test_no_resource_warning_on_failure?
        except Exception as exc:
            # Pass Pandas exceptions at this point on a `IOError`s
            raise IOError(exc)
        else:
            _profile_df = _dataset_df[[4, 5]]
            _profile_df.columns = ('x', 'h')
            _x = _profile_df['x'].values
            # We store the profile (and metadata) in the object since there
            # can only be a single data channel.
            self._h = _profile_df['h'].values

            # check if positions are distributed uniformly
            _diff = np.diff(_x)
            if np.all(np.isclose(_diff, _diff[0])):
                self._uniform = True
            else:
                self._uniform = False
                # height values are assigned to the "end" of each "pixel" in the
                # xlsx format, starting at non-zero x. Here we shift positions
                # by half a distance each to have non-uniform profiles centered
                _x = _x - np.insert(_diff, 0, _x[0]) * 0.5

            # try extracting simple roughness metrics from first column
            _roughness_metrics_list = list(_dataset_df[~_dataset_df[0].isnull()][0].apply(
                lambda g: {
                    **{key: float(value) if key == 'value' else value for key, value in
                       R_metric_regex.match(g).groupdict().items()}}
            ))

            # try to infer heights unit from roughness metrics
            _h_unit = _roughness_metrics_list[0]['unit']

            _date_string = _metadata_df[4][1]

            # remove all whitespace from date string
            _date_string = re.sub(r"\s+", "", _date_string, flags=re.UNICODE)

            # try to infer x unit from cut off
            _cut_off_string = _metadata_df[4][47]
            _cut_off_dict = cut_off_regex.match(_cut_off_string).groupdict()
            _x_unit = _cut_off_dict['unit']

            # try to convert x unit to h unit if pint available
            try:
                import pint
                ureg = pint.UnitRegistry()
                self._x = (_x * ureg[_x_unit].to(ureg[_h_unit])).magnitude
            except ImportError:  # if pint not available, assert it's mm to um
                if _x_unit != 'mm' or _h_unit not in set(['µm', 'um']):
                    raise ValueError(
                        "Unexpected unit pairing [x] = %s and [h] = %s",
                        _x_unit, _h_unit)
                self._x = _x * 1000.0  # convert mm to um

            self._unit = _h_unit

            # we assume the data series to start at zero x
            self._physical_sizes = np.max(self._x) - np.min(self._x)

            self._info = {
                'roughness_metrics': _roughness_metrics_list,
                'cut_off': _cut_off_dict,
                'acquisition_time': str(datetime.strptime(_date_string, '%d-%b-%Y'))
            }
            self._channels = [ChannelInfo(self,
                                          0,  # channel index
                                          # Since the is only a single channel and the file has no channel information,
                                          # the name is 'Default'
                                          name='Default',
                                          dim=1,
                                          uniform=self._uniform,
                                          nb_grid_pts=len(self._h),
                                          unit=self._unit,
                                          physical_sizes=self._physical_sizes,
                                          info=self._info)
                              ]

    # Return list of channels (here only a single one)
    @property
    def channels(self):
        return self._channels

    def topography(self, channel_index=None, physical_sizes=None,
                   height_scale_factor=None, unit=None, info={},
                   periodic=None, subdomain_locations=None,
                   nb_subdomain_grid_pts=None):

        # Check if only a subdomain should be loaded
        if subdomain_locations is not None or \
                nb_subdomain_grid_pts is not None:
            raise RuntimeError(
                'This reader does not support MPI parallelization.')

        # Check that channel_index is valid
        if channel_index is not None and channel_index != 0:
            raise ValueError('`channel_index` must be None or 0.')

        # If no channel index is given, we use the default index
        if channel_index is None:
            channel_index = self._default_channel_index

        # Units are specified in the XLSX file and cannot be overridden
        if unit is not None:
            if self._unit is not None:
                raise MetadataAlreadyFixedByFile('unit')
        else:
            unit = self._unit
        # Augment info dictionary with user-specified data
        _info = self._info.copy()
        _info.update(info)

        # Get channel information (there is only one)
        channel = self._channels[channel_index]

        # Make sure `physical_sizes` is present, either fixed by the file
        # or given by the user. Also make sure that if the user specifies it
        # it cannot be set in the file. (We cannot override metadata from
        # files.)
        physical_sizes = self._check_physical_sizes(physical_sizes,
                                                    channel.physical_sizes)

        # may a reader return either NonuniformLineScan or UniformLineScan?
        if self._uniform:
            # Return a uniform line scan if data is equally spaced
            topography = UniformLineScan(
                self._h, physical_sizes,
                periodic=False if periodic is None else periodic,
                unit=unit,
                info=_info)
        else:
            if periodic is not None and periodic:
                raise ValueError('Mitutoyo reader found nonuniform data, and the user specified that it is periodic. '
                                 'Nonuniform line scans cannot be periodic.')

            # Return a nonuniform line scan otherwise
            topography = NonuniformLineScan(
                self._x,
                self._h,
                unit=unit,
                info=_info)

        # Check if there is a user-specified height scale factor
        if height_scale_factor is not None:
            topography = topography.scale(height_scale_factor)

        return topography

    channels.__doc__ = ReaderBase.channels.__doc__
    topography.__doc__ = ReaderBase.topography.__doc__
