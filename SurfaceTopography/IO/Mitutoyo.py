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
from .common import CHANNEL_NAME_INFO_KEY


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
Load topography information stored as excel spread sheet by Mitutoyo SurfTest
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

        # Should readers call constructor of base class? NPY does
        super().__init__()

        # Should these fields be initialized? NC does
        self._physical_sizes = None
        self._periodic = False
        self._unit = None
        self._info = {}

        # Should the reader explicitly open the file if not yet done so? DI does
        close_file = False
        if not hasattr(fobj, 'read'):
            fobj = open(fobj, 'rb')
            close_file = True

        try:
            # pandas exceptions not documented for pd.read_excel
            _dataset_df = pd.read_excel(fobj, sheet_name='DATA', header=None)
            _metadata_df = pd.read_excel(fobj, sheet_name='Certificate', header=None)
            # How should a reader properly fulfill test_no_resource_warning_on_failure?
        except Exception as exc:
            raise IOError(exc)
        else:
            _profile_df = _dataset_df[[4, 5]]
            _profile_df.columns = ('x', 'h')
            _x = _profile_df['x'].values
            self._profile = _profile_df['h'].values

            # check if positions are distributed uniformly
            _diff = np.diff(_x)
            if np.all(np.isclose(_diff, _diff[0])):
                self._uniform = True
            else:
                self._uniform = False

            # try extracting simple roughness metrics from first column
            _roughness_metrics_list = list(_dataset_df[~_dataset_df[0].isnull()][0].apply(
                    lambda g: {
                        **{key: float(value) if key == 'value' else value for key, value in
                           R_metric_regex.match(g).groupdict().items()}}
                ))

            # try to infer heights unit from roughness metrics
            if _roughness_metrics_list[0]['unit'] == 'µm':
                _h_unit = 'um'
            else:  # No test case for other unit than um so far
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
                self._x = _x*ureg[_x_unit].to(ureg[_h_unit])
            except ImportError:  # if pint not available, assert it's mm to um
                if _x_unit != 'mm' or _h_unit != 'um':
                    raise ValueError(
                        "Unexpected unit pairing [x] = %s and [h] = %s",
                        _x_unit, _h_unit)
                self._x = _x*1000.0  # convert mm to um

            self._unit = _h_unit

            # we assume the data series to start at zero x
            self._physical_sizes = np.max(self._x)

            channel_name = 'default'
            self._info[CHANNEL_NAME_INFO_KEY] = channel_name
            self._info['roughness_metrics'] = _roughness_metrics_list
            self._info['cut_off'] = _cut_off_dict
            self._info['acquisition_time'] = str(datetime.strptime(
                                                 _date_string, '%d-%b-%Y'))
            # Should the name of a single channel be 'Default'?
            # That's the case for NPY or NC
            self._channels = [ChannelInfo(self,
                                          0,  # channel index
                                          name='Default',
                                          dim=1,
                                          uniform=self._uniform,
                                          nb_grid_pts=len(self._profile),
                                          unit=self._unit,
                                          physical_sizes=self._physical_sizes,
                                          info=self._info)
                              ]
        finally:
            if close_file:
                fobj.close()

    # must a reader define this property? most do
    @property
    def channels(self):
        return self._channels

    def topography(self, channel_index=None, physical_sizes=None,
                   height_scale_factor=None, unit=None, info={},
                   periodic=False, subdomain_locations=None,
                   nb_subdomain_grid_pts=None):

        # Should a reader check on this? NPY does
        if channel_index is not None and channel_index != 0:
            raise ValueError('`channel_index` must be None or 0.')

        # Should a reader assume a default channel index? DI does
        if channel_index is None:
            channel_index = self._default_channel_index

        # Should a reader prohibit overriding unit in this way? DI does
        if unit is not None:
            if self._unit is not None:
                raise MetadataAlreadyFixedByFile('unit')
        else:
            unit = self._unit

        # Should a reader augment its info like this? NC does
        _info = self._info.copy()
        _info.update(info)

        channel = self._channels[channel_index]

        # Should a reader retrieve physical_sizes like this? NC does ...
        physical_sizes = self._check_physical_sizes(physical_sizes,
                                                    channel.physical_sizes)
        # .. but NPY only does
        # physical_sizes = self._check_physical_sizes(physical_sizes)

        # may a reader return either NonuniformLineScan or UniformLineScan?
        if self._uniform:
            # Should a reader return Topography or UniformLineScan for uniform 1D data?
            topography = UniformLineScan(
                self._profile, physical_sizes,
                periodic=self._periodic if periodic is None else periodic,
                unit=unit,
                info=_info)
        else:
            topography = NonuniformLineScan(
                self._x,
                self._profile,
                unit=unit,
                info=_info)

        if height_scale_factor is not None:
            topography = topography.scale(height_scale_factor)

        return topography

    channels.__doc__ = ReaderBase.channels.__doc__
    topography.__doc__ = ReaderBase.topography.__doc__
