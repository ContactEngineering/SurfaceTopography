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
# from ..Exceptions import FileFormatMismatch

from datetime import datetime

from ..UniformLineScanAndTopography import Topography
from .Reader import ReaderBase, ChannelInfo
from .common import CHANNEL_NAME_INFO_KEY


R_metric_regex = re.compile('(?P<key>[a-zA-Z]+)\s+(?P<value>[+-]?(?:[0-9]*[.])?[0-9]+)\s+(?P<unit>[^\s]+)')


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
        super().__init__()

        close_file = False
        if not hasattr(fobj, 'read'):
            fobj = open(fobj, 'rb')
            close_file = True

        try:
            _dataset_df = pd.read_excel(fobj, sheet_name='DATA', header=None)
            _profile_df = _dataset_df[[4, 5]]
            _profile_df.columns = ('x', 'h')
            _x = _profile_df[['x']].values
            self._profile = _profile_df[['h']].values

            # try extracting simple roughness metrics from first column
            _roughness_metrics_list = list(_dataset_df[~_dataset_df[0].isnull()][0].apply(
                    lambda g: {
                        **{key: float(value) if key == 'value' else value for key, value in
                           R_metric_regex.match(g).groupdict().items()}}
                ))

            _metadata_df = pd.read_excel(fobj, sheet_name='Certificate', header=None)
            _date_string = _metadata_df[4][1]
            # remove all whitespace from date string
            _date_string = re.sub(r"\s+", "", _date_string, flags=re.UNICODE)

            channel_name = 'default'
            info = {}
            info[CHANNEL_NAME_INFO_KEY] = channel_name
            info['roughness_metrics'] = _roughness_metrics_list
            info['acquisition_time'] = str(datetime.strptime(_date_string,
                                                             '%d-%b-%Y'))

            self._physical_sizes = np.max(_x)

            self._channels = [ChannelInfo(self,
                                          0,  # channel index
                                          name='default',
                                          dim=1,
                                          uniform=True,
                                          nb_grid_pts=len(self._profile),
                                          unit='um',
                                          physical_sizes=self._physical_sizes,
                                          info=info)
                              ]
        except Exception as exc:
            raise IOError(exc)
        finally:
            if close_file:
                fobj.close()

    @property
    def channels(self):
        return self._channels

    def topography(self, channel_index=None, physical_sizes=None,
                   height_scale_factor=None, unit=None, info={},
                   periodic=False, subdomain_locations=None,
                   nb_subdomain_grid_pts=None):

        if channel_index is not None and channel_index != 0:
            raise ValueError('`channel_index` must be None or 0.')

        if channel_index is None:
            channel_index = self._default_channel_index

        channel = self._channels[channel_index]

        physical_sizes = self._check_physical_sizes(physical_sizes,
                                                    channel.physical_sizes)

        topography = Topography(
            heights=self._profile,
            physical_sizes=physical_sizes,
            unit=unit,
            info=info
        )

        if height_scale_factor is not None:
            topography = topography.scale(height_scale_factor)

        return topography

    channels.__doc__ = ReaderBase.channels.__doc__
    topography.__doc__ = ReaderBase.topography.__doc__

