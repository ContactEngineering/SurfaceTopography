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

"""Compute properties as averages over individual results."""

import logging

import numpy as np
from collections import defaultdict

from ..Support.Regression import resample
from .SurfaceContainer import SurfaceContainer

_log = logging.Logger(__name__)


def log_average(self, function_name, unit, nb_points_per_decade=10, reliable=True, progress_callback=None, **kwargs):
    """
    Return average property on a logarithmic grid.

    Parameters
    ----------
    self : SurfaceContainer
        Container object containing the underlying data sets.
    function_name : str
        Function for which to compute the average, e.g.
        'autocorrelation_from_profile'.
    unit : str
        Unit of the distance array. All topographies are converted to this
        unit before the properties are computed.
    nb_points_per_decade : int, optional
        Number of points per decade in length for automatic grid construction.
        (Default: 10)
    reliable : bool, optional
        Only return data deemed reliable. (Default: True)
    progress_callback : func, optional
        Function taking iteration and the total number of iterations as
        arguments for progress reporting. (Default: None)
    """
    results = defaultdict(list)
    for i, topography in enumerate(self):
        if progress_callback is not None:
            progress_callback(i, len(self))

        topography = topography.to_unit(unit)

        # Compute property... but do not allow any resampling
        func = getattr(topography, function_name)
        x, y = func(reliable=reliable, resampling_method=None, **kwargs)

        # Construct grid in this range
        m = x > 0
        if np.sum(m) < 1:
            _log.warning(f'Topography {topography} contributes no data to average.')
            continue
        lower, upper = np.min(x[m]), np.max(x)
        lower_decade, upper_decade = int(np.floor(np.log10(lower))), int(np.ceil(np.log10(upper)))

        # Resample onto the same grid
        collocation_points, bin_edges, resampled_y, variance = resample(
            x, y, collocation='log', min_value=10 ** lower_decade, max_value=10 ** upper_decade,
            nb_points=(upper_decade - lower_decade) * nb_points_per_decade + 1)

        # Store for later averaging
        for j, (xval, yval) in enumerate(zip(collocation_points, resampled_y)):
            results[lower_decade * nb_points_per_decade + j] += [(xval, yval)]

    if progress_callback is not None:
        progress_callback(len(self), len(self))

    avgresults = np.array(sorted([np.nanmean(vals, axis=0) for vals in results.values()], key=lambda x: x[0]))
    return avgresults.T[0], avgresults.T[1]


# All container functionsa are 'profile' functions, because container can contain mixtures of topgoraphic maps and line
# scans.
SurfaceContainer.register_function(
    'autocorrelation', lambda self, unit, **kwargs: log_average(self, 'autocorrelation_from_profile', unit, **kwargs))
SurfaceContainer.register_function(
    'power_spectrum', lambda self, unit, **kwargs: log_average(self, 'power_spectrum_from_profile', unit, **kwargs))
SurfaceContainer.register_function(
    'variable_bandwidth',
    lambda self, unit, **kwargs: log_average(self, 'variable_bandwidth_from_profile', unit, **kwargs))
