#
# Copyright 2017-2024 Lars Pastewka
#           2018-2020, 2022-2023 Antoine Sanner
#           2015-2016 Till Junge
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

"""
defines all surface types used in SurfaceTopography
"""

# These imports are required to register the analysis functions!
import SurfaceTopography.Generic.Curvature  # noqa: F401
import SurfaceTopography.Generic.Fractional  # noqa: F401
import SurfaceTopography.Generic.ReliabilityCutoff  # noqa: F401
import SurfaceTopography.Generic.RobustStatistics  # noqa: F401
import SurfaceTopography.Generic.ScaleDependentStatistics  # noqa: F401
import SurfaceTopography.Generic.ScanningProbe  # noqa: F401
import SurfaceTopography.Generic.Slope  # noqa: F401
import SurfaceTopography.Nonuniform.Autocorrelation  # noqa: F401
import SurfaceTopography.Nonuniform.BearingArea  # noqa: F401
# import SurfaceTopography.Models # noqa: F401
import SurfaceTopography.Nonuniform.common  # noqa: F401
import SurfaceTopography.Nonuniform.Converters  # noqa: F401
import SurfaceTopography.Nonuniform.Derivative  # noqa: F401
import SurfaceTopography.Nonuniform.Detrending  # noqa: F401
import SurfaceTopography.Nonuniform.Interpolation  # noqa: F401
import SurfaceTopography.Nonuniform.PowerSpectrum  # noqa: F401
import SurfaceTopography.Nonuniform.ScalarParameters  # noqa: F401
import SurfaceTopography.Nonuniform.VariableBandwidth  # noqa: F401
import SurfaceTopography.ScanningProbe  # noqa: F401
import SurfaceTopography.Support.Bibliography  # noqa: F401
import SurfaceTopography.Uniform.Autocorrelation  # noqa: F401
import SurfaceTopography.Uniform.BearingArea  # noqa: F401
import SurfaceTopography.Uniform.common  # noqa: F401
import SurfaceTopography.Uniform.Converters  # noqa: F401
import SurfaceTopography.Uniform.Derivative  # noqa: F401
import SurfaceTopography.Uniform.Detrending  # noqa: F401
import SurfaceTopography.Uniform.Filtering  # noqa: F401
import SurfaceTopography.Uniform.Imputation  # noqa: F401
import SurfaceTopography.Uniform.Integration  # noqa: F401
import SurfaceTopography.Uniform.Interpolation  # noqa: F401
import SurfaceTopography.Uniform.PowerSpectrum  # noqa: F401
import SurfaceTopography.Uniform.ScalarParameters  # noqa: F401
import SurfaceTopography.Uniform.VariableBandwidth  # noqa: F401

from .Container import (open_container, read_container,  # noqa: F401
                        read_published_container)
# This needs to be the first import, because __version__ is needed in the
# packages that are subsequently imported.
from .DiscoverVersion import __version__  # noqa: F401
from .IO import open_topography, read_topography  # noqa: F401
from .NonuniformLineScan import NonuniformLineScan  # noqa: F401
from .Special import PlasticTopography, make_sphere  # noqa: F401
from .UniformLineScanAndTopography import Topography  # noqa: F401
from .UniformLineScanAndTopography import UniformLineScan  # noqa: F401

# Add contact.engineering paper to bibliography
SurfaceTopography.Support.Bibliography._default_dois = set(['10.1088/2051-672X/ac860a'])
