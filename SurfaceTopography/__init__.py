#
# Copyright 2017-2021 Lars Pastewka
#           2018-2020 Antoine Sanner
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

from .Container import SurfaceContainer, read_container, read_published_container  # noqa: F401
from .HeightContainer import UniformTopographyInterface  # noqa: F401
from .IO import open_topography, read_topography  # noqa: F401
from .NonuniformLineScan import NonuniformLineScan  # noqa: F401
from .Special import make_sphere, PlasticTopography  # noqa: F401
from .UniformLineScanAndTopography import Topography, UniformLineScan  # noqa: F401

# These imports are required to register the analysis functions!
import SurfaceTopography.Converters  # noqa: F401
import SurfaceTopography.Generic.Curvature  # noqa: F401
import SurfaceTopography.Generic.ScanningProbe  # noqa: F401
import SurfaceTopography.Generic.Slope  # noqa: F401
import SurfaceTopography.Nonuniform.common  # noqa: F401
import SurfaceTopography.Nonuniform.Autocorrelation  # noqa: F401
import SurfaceTopography.Nonuniform.ScaleDependentStatistics  # noqa: F401
import SurfaceTopography.Nonuniform.ScalarParameters  # noqa: F401
import SurfaceTopography.Nonuniform.PowerSpectrum  # noqa: F401
import SurfaceTopography.Nonuniform.VariableBandwidth  # noqa: F401
import SurfaceTopography.Uniform.common  # noqa: F401
import SurfaceTopography.Uniform.Interpolation  # noqa: F401
import SurfaceTopography.Uniform.Filtering  # noqa: F401
import SurfaceTopography.Uniform.Autocorrelation  # noqa: F401
import SurfaceTopography.Uniform.PowerSpectrum  # noqa: F401
import SurfaceTopography.Uniform.ScaleDependentStatistics  # noqa: F401
import SurfaceTopography.Uniform.ScalarParameters  # noqa: F401
import SurfaceTopography.Uniform.VariableBandwidth  # noqa: F401

try:
    from importlib.metadata import version

    __version__ = version(__name__)
except ImportError:
    from pkg_resources import get_distribution

    __version__ = get_distribution(__name__).version
