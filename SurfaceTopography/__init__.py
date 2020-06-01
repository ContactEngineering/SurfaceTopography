#
# Copyright 2018, 2020 Antoine Sanner
#           2018-2019 Lars Pastewka
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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
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
defines all surface types used in PyCo
"""

from .UniformLineScanAndTopography import Topography, UniformLineScan
from .HeightContainer import UniformTopographyInterface
from .NonuniformLineScan import NonuniformLineScan
from .Special import make_sphere, PlasticTopography
from .IO import open_topography, read_topography

from SurfaceTopography.IO import NPYReader

# These imports are required to register the analysis functions!
import SurfaceTopography.Converters
import SurfaceTopography.Uniform.common
import SurfaceTopography.Uniform.Interpolation
import SurfaceTopography.Uniform.Filtering
import SurfaceTopography.Uniform.Autocorrelation
import SurfaceTopography.Uniform.PowerSpectrum
import SurfaceTopography.Uniform.ScalarParameters
import SurfaceTopography.Uniform.VariableBandwidth
import SurfaceTopography.Nonuniform.common
import SurfaceTopography.Nonuniform.Autocorrelation
import SurfaceTopography.Nonuniform.ScalarParameters
import SurfaceTopography.Nonuniform.PowerSpectrum
import SurfaceTopography.Nonuniform.VariableBandwidth

try:
    from importlib.metadata import version
    __version__ = version(__name__)
except ImportError:
    from pkg_resources import get_distribution
    __version__ = get_distribution(__name__).version
