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

from PyCo.SurfaceTopography.IO import NPYReader

# These imports are required to register the analysis functions!
import PyCo.SurfaceTopography.Converters
import PyCo.SurfaceTopography.Uniform.common
import PyCo.SurfaceTopography.Uniform.Interpolation
import PyCo.SurfaceTopography.Uniform.Filtering
import PyCo.SurfaceTopography.Uniform.Autocorrelation
import PyCo.SurfaceTopography.Uniform.PowerSpectrum
import PyCo.SurfaceTopography.Uniform.ScalarParameters
import PyCo.SurfaceTopography.Uniform.VariableBandwidth
import PyCo.SurfaceTopography.Nonuniform.common
import PyCo.SurfaceTopography.Nonuniform.Autocorrelation
import PyCo.SurfaceTopography.Nonuniform.ScalarParameters
import PyCo.SurfaceTopography.Nonuniform.PowerSpectrum
import PyCo.SurfaceTopography.Nonuniform.VariableBandwidth

try:
    from importlib.metadata import version
    __version__ = version(__name__)
except ImportError:
    from pkg_resources import get_distribution
    __version__ = get_distribution(__name__).version
