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

from PyCo.Topography.IO import NPYReader

# These imports are required to register the analysis functions!
import PyCo.Topography.Converters
import PyCo.Topography.Uniform.common
import PyCo.Topography.Uniform.Interpolation
import PyCo.Topography.Uniform.Autocorrelation
import PyCo.Topography.Uniform.PowerSpectrum
import PyCo.Topography.Uniform.ScalarParameters
import PyCo.Topography.Uniform.VariableBandwidth
import PyCo.Topography.Nonuniform.common
import PyCo.Topography.Nonuniform.Autocorrelation
import PyCo.Topography.Nonuniform.ScalarParameters
import PyCo.Topography.Nonuniform.PowerSpectrum
import PyCo.Topography.Nonuniform.VariableBandwidth
