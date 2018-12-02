#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   __init__.py

@author Till Junge <till.junge@kit.edu>

@date   26 Jan 2015

@brief  defines all surface types used in PyCo

@section LICENCE

Copyright 2015-2017 Till Junge, Lars Pastewka

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from .common import compute_derivative
from .Autocorrelation import autocorrelation_1D, autocorrelation_2D
from .Detrending import (tilt_from_height, tilt_from_slope, tilt_and_curvature, shift_and_tilt,
                         shift_and_tilt_approx, shift_and_tilt_from_slope)
from .PowerSpectrum import power_spectrum_1D, power_spectrum_2D
from .ScalarParameters import rms_height, rms_slope, rms_curvature
from .TopographyDescription import (CompoundTopography, DetrendedTopography,
                                    UniformNumpyTopography, PlasticTopography,
                                    ScaledTopography, Sphere, Topography,
                                    TranslatedTopography)

from .FromFile import (NumpyTxtSurface, NumpyAscSurface, read, read_asc,
                       read_di, read_h5, read_hgt, read_ibw, read_mat,
                       read_matrix, read_opd, read_x3p, read_xyz)
from .ParallelFromFile import MPITopographyLoader