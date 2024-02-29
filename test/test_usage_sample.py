#
# Copyright 2023 Lars Pastewka
#           2022 Johannes Hörmann
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

"""Tests for usage samples in documentation."""

import os

from SurfaceTopography import open_topography


def test_usage_sample(file_format_examples):
    # get a handle to the file ("reader")
    reader = open_topography(os.path.join(file_format_examples, "opd-1.opd"))

    # each file has a list of channels (one or more)
    print(reader.channels)  # returns list of channels
    ch = reader.channels[0]  # first channel, alternatively use ..
    ch = reader.default_channel  # .. - one of the channels is the "default" channel

    # each channel has some defined meta data
    print(ch.name)  # channel name
    print(ch.physical_sizes)  # lateral dimensions
    print(ch.nb_grid_pts)  # number of grid points
    print(ch.dim)  # number of dimensions (1 or 2)
    print(ch.info)  # more metadata, e.g. 'unit' if unit was given in file

    # you can get a topography from a channel
    topo = ch.topography()   # here meta data from the file is taken
    # topo = ch.topography(physical_sizes=(20,30))   # like this, you can overwrite meta data in file

    # each topography has a rich set of methods and properties for meta data and analysis
    print(topo.physical_sizes)  # lateral dimension
    print(topo.rms_height_from_area())  # Root mean square of heights
