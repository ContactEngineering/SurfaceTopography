#
# Copyright 2020 Lars Pastewka
#           2020 Antoine Sanner
#           2020 Michael Röttger
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
import matplotlib.pyplot as plt

from SurfaceTopography import open_topography


# from SurfaceTopography import SurfaceTopography

# plt.ion()
# fn='../data/issues/230/di1.txt'
# fn='tests/file_format_examples/opdx-2.opdx'
# fn='tests/file_format_examples/opd-1.opd'
# fn='tests/file_format_examples/example2.x3p'
# fn='tests/file_format_examples/mi-1.mi'

def plot(fn):
    r = open_topography(fn)

    t = r.topography()
    if 'unit' in t.info:
        unit = t.info['unit']
    else:
        unit = '?'
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("{}, channel {}".format(fn, r.default_channel.name))

    ax = fig.add_subplot(2, 2, 1)
    ax.set_title("pcolormesh(t.heights().T)")
    ax.pcolormesh(t.heights().T)

    ax = fig.add_subplot(2, 2, 2)
    ax.set_title("pcolormesh(*t.positions_and_heights())")
    ax.set_xlabel("x [{}]".format(unit))
    ax.set_ylabel("y [{}]".format(unit))
    ax.pcolormesh(*t.positions_and_heights())

    ax = fig.add_subplot(2, 2, 3)
    ax.set_title("above is correct, if this is like Gwyddion")
    ax.pcolormesh(np.flipud(t.heights().T))

    ax = fig.add_subplot(2, 2, 4)
    ax.set_title("imshow(t.heights().T)")
    extent = (
        0, t.physical_sizes[0], t.physical_sizes[1], 0
    )
    ax.imshow(t.heights().T, extent=extent)
    fig.subplots_adjust(hspace=0.5)
    fig.show()

    h = t.heights()
    for i, j in [(0, 0), (0, -1), (-1, 0), (-1, -1)]:
        print("h[{},{}] == {}".format(i, j, h[i, j]))

    return t


if __name__ == '__main__':

    plt.close('all')
    # plt.ion()

    filenames = [
        # '../data/issues/230/di1.txt',  # this file you have to export
        # yourself with gwyddion
        'tests/file_format_examples/di-1.di',
        'tests/file_format_examples/opdx-2.opdx',
        'tests/file_format_examples/opd-1.opd',
        'tests/file_format_examples/example2.x3p',
        'tests/file_format_examples/mi-1.mi',
        'tests/file_format_examples/ibw-1.ibw',
    ]

    for fn in filenames:
        t = plot(fn)

    input("Press enter to proceed - last topography in variable 't'")
