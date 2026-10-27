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

"""
Filter pipelines for data imputation (filling undefined data points)
"""

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from ..HeightContainer import UniformTopographyInterface
from ..UniformLineScanAndTopography import DecoratedUniformTopography


class InterpolateUndefinedDataHarmonic(DecoratedUniformTopography):
    """
    Replace undefined data points by interpolation of neighboring
    points with harmonic functions (solutions of the Laplace equation).
    """

    name = 'interpolate_undefined_data_harmonic'

    def __init__(self, topography, info={}):
        super().__init__(topography, info=info)

    @property
    def has_undefined_data(self):
        """
        By definition, this topography has no undefined data.
        """
        return False

    def heights(self):
        """
        Computes the topography with filled in data points.
        """
        heights = self.parent_topography.heights().copy()
        if super().has_undefined_data:
            dim = self.dim

            # Undefined points are the unknowns of a discrete Laplace equation;
            # defined points enter as Dirichlet boundary values. Disconnected
            # patches of undefined data do not couple to each other, so all of
            # them are solved in a single sparse system. (Solving per patch is
            # equivalent but costs a pass over the full grid for every patch.)
            mask = np.ma.getmaskarray(heights)
            values = np.ma.getdata(heights)
            nb_unknowns = np.sum(mask)

            # Unique index for each undefined point
            unknown_index = np.zeros(mask.shape, dtype=int)
            unknown_index[mask] = np.arange(nb_unknowns)

            # Assemble Laplace matrix. Each undefined point couples to its
            # nearest neighbors; for nonperiodic topographies, stencil legs
            # that cross the domain boundary must be dropped (and the diagonal
            # reduced correspondingly, yielding a natural boundary condition).
            # Legs ending on a defined point move to the right hand side.
            diagonal = np.zeros(nb_unknowns)
            rhs = np.zeros(nb_unknowns)
            rows = []
            cols = []
            for axis in range(dim):
                for shift in [1, -1]:
                    neighbor_index = np.roll(unknown_index, shift, axis)
                    neighbor_mask = np.roll(mask, shift, axis)
                    neighbor_values = np.roll(values, shift, axis)
                    valid = mask.copy()
                    if not self.is_periodic:
                        # `np.roll(a, 1)[p]` is the neighbor a[p - 1], so
                        # for shift == 1 the wrapped (invalid) entries are
                        # at the low edge of the axis, and at the high edge
                        # for shift == -1
                        edge = [slice(None)] * dim
                        edge[axis] = 0 if shift == 1 else -1
                        valid[tuple(edge)] = False
                    diagonal[unknown_index[valid]] -= 1
                    # Neighbor is undefined: off-diagonal matrix entry
                    sel = np.logical_and(valid, neighbor_mask)
                    rows += [unknown_index[sel]]
                    cols += [neighbor_index[sel]]
                    # Neighbor is defined: Dirichlet value on right hand side
                    sel = np.logical_and(valid, np.logical_not(neighbor_mask))
                    np.subtract.at(rhs, unknown_index[sel], neighbor_values[sel])

            laplace = scipy.sparse.coo_matrix(
                (np.concatenate([diagonal] + [np.ones(len(r)) for r in rows]),
                 (np.concatenate([np.arange(nb_unknowns)] + rows),
                  np.concatenate([np.arange(nb_unknowns)] + cols))),
                shape=(nb_unknowns, nb_unknowns))

            # Solve for undefined heights
            heights[mask] = scipy.sparse.linalg.spsolve(laplace.tocsr(), rhs)
        return heights


def interpolate_undefined_data(self, method='harmonic'):
    """
    Imputation of undefined data points in topography information that
    typically occurs in optical measurements.

    Parameters
    ----------
    self : SurfaceTopography.Topography or SurfaceTopography.UniformLineScan
        Input topography containing undefined data points.
    method : str
        Imputation methods. Options
           'harmonic': Interpolate with harmonic functions
        (Default: 'harmonic')
    """
    if method == 'harmonic':
        return self.interpolate_undefined_data_with_harmonic_function()
    else:
        raise ValueError(f"Unsupported imputation method '{method}'.")


UniformTopographyInterface.register_function('interpolate_undefined_data',
                                             interpolate_undefined_data)
UniformTopographyInterface.register_function('interpolate_undefined_data_with_harmonic_function',
                                             InterpolateUndefinedDataHarmonic)
