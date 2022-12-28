/*
@file   bicubic.h

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   31 Mar 2020

@brief  Bicubic interpolation of two-dimensional maps

@section LICENCE

Copyright 2020 Lars Pastewka

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
*/

#ifndef __BICUBIC_H
#define __BICUBIC_H

#include <cstddef>
#include <vector>

#include <Eigen/Dense>

#include <Python.h>

constexpr int NPARA{4*4};   // 4^dim
constexpr int NCORN{4};

inline int
_wrap(int x, int nx) { while (x >= nx) x -= nx; while (x < 0) x += nx; return x; }

inline ptrdiff_t
_row_major(int x, int y, int nx, int ny) { return y + static_cast<ptrdiff_t>(ny)*x; }

inline ptrdiff_t
_row_major(int x, int y, int z, int nx, int ny, int nz) {
  return z + static_cast<ptrdiff_t>(nz)*(y + static_cast<ptrdiff_t>(ny)*x);
}


class Bicubic {
 public:
  /*
   * Constructor
   *   n1, n2: number of grid points
   *   values: map containing the values to be interpolated in row major storage
   *           (Those values are only used in the constructor if lowmem is false,
   *            but are required during evaluation if lowmem is true. This means the
   *            pointer must remain valid throughout the lifetime of the Bicubic class
   *            if lowmem is set to true.)
   *   derivativex, derivativey: map containing the derivative values (can be NULL)
   *   interp: interpolate derivative using a finite-differences scheme if true
   *   lowmem: don't store spline coefficients but recompute on evaluation if true
   *           (This is slow but saves memory.)
   */
  Bicubic(int n1, int n2, double *values, double *derivativex, double *derivativey, bool interp, bool lowmem);
  ~Bicubic();

  void eval(double x, double y, double &f);
  void eval(double x, double y, double &f, double &dfdx, double &dfdy);
  void eval(double x, double y, double &f, double &dfdx, double &dfdy,
            double &d2fdxdx, double &d2fdydy, double &d2fdxdy);

 protected:
  /* table dimensions */
  int n1_, n2_;

  /* interpolate derivatives */
  bool interp_;

  /* use slow, but low memory implementation */
  bool lowmem_;

  /* values */
  const Eigen::Map<Eigen::ArrayXd> values_;

  /* derivatives */
  bool has_derivativex_, has_derivativey_;
  const Eigen::Map<Eigen::ArrayXd> derivativex_, derivativey_;

  /* spline coefficients */
  Eigen::Array<double, NPARA, Eigen::Dynamic> coeff_;

  /* spline coefficients if lowmem is true */
  Eigen::ArrayXd coeff_lowmem_;

  /* lhs matrix */
  Eigen::Matrix<double, NPARA, NPARA> A_;

  Eigen::Array<double, NPARA, 1>
  get_spline_coefficients(int i1, int i2) {
    if (this->coeff_.size() > 0) {
      return this->coeff_.col(_row_major(i1, i2, this->n1_, this->n2_));
    }
    else {
      return compute_spline_coefficients(i1, i2, this->values_, this->has_derivativex_, this->derivativex_,
                                         this->has_derivativey_, this->derivativey_);
    }
  }

  Eigen::Matrix<double, NPARA, 1>
  compute_spline_coefficients(int, int, const Eigen::Ref<Eigen::ArrayXd> &,
                              bool, const Eigen::Ref<Eigen::ArrayXd> &,
                              bool, const Eigen::Ref<Eigen::ArrayXd> &);
};

typedef struct {
  PyObject_HEAD

  Bicubic *map_;

} bicubic_t;

extern PyTypeObject bicubic_type;

#endif
