/*
@file   bicubic.cpp

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   31 Mar 2020

@brief  Bicubic interpolation of two-dimensional maps

@section LICENCE

Copyright 2020-2025 Lars Pastewka

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

#include <cmath>
#include <stdexcept>

#include <pybind11/eigen.h>

#include "bicubic.h"


Bicubic::Bicubic(const Eigen::Ref<const RowMajorXXd>& values,
                 std::optional<Eigen::Ref<const RowMajorXXd>> derivativex_opt,
                 std::optional<Eigen::Ref<const RowMajorXXd>> derivativey_opt)
  : n1_{static_cast<int>(values.rows())},
    n2_{static_cast<int>(values.cols())},
    interp_{!derivativex_opt && !derivativey_opt},
    values_(n1_ * n2_),
    has_derivativex_{derivativex_opt.has_value()},
    has_derivativey_{derivativey_opt.has_value()},
    derivativex_(has_derivativex_ ? n1_ * n2_ : 0),
    derivativey_(has_derivativey_ ? n1_ * n2_ : 0),
    coeff_(NPARA, n1_ * n2_)
{
  // Copy values to internal storage (row-major to linear)
  for (int i = 0; i < n1_; ++i) {
    for (int j = 0; j < n2_; ++j) {
      values_(_row_major(i, j, n1_, n2_)) = values(i, j);
    }
  }

  // Copy derivatives if provided
  if (derivativex_opt) {
    auto &derivativex = *derivativex_opt;
    if (derivativex.rows() != n1_ || derivativex.cols() != n2_) {
      throw std::invalid_argument("x-derivative must have same shape as values.");
    }
    for (int i = 0; i < n1_; ++i) {
      for (int j = 0; j < n2_; ++j) {
        derivativex_(_row_major(i, j, n1_, n2_)) = derivativex(i, j);
      }
    }
  }

  if (derivativey_opt) {
    auto &derivativey = *derivativey_opt;
    if (derivativey.rows() != n1_ || derivativey.cols() != n2_) {
      throw std::invalid_argument("y-derivative must have same shape as values.");
    }
    for (int i = 0; i < n1_; ++i) {
      for (int j = 0; j < n2_; ++j) {
        derivativey_(_row_major(i, j, n1_, n2_)) = derivativey(i, j);
      }
    }
  }

  const int box1[NCORN] = { 0,1,1,0 };
  const int box2[NCORN] = { 0,0,1,1 };

  /*
   * calculate 2-d cubic parameters within each box.
   *
   * normalised coordinates.
   *  4--<--3
   *  |     ^
   *  v     |
   *  1-->--2
   */

  /*
   * for each box, create and solve the matrix equation.
   *    / values of  \     /              \     / function and \
   *  a |  products  | * x | coefficients | = b |  derivative  |
   *    \within cubic/     \ of 2d cubic  /     \    values    /
   */

  /*
   * construct the matrix.
   * this is the same for all boxes as coordinates are normalised.
   * loop through corners.
   */

  for (int irow = 0; irow < NCORN; irow++) {
    int ci1 = box1[irow];
    int ci2 = box2[irow];
    /* loop through powers of variables. */
    for (int npow1 = 0; npow1 <= 3; npow1++) {
      for (int npow2 = 0; npow2 <= 3; npow2++) {
        int              npow1m = npow1-1;
        if (npow1m < 0)  npow1m = 0;
        int              npow2m = npow2-1;
        if (npow2m < 0)  npow2m=0;

        int icol = _row_major(npow1, npow2, 4, 4);

        /* values of products within cubic and derivatives. */
        A_(irow   , icol) = 1.0*(      pow(ci1, npow1 )      *pow(ci2, npow2 ) );
        A_(irow+4 , icol) = 1.0*(npow1*pow(ci1, npow1m)      *pow(ci2, npow2 ) );
        A_(irow+8 , icol) = 1.0*(      pow(ci1, npow1 )*npow2*pow(ci2, npow2m) );
        A_(irow+12, icol) = 1.0*(npow1*pow(ci1, npow1m)*npow2*pow(ci2, npow2m) );
      }
    }
  }

  /*
   * invert A matrix.
   */
  this->A_ = this->A_.inverse();

  /*
   * Compute all spline coefficients and store them.
   */
  for (int i1 = 0; i1 < this->n1_; i1++) {
    for (int i2 = 0; i2 < this->n2_; i2++) {
      this->coeff_.col(_row_major(i1, i2, this->n1_, this->n2_)) =
          compute_spline_coefficients(i1, i2, this->values_, this->has_derivativex_, this->derivativex_,
                                      this->has_derivativey_, this->derivativey_);
    }
  }
}


Bicubic::~Bicubic()
{
}


Eigen::Matrix<double, NPARA, 1>
Bicubic::compute_spline_coefficients(int i1, int i2, const Eigen::Ref<Eigen::ArrayXd> &values,
                                     bool has_derivativex, const Eigen::Ref<Eigen::ArrayXd> &derivativex,
                                     bool has_derivativey, const Eigen::Ref<Eigen::ArrayXd> &derivativey) {
  const int box1[NCORN] = { 0,1,1,0 };
  const int box2[NCORN] = { 0,0,1,1 };

  /*
   * construct the 16 r.h.s. vectors ( 1 for each box ).
   * loop through boxes.
   */

  Eigen::Matrix<double, NPARA, 1> B;

  for (int irow = 0; irow < NCORN; irow++) {
    int ci1 = box1[irow]+i1;
    int ci2 = box2[irow]+i2;
    /* wrap to box */
    ci1 = _wrap(ci1, this->n1_);
    ci2 = _wrap(ci2, this->n2_);
    /* values of function and derivatives at corner. */
    B(irow) = values(_row_major(ci1, ci2, this->n1_, this->n2_));
    /* interpolate derivatives */
    if (this->interp_) {
      int ci1p = ci1+1;
      int ci1m = ci1-1;
      int ci2p = ci2+1;
      int ci2m = ci2-1;
      ci1p = _wrap(ci1p, this->n1_);
      ci1m = _wrap(ci1m, this->n1_);
      ci2p = _wrap(ci2p, this->n2_);
      ci2m = _wrap(ci2m, this->n2_);
      B(irow+4) = (
        values(_row_major(ci1p, ci2, this->n1_, this->n2_)) -
        values(_row_major(ci1m, ci2, this->n1_, this->n2_))
        )/2;
      B(irow+8) = (
        values(_row_major(ci1, ci2p, this->n1_, this->n2_)) -
        values(_row_major(ci1, ci2m, this->n1_, this->n2_))
        )/2;
    }
    else {
      if (has_derivativex) {
        B(irow+4) = derivativex(_row_major(ci1, ci2, this->n1_, this->n2_));
      }
      else {
        B(irow+4) = 0.0;
      }
      if (has_derivativey) {
        B(irow+8) = derivativey(_row_major(ci1, ci2, this->n1_, this->n2_));
      }
      else {
        B(irow+8) = 0.0;
      }
    }
    B(irow+12) = 0.0;
  }

  return this->A_ * B;
}


void
Bicubic::eval(double x, double y, double &f)
{
  int xbox = static_cast<int>(floor(x));
  int ybox = static_cast<int>(floor(y));

  /*
   * find which box we're in and convert to normalised coordinates.
   */
  double dx = x - xbox;
  double dy = y - ybox;
  xbox = _wrap(xbox, this->n1_);
  ybox = _wrap(ybox, this->n2_);

  /*
   * get spline coefficients
   */
  const auto coeffi{get_spline_coefficients(xbox, ybox)};

  /*
   * compute splines
   */
  f = 0.0;
  for (int i = 3; i >= 0; i--) {
    double sf = 0.0;
    for (int j = 3; j >= 0; j--) {
      sf = sf*dy + coeffi(_row_major(i, j, 4, 4));
    }
    f = f*dx + sf;
  }
}


void
Bicubic::eval(double x, double y, double &f, double &dfdx, double &dfdy)
{
  int xbox = static_cast<int>(floor(x));
  int ybox = static_cast<int>(floor(y));

  /*
   * find which box we're in and convert to normalised coordinates.
   */
  double dx = x - xbox;
  double dy = y - ybox;
  xbox = _wrap(xbox, this->n1_);
  ybox = _wrap(ybox, this->n2_);

  /*
   * get spline coefficients
   */
  const auto coeffi{get_spline_coefficients(xbox, ybox)};

  /*
   * compute splines
   */
  f = 0.0;
  dfdx = 0.0;
  dfdy = 0.0;
  for (int i = 3; i >= 0; i--) {
    double sf   = 0.0;
    double sfdy = 0.0;
    for (int j = 3; j >= 0; j--) {
      double coefij{coeffi(_row_major(i, j, 4, 4))};
      sf = sf*dy + coefij;
      if (j > 0)  sfdy = sfdy*dy + j*coefij;
    }
    f = f*dx + sf;
    if (i > 0)  dfdx = dfdx*dx + i*sf;
    dfdy = dfdy*dx + sfdy;
  }
}


void
Bicubic::eval(double x, double y, double &f,
              double &dfdx, double &dfdy,
              double &d2fdxdx, double &d2fdydy, double &d2fdxdy)

{
  int xbox = static_cast<int>(floor(x));
  int ybox = static_cast<int>(floor(y));

  /*
   * find which box we're in and convert to normalised coordinates.
   */
  double dx = x - xbox;
  double dy = y - ybox;
  xbox = _wrap(xbox, this->n1_);
  ybox = _wrap(ybox, this->n2_);

  /*
   * get spline coefficients
   */
  const auto coeffi{get_spline_coefficients(xbox, ybox)};

  /*
   * compute splines
   */
  f = 0.0;
  dfdx = 0.0;
  dfdy = 0.0;
  d2fdxdx = 0.0;
  d2fdydy = 0.0;
  d2fdxdy = 0.0;
  for (int i = 3; i >= 0; i--) {
    double sf{0.0};
    double sfdy{0.0};
    double s2fdydy{0.0};
    for (int j = 3; j >= 0; j--) {
      double coefij = coeffi(_row_major(i, j, 4, 4));
      sf = sf*dy + coefij;
      if (j > 0)  sfdy = sfdy*dy + j*coefij;
      if (j > 1)  s2fdydy = s2fdydy*dy + j*(j-1)*coefij;
    }
    f = f*dx + sf;
    if (i > 0)  dfdx = dfdx*dx + i*sf;
    if (i > 1)  d2fdxdx = d2fdxdx*dx + i*(i-1)*sf;
    dfdy = dfdy*dx + sfdy;
    if (i > 0)  d2fdxdy = d2fdxdy*dx + i*sfdy;
  }
}


py::object
Bicubic::call(py::object py_x, py::object py_y, int derivative)
{
  if (derivative < 0 || derivative > 2) {
    throw std::invalid_argument("'derivative' keyword argument must be 0, 1 or 2.");
  }

  // Check if inputs are scalars
  if (py::isinstance<py::float_>(py_x) || py::isinstance<py::int_>(py_x)) {
    if (py::isinstance<py::float_>(py_y) || py::isinstance<py::int_>(py_y)) {
      // Scalar inputs
      double x = py_x.cast<double>();
      double y = py_y.cast<double>();
      double v, dx_out, dy_out;
      eval(x, y, v, dx_out, dy_out);
      return py::float_(v);
    }
  }

  // Array inputs - convert to numpy arrays
  py::array_t<double> x_arr = py::array_t<double>::ensure(py_x);
  py::array_t<double> y_arr = py::array_t<double>::ensure(py_y);

  if (!x_arr || !y_arr) {
    throw std::invalid_argument("Could not convert inputs to arrays.");
  }

  // Check dimensions match
  if (x_arr.ndim() != y_arr.ndim()) {
    throw std::invalid_argument("x- and y-components need to have identical number of dimensions.");
  }

  // Check shapes match
  for (py::ssize_t i = 0; i < x_arr.ndim(); ++i) {
    if (x_arr.shape(i) != y_arr.shape(i)) {
      throw std::invalid_argument("x- and y-components vectors need to have the same length.");
    }
  }

  // Get total number of elements
  py::ssize_t n = x_arr.size();

  // Get pointers
  auto x_buf = x_arr.request();
  auto y_buf = y_arr.request();
  double *x = static_cast<double*>(x_buf.ptr);
  double *y = static_cast<double*>(y_buf.ptr);

  // Create output arrays with same shape
  std::vector<py::ssize_t> shape(x_arr.shape(), x_arr.shape() + x_arr.ndim());
  py::array_t<double> v_arr(shape);
  double *v = static_cast<double*>(v_arr.request().ptr);

  if (derivative > 0) {
    py::array_t<double> dx_arr(shape);
    py::array_t<double> dy_arr(shape);
    double *dx_out = static_cast<double*>(dx_arr.request().ptr);
    double *dy_out = static_cast<double*>(dy_arr.request().ptr);

    if (derivative > 1) {
      py::array_t<double> d2x_arr(shape);
      py::array_t<double> d2y_arr(shape);
      py::array_t<double> d2xy_arr(shape);
      double *d2x = static_cast<double*>(d2x_arr.request().ptr);
      double *d2y = static_cast<double*>(d2y_arr.request().ptr);
      double *d2xy = static_cast<double*>(d2xy_arr.request().ptr);

      for (py::ssize_t i = 0; i < n; ++i) {
        eval(x[i], y[i], v[i], dx_out[i], dy_out[i], d2x[i], d2y[i], d2xy[i]);
      }
      return py::make_tuple(v_arr, dx_arr, dy_arr, d2x_arr, d2y_arr, d2xy_arr);
    } else {
      for (py::ssize_t i = 0; i < n; ++i) {
        eval(x[i], y[i], v[i], dx_out[i], dy_out[i]);
      }
      return py::make_tuple(v_arr, dx_arr, dy_arr);
    }
  } else {
    for (py::ssize_t i = 0; i < n; ++i) {
      eval(x[i], y[i], v[i]);
    }
    return v_arr;
  }
}
