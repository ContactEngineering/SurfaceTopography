/*
@file   bicubic.cpp

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

#include <cmath>
#include <stdexcept>
#include <iostream>

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL PYCO_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include "bicubic.h"


/*
 * values are supposed to be of size [0:nx][0:ny] and stored in row-major order
 */
Bicubic::Bicubic(int n1, int n2, double *values, double *derivativex, double *derivativey, bool interp, bool lowmem)
  : n1_{n1}, n2_{n2}, interp_{interp}, lowmem_{lowmem}, values_{values, n1*n2},
    has_derivativex_{derivativex != nullptr}, has_derivativey_{derivativey != nullptr},
    derivativex_{derivativex, n1*n2}, derivativey_{derivativey, n1*n2},
    coeff_(NPARA, lowmem ? 0 : n1*n2), coeff_lowmem_(lowmem ? NPARA : 0)
{
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

  /* --- */

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
   * if low mem is not requested, we compute all spline coefficients here and store them.
   */

  if (this->coeff_.size()) {
    for (int i1 = 0; i1 < this->n1_; i1++) {
      for (int i2 = 0; i2 < this->n2_; i2++) {
        this->coeff_.col(_row_major(i1, i2, this->n1_, this->n2_)) =
            compute_spline_coefficients(i1, i2, this->values_, this->has_derivativex_, this->derivativex_,
                                        this->has_derivativey_, this->derivativey_);
      }
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


/* Allocate new instance */

static PyObject *
bicubic_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  bicubic_t *self;

  self = (bicubic_t *)type->tp_alloc(type, 0);

  self->map_ = NULL;

  return (PyObject *) self;
}


/* Release allocated memory */

static void
bicubic_dealloc(bicubic_t *self)
{
  if (self->map_)
    delete self->map_;
  self->map_ = NULL;

  Py_TYPE(self)->tp_free((PyObject*) self);
}


/* Initialize instance */

static int
bicubic_init(bicubic_t *self, PyObject *args, PyObject *kwargs)
{
  PyObject *py_values_in, *py_derivativex_in{NULL}, *py_derivativey_in{NULL};

  if (!PyArg_ParseTuple(args, "O|OO", &py_values_in, &py_derivativex_in, &py_derivativey_in))
    return -1;

  PyObject *py_values, *py_derivativex{NULL}, *py_derivativey{NULL};
  npy_intp nx, ny;

  py_values = PyArray_FROMANY(py_values_in, NPY_DOUBLE, 2, 2, NPY_ARRAY_C_CONTIGUOUS);
  if (!py_values)
    return -1;
  nx = PyArray_DIM((PyArrayObject *) py_values, 0);
  ny = PyArray_DIM((PyArrayObject *) py_values, 1);

  if (py_derivativex_in) {
    py_derivativex = PyArray_FROMANY(py_derivativex_in, NPY_DOUBLE, 2, 2, NPY_ARRAY_C_CONTIGUOUS);
    if (!py_derivativex)
      return -1;
    if (PyArray_DIM((PyArrayObject *) py_derivativex, 0) != nx || PyArray_DIM((PyArrayObject *) py_derivativex, 1) != ny) {
	  PyErr_SetString(PyExc_ValueError, "x-derivative must have same shape as values.");
	  return -1;
    }
  }

  if (py_derivativey_in) {
    py_derivativey = PyArray_FROMANY(py_derivativey_in, NPY_DOUBLE, 2, 2, NPY_ARRAY_C_CONTIGUOUS);
    if (!py_derivativey)
      return -1;
    if (PyArray_DIM((PyArrayObject *) py_derivativey, 0) != nx || PyArray_DIM((PyArrayObject *) py_derivativey, 1) != ny) {
	  PyErr_SetString(PyExc_ValueError, "y-derivative must have same shape as values.");
	  return -1;
    }
  }

  double *derivativex{NULL}, *derivativey{NULL};
  if (py_derivativex) {
    derivativex = static_cast<double*>(PyArray_DATA((PyArrayObject *) py_derivativex));
  }
  if (py_derivativey) {
    derivativey = static_cast<double*>(PyArray_DATA((PyArrayObject *) py_derivativey));
  }

  if (derivativex || derivativey) {
    self->map_ = new Bicubic(nx, ny, static_cast<double*>(PyArray_DATA((PyArrayObject *) py_values)), derivativex, derivativey, false,
                             false);
  }
  else {
    self->map_ = new Bicubic(nx, ny, static_cast<double*>(PyArray_DATA((PyArrayObject *) py_values)), NULL, NULL, true, false);
  }
  Py_DECREF(py_values);
  if (py_derivativex)  Py_DECREF(py_derivativex);
  if (py_derivativey)  Py_DECREF(py_derivativey);

  return 0;
}


/* Call object */

static PyObject *
bicubic_call(bicubic_t *self, PyObject *args, PyObject *kwargs)
{
  static char *kwlist[]{"x", "y", "derivative", NULL};

  PyObject *py_x, *py_y;
  int derivative = 0;

  /* We support passing coordinates (x, y), numpy arrays (x, y)
     and numpy arrays r */

  py_x = NULL;
  py_y = NULL;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|Oi", kwlist, &py_x, &py_y, &derivative))
    return NULL;

  if (derivative < 0 || derivative > 2) {
    PyErr_SetString(PyExc_ValueError, "'derivative' keyword argument must be 0, 1 or 2.");
    return NULL;
  }

  if (!py_y) {
    /* This should a single numpy array r */

    PyObject *py_r;
    py_r = PyArray_FROMANY(py_x, NPY_DOUBLE, 2, 2, 0);
    if (!py_r)
      return NULL;

    if (PyArray_DIM((PyArrayObject *) py_r, 1) != 2) {
      PyErr_SetString(PyExc_ValueError, "Map index needs to have x- and y-component only.");
      return NULL;
    }

    npy_intp n = PyArray_DIM((PyArrayObject *) py_r, 0);
    double *r = (double *) PyArray_DATA((PyArrayObject *) py_r);

    PyObject *py_v = PyArray_SimpleNew(1, &n, NPY_DOUBLE);
    double *v = (double *) PyArray_DATA((PyArrayObject *) py_v);

    for (int i = 0; i < n; i++) {
      self->map_->eval(r[2*i], r[2*i+1], v[i]);
    }

    Py_DECREF(py_r);

    return py_v;
  }
  else if ((PyFloat_Check(py_x) || PyLong_Check(py_x)) && (PyFloat_Check(py_y) || PyLong_Check(py_y))) {
    /* x and y are specified separately, and are scalars */

    double v, dx, dy;
    self->map_->eval(PyFloat_AsDouble(py_x), PyFloat_AsDouble(py_y), v, dx, dy);
    return PyFloat_FromDouble(v);
  }
  else {
    /* x and y are specified separately */
    PyObject *py_xd, *py_yd;
    py_xd = PyArray_FROMANY(py_x, NPY_DOUBLE, 1, 3, 0);
    if (!py_xd)
      return NULL;
    py_yd = PyArray_FROMANY(py_y, NPY_DOUBLE, 1, 3, 0);
    if (!py_yd)
      return NULL;

    /* Check that x and y have the same number of dimensions */
    if (PyArray_NDIM((PyArrayObject *) py_xd) != PyArray_NDIM((PyArrayObject *) py_yd)) {
      PyErr_SetString(PyExc_ValueError, "x- and y-components need to have identical number of dimensions.");
      return NULL;
    }

    /* Check that x and y have the same length in each dimension */
    int ndims = PyArray_NDIM((PyArrayObject *) py_xd);
    npy_intp *dims = PyArray_DIMS((PyArrayObject *) py_xd);
    npy_intp n = 1;
    for (int i = 0; i < ndims; i++) {
      npy_intp d = PyArray_DIM((PyArrayObject *) py_yd, i);

      if (dims[i] != d) {
	    PyErr_SetString(PyExc_ValueError, "x- and y-components vectors need to have the same length.");
	    return NULL;
      }

      n *= d;
    }

    double *x{static_cast<double *>(PyArray_DATA((PyArrayObject *) py_xd))};
    double *y{static_cast<double *>(PyArray_DATA((PyArrayObject *) py_yd))};

    PyObject *py_v{PyArray_SimpleNew(ndims, dims, NPY_DOUBLE)};
    double *v{static_cast<double *>(PyArray_DATA((PyArrayObject *) py_v))};

    PyObject *py_return_value;

    if (derivative > 0) {
      PyObject *py_dx{PyArray_SimpleNew(ndims, dims, NPY_DOUBLE)};
      PyObject *py_dy{PyArray_SimpleNew(ndims, dims, NPY_DOUBLE)};
      double *dx{static_cast<double *>(PyArray_DATA((PyArrayObject *) py_dx))};
      double *dy{static_cast<double *>(PyArray_DATA((PyArrayObject *) py_dy))};
      if (derivative > 1) {
        PyObject *py_d2x{PyArray_SimpleNew(ndims, dims, NPY_DOUBLE)};
        PyObject *py_d2y{PyArray_SimpleNew(ndims, dims, NPY_DOUBLE)};
        PyObject *py_d2xy{PyArray_SimpleNew(ndims, dims, NPY_DOUBLE)};
        double *d2x{static_cast<double *>(PyArray_DATA((PyArrayObject *) py_d2x))};
        double *d2y{static_cast<double *>(PyArray_DATA((PyArrayObject *) py_d2y))};
        double *d2xy{static_cast<double *>(PyArray_DATA((PyArrayObject *) py_d2xy))};
        for (int i = 0; i < n; i++) {
          self->map_->eval(x[i], y[i], v[i], dx[i], dy[i], d2x[i], d2y[i], d2xy[i]);
        }
        py_return_value = PyTuple_Pack(6, py_v, py_dx, py_dy, py_d2x, py_d2y, py_d2xy);
        Py_DECREF(py_d2x);
        Py_DECREF(py_d2y);
        Py_DECREF(py_d2xy);
      }
      else {
        for (int i = 0; i < n; i++) {
          self->map_->eval(x[i], y[i], v[i], dx[i], dy[i]);
        }
        py_return_value = PyTuple_Pack(3, py_v, py_dx, py_dy);
      }
      Py_DECREF(py_v);
      Py_DECREF(py_dx);
      Py_DECREF(py_dy);
    }
    else {
      for (int i = 0; i < n; i++) {
        self->map_->eval(x[i], y[i], v[i]);
      }
      py_return_value = py_v;
    }

    Py_DECREF(py_xd);
    Py_DECREF(py_yd);

    return py_return_value;
  }
}


/* Class declaration */

PyTypeObject bicubic_type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "bicubic.Bicubic",                                /*tp_name*/
    sizeof(bicubic_t),                                /*tp_basicsize*/
    0,                                                /*tp_itemsize*/
    (destructor)bicubic_dealloc,                      /*tp_dealloc*/
    0,                                                /*tp_print*/
    0,                                                /*tp_getattr*/
    0,                                                /*tp_setattr*/
    0,                                                /*tp_compare*/
    0,                                                /*tp_repr*/
    0,                                                /*tp_as_number*/
    0,                                                /*tp_as_sequence*/
    0,                                                /*tp_as_mapping*/
    0,                                                /*tp_hash */
    (ternaryfunc)bicubic_call,                        /*tp_call*/
    0,                                                /*tp_str*/
    0,                                                /*tp_getattro*/
    0,                                                /*tp_setattro*/
    0,                                                /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,         /*tp_flags*/
    "Bicubic interpolation of two-dimensional maps",  /* tp_doc */
    0,                                                /* tp_traverse */
    0,                                                /* tp_clear */
    0,                                                /* tp_richcompare */
    0,                                                /* tp_weaklistoffset */
    0,                                                /* tp_iter */
    0,                                                /* tp_iternext */
    0,                                                /* tp_methods */
    0,                                                /* tp_members */
    0,                                                /* tp_getset */
    0,                                                /* tp_base */
    0,                                                /* tp_dict */
    0,                                                /* tp_descr_get */
    0,                                                /* tp_descr_set */
    0,                                                /* tp_dictoffset */
    (initproc)bicubic_init,                           /* tp_init */
    0,                                                /* tp_alloc */
    bicubic_new,                                      /* tp_new */
};
