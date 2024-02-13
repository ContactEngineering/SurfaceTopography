/*
@file   _SurfaceTopography.cpp

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   04 Sep 2018

@brief  PyCo C++ extensions

@section LICENCE

Copyright 2015-2018 Till Junge, Lars Pastewka

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

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL PYCO_ARRAY_API
#include <numpy/arrayobject.h>

#include "autocorrelation.h"
#include "bicubic.h"
#include "patchfinder.h"

static PyMethodDef SurfaceTopography_methods[] = {
  {"assign_patch_numbers", assign_patch_numbers, METH_VARARGS},
  {"assign_segment_numbers", assign_segment_numbers, METH_VARARGS},
  {"correlation_function", correlation_function, METH_VARARGS},
  {"closest_patch_map", closest_patch_map, METH_VARARGS},
  {"distance_map", distance_map, METH_VARARGS},
  {"nonuniform_autocorrelation", nonuniform_autocorrelation, METH_VARARGS},
  {"perimeter_length", perimeter_length, METH_VARARGS},
  {"shortest_distance", shortest_distance, METH_VARARGS},
  {NULL, NULL, 0, NULL}     /* Sentinel - marks the end of this structure */
};

/*
 * Module initialization
 */

#ifndef PyMODINIT_FUNC  /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif

/*
 * Module declaration
 */

#define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
#define MOD_DEF(ob, name, methods, doc) \
    static struct PyModuleDef moduledef = { \
        PyModuleDef_HEAD_INIT, name, doc, -1, methods, }; \
    ob = PyModule_Create(&moduledef);

MOD_INIT(_SurfaceTopography)  {
  PyObject *m;

  import_array();

  MOD_DEF(m, "_SurfaceTopography", SurfaceTopography_methods,
          "C support functions for SurfaceTopography");

  if (PyType_Ready(&bicubic_type) < 0)
    return NULL;

  Py_INCREF(&bicubic_type);
  PyModule_AddObject(m, "Bicubic",
		     (PyObject *) &bicubic_type);

    return m;
}
