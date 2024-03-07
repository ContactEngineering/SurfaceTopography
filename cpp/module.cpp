/*
Copyright 2024 Lars Pastewka

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

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

#include "bearing_area.h"
#include "moments.h"


PYBIND11_MODULE(_SurfaceTopographyPP, mod) {
    mod.doc() = "C++ support functions for SurfaceTopography";

    mod.def("nonuniform_bearing_area", &nonuniform_bearing_area, "Bearing area of a nonuniform line scan");
    mod.def("uniform1d_bearing_area", &uniform1d_bearing_area, "Bearing area of a uniform line scan");
    mod.def("uniform2d_bearing_area", &uniform2d_bearing_area, "Bearing area of a topography map");

    mod.def("nonuniform_mean", &nonuniform_moment<1>, "Mean of a nonuniform line scan",
            py::arg("x"), py::arg("h"), py::arg("ref_h") = 0.0);
    mod.def("uniform1d_mean", &uniform1d_moment<1>, "Mean of a uniform line scan",
            py::arg("h"), py::arg("periodic"), py::arg("ref_h") = 0.0);
    mod.def("uniform2d_mean", &uniform2d_moment<1>, "Mean of a topography map",
            py::arg("h"), py::arg("periodic"), py::arg("ref_h") = 0.0);

    mod.def("nonuniform_variance", &nonuniform_moment<2>, "Second moment of a nonuniform line scan",
            py::arg("x"), py::arg("h"), py::arg("ref_h") = 0.0);
    mod.def("uniform1d_variance", &uniform1d_moment<2>, "Second moment of a uniform line scan",
            py::arg("h"), py::arg("periodic"), py::arg("ref_h") = 0.0);
    mod.def("uniform2d_variance", &uniform2d_moment<2>, "Second moment of a topography map",
            py::arg("h"), py::arg("periodic"), py::arg("ref_h") = 0.0);

    mod.def("nonuniform_moment3", &nonuniform_moment<3>, "Second moment of a nonuniform line scan",
            py::arg("x"), py::arg("h"), py::arg("ref_h") = 0.0);
    mod.def("uniform1d_moment3", &uniform1d_moment<3>, "Second moment of a uniform line scan",
            py::arg("h"), py::arg("periodic"), py::arg("ref_h") = 0.0);
    mod.def("uniform2d_moment3", &uniform2d_moment<3>, "Second moment of a topography map",
            py::arg("h"), py::arg("periodic"), py::arg("ref_h") = 0.0);

    mod.def("nonuniform_moment4", &nonuniform_moment<4>, "Second moment of a nonuniform line scan",
            py::arg("x"), py::arg("h"), py::arg("ref_h") = 0.0);
    mod.def("uniform1d_moment4", &uniform1d_moment<4>, "Second moment of a uniform line scan",
            py::arg("h"), py::arg("periodic"), py::arg("ref_h") = 0.0);
    mod.def("uniform2d_moment4", &uniform2d_moment<4>, "Second moment of a topography map",
            py::arg("h"), py::arg("periodic"), py::arg("ref_h") = 0.0);
}
