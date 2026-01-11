/*
Copyright 2024-2025 Lars Pastewka

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
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

// Former C module components
#include "autocorrelation.h"
#include "bicubic.h"
#include "patchfinder.h"

// Former C++ module components
#include "bearing_area.h"
#include "moments.h"


PYBIND11_MODULE(_SurfaceTopography, mod) {
    mod.doc() = "C++ support functions for SurfaceTopography";

    // === Autocorrelation (from former C module) ===
    mod.def("nonuniform_autocorrelation", &nonuniform_autocorrelation,
            "Height-difference autocorrelation of nonuniform line scans",
            py::arg("x"), py::arg("h"), py::arg("physical_size"),
            py::arg("distances") = std::nullopt);

    // === Patch/geometry analysis (from former C module) ===
    mod.def("assign_patch_numbers", &assign_patch_numbers,
            "Assign unique numbers to connected patches",
            py::arg("map"), py::arg("periodic"), py::arg("stencil") = std::nullopt);
    mod.def("assign_segment_numbers", &assign_segment_numbers,
            "Assign unique numbers to connected 1D segments",
            py::arg("map"));
    mod.def("distance_map", &distance_map,
            "Compute distance from each point to nearest marked point",
            py::arg("map"));
    mod.def("closest_patch_map", &closest_patch_map,
            "Compute the tag of the closest patch for each point",
            py::arg("map"));
    mod.def("shortest_distance", &shortest_distance,
            "Compute shortest distance between patches",
            py::arg("fromc"), py::arg("fromp"), py::arg("to"), py::arg("maxd") = -1);
    mod.def("correlation_function", &correlation_function,
            "Compute real-space correlation function between two maps",
            py::arg("map1"), py::arg("map2"), py::arg("max_dist"));
    mod.def("perimeter_length", &perimeter_length,
            "Compute total perimeter length of marked regions",
            py::arg("map"));

    // === Bicubic interpolation (from former C module) ===
    py::class_<Bicubic>(mod, "Bicubic")
        .def(py::init<const Eigen::Ref<const RowMajorXXd>&,
                      std::optional<Eigen::Ref<const RowMajorXXd>>,
                      std::optional<Eigen::Ref<const RowMajorXXd>>>(),
             "Bicubic interpolation of two-dimensional maps",
             py::arg("values"),
             py::arg("derivativex") = std::nullopt,
             py::arg("derivativey") = std::nullopt)
        .def("__call__", &Bicubic::call,
             py::arg("x"), py::arg("y"), py::arg("derivative") = 0);

    // === Bearing area (from former C++ module) ===
    mod.def("nonuniform_bearing_area", &nonuniform_bearing_area, "Bearing area of a nonuniform line scan");
    mod.def("uniform1d_bearing_area", &uniform1d_bearing_area, "Bearing area of a uniform line scan");
    mod.def("uniform2d_bearing_area", &uniform2d_bearing_area, "Bearing area of a topography map");

    // === Moments (from former C++ module) ===
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

    mod.def("nonuniform_moment3", &nonuniform_moment<3>, "Third moment of a nonuniform line scan",
            py::arg("x"), py::arg("h"), py::arg("ref_h") = 0.0);
    mod.def("uniform1d_moment3", &uniform1d_moment<3>, "Third moment of a uniform line scan",
            py::arg("h"), py::arg("periodic"), py::arg("ref_h") = 0.0);
    mod.def("uniform2d_moment3", &uniform2d_moment<3>, "Third moment of a topography map",
            py::arg("h"), py::arg("periodic"), py::arg("ref_h") = 0.0);

    mod.def("nonuniform_moment4", &nonuniform_moment<4>, "Fourth moment of a nonuniform line scan",
            py::arg("x"), py::arg("h"), py::arg("ref_h") = 0.0);
    mod.def("uniform1d_moment4", &uniform1d_moment<4>, "Fourth moment of a uniform line scan",
            py::arg("h"), py::arg("periodic"), py::arg("ref_h") = 0.0);
    mod.def("uniform2d_moment4", &uniform2d_moment<4>, "Fourth moment of a topography map",
            py::arg("h"), py::arg("periodic"), py::arg("ref_h") = 0.0);
}
