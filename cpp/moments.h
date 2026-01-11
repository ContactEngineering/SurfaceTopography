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

/* Compute moments based on linear interpolation of the line scans or
   topography. For a topography, linear interpolation means each pixel
   consists of two triangles. */

#include <cmath>

#include "eigen_helper.h"

template <int order>
class _LineScanMoment {
public:
    static double eval(double h1, double h2) {
        if (std::abs(h2 - h1) < 1e-12) {
            return 0;
        }
        // This is the generic expression, but it has numerical issues when h1 and h2 are close to each other
        return (std::pow(h2, order+1) - std::pow(h1, order+1)) / (h2 - h1);
    }
};

// Specialize for specific values to avoid numerical issues
template <>
class _LineScanMoment<1> {
public:
    static double eval(double h1, double h2) {
        return h1 + h2;
    }
};

template <>
class _LineScanMoment<2> {
public:
    static double eval(double h1, double h2) {
        return h1*h1 + h1*h2 + h2*h2;
    }
};

template <>
class _LineScanMoment<3> {
public:
    static double eval(double h1, double h2) {
        return (h1 + h2)*(h1*h1 + h2*h2);
    }
};


template <int order>
double nonuniform_moment(Eigen::Ref<Eigen::ArrayXd> topography_x, Eigen::Ref<Eigen::ArrayXd> topography_h,
                         double ref_h) {
    if (topography_x.size() != topography_h.size()) {
        throw std::runtime_error("`topography_x` and `topography_h` must have the same size");
    }

    /* The physical length of the line scan */
    const double physical_size{topography_x.tail<1>().value() - topography_x.head<1>().value()};

    /* Accumulator for moment */
    double moment{0};

    /* Compute moment */
    for (int i{0}; i < topography_x.size()-1; i++) {
        const double hi{topography_h(i) - ref_h}, hi1{topography_h(i+1) - ref_h};
        double dx = topography_x(i+1) - topography_x(i);
        moment += dx * _LineScanMoment<order>::eval(hi, hi1);
    }

    return moment / ((order+1) * physical_size);
}


template <int order>
double uniform1d_moment(Eigen::Ref<Eigen::ArrayXd> topography_h, bool periodic, double ref_h) {
    /* Accumulator for moment */
    double moment{0};
    int physical_size{0};

    /* Compute moment */
    const auto maxi{periodic ? topography_h.size() : topography_h.size()-1};
    for (int i{0}; i < maxi; i++) {
        const auto i1{i < topography_h.size()-1 ? i+1 : 0};
        const double hi{topography_h(i) - ref_h}, hi1{topography_h(i1) - ref_h};
        /* Check for NaNs and only add if there are no NaNs */
        if (hi == hi && hi1 == hi1) {
            moment += _LineScanMoment<order>::eval(hi, hi1);
            physical_size += 1;
        }
    }

    return moment / ((order+1) * physical_size);
}


template <int order>
class _TriangleMoment {
public:
    static double eval(double h1_in, double h2_in, double h3_in) {
        double h1{h1_in}, h2{h2_in}, h3{h3_in};

        /* Sort h1, h2, h3 in ascending order */
        if (h1 > h2)  std::swap(h1, h2);
        if (h2 > h3)  std::swap(h2, h3);
        if (h1 > h2)  std::swap(h1, h2);

        /* Compute moment */
        return ((std::pow(h2, order+2) - std::pow(h1, order+2)) / (h2 - h1) +
                (std::pow(h3, order+2) - std::pow(h2, order+2)) / (h3 - h2)) / (order + 2) -
               (h1*(std::pow(h2, order+1) - std::pow(h1, order+1)) / (h2 - h1) +
                h3*(std::pow(h3, order+1) - std::pow(h2, order+1)) / (h3 - h2)) / (order + 1);
    }
};

template <>
class _TriangleMoment<1> {
public:
    static double eval(double h1, double h2, double h3) {
        /* Sort h1, h2, h3 in ascending order */
        if (h1 > h2)  std::swap(h1, h2);
        if (h2 > h3)  std::swap(h2, h3);
        if (h1 > h2)  std::swap(h1, h2);

        /* Compute moment */
        return (4*h2*h2 - h1*h1 - h3*h3 - h1*h2 - h2*h3) / 6;
    }
};

template <>
class _TriangleMoment<2> {
public:
    static double eval(double h1, double h2, double h3) {
        /* Sort h1, h2, h3 in ascending order */
        if (h1 > h2)  std::swap(h1, h2);
        if (h2 > h3)  std::swap(h2, h3);
        if (h1 > h2)  std::swap(h1, h2);

        /* Compute moment */
        return (6*h2*h2*h2 - h1*h1*h1 - h3*h3*h3 - h1*h1*h2 - h1*h2*h2 - h2*h2*h3 - h2*h3*h3) / 12;
    }
};

template <>
class _TriangleMoment<3> {
public:
    static double eval(double h1, double h2, double h3) {
        /* Sort h1, h2, h3 in ascending order */
        if (h1 > h2)  std::swap(h1, h2);
        if (h2 > h3)  std::swap(h2, h3);
        if (h1 > h2)  std::swap(h1, h2);

        /* Compute moment */
        return (8*h2*h2*h2*h2 - h1*h1*h1*h1 - h3*h3*h3*h3 - h1*h1*h1*h2 - h1*h1*h2*h2 - h1*h2*h2*h2 - h2*h2*h2*h3 -
                h2*h2*h3*h3 - h2*h3*h3*h3) / 20;
    }
};


template <int order>
double uniform2d_moment(Eigen::Ref<RowMajorXXd> topography_h, bool periodic, double ref_h) {
    /* Number of grid points for looping */
    const auto nx{periodic ? topography_h.rows() : topography_h.rows()-1};
    const auto ny{periodic ? topography_h.cols() : topography_h.cols()-1};

    /* Accumulator for moment */
    double moment{0};
    int projected_area{0};

    /* Compute moment. Loop assumes column-major storage */
    for (int x{0}; x < nx; x++) {
        const auto x1{x < topography_h.rows()-1 ? x+1 : 0};
        for (int y{0}; y < ny; y++) {
            const auto y1{y < topography_h.cols()-1 ? y+1 : 0};
            const double h00{topography_h(x, y)};
            const double h10{topography_h(x1, y)};
            const double h01{topography_h(x, y1)};
            const double h11{topography_h(x1, y1)};
            /* Check for NaNs and only add triangle if there are no NaNs */
            if (h00 == h00 && h10 == h10 && h01 == h01) {
                moment += _TriangleMoment<order>::eval(h00 - ref_h, h10 - ref_h, h01 - ref_h);
                projected_area += 1;
            }
            /* Check for NaNs and only add triangle if there are no NaNs */
            if (h10 == h10 && h11 == h11 && h01 == h01) {
                moment += _TriangleMoment<order>::eval(h10 - ref_h, h11 - ref_h, h01 - ref_h);
                projected_area += 1;
            }
        }
    }

    return moment / projected_area;
}
