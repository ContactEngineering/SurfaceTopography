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

#include <algorithm>

#include "bearing_area.h"


Eigen::ArrayXd nonuniform_bearing_area(Eigen::Ref<Eigen::ArrayXd> topography_x, Eigen::Ref<Eigen::ArrayXd> topography_h,
                                       Eigen::Ref<Eigen::ArrayXd> heights) {
    if (topography_x.size() != topography_h.size()) {
        throw std::runtime_error("`topography_x` and `topography_h` must have the same size");
    }

    /* The physical length of the line scan */
    const double physical_size{topography_x.tail<1>().value() - topography_x.head<1>().value()};

    /* Bearing area values for each input height */
    Eigen::ArrayXd fractional_bearing_areas(heights.size());

    /* Compute bearing areas */
    for (int j{0}; j < heights.size(); j++) {
        double bearing_area{0};
        for (int i{0}; i < topography_x.size()-1; i++) {
            const double hi{topography_h(i)}, hi1{topography_h(i+1)};
            double dx = topography_x(i+1) - topography_x(i);
            if (heights(j) < hi && heights(j) < hi1) {
                bearing_area += dx;
            } else if (!(heights(j) > hi && heights(j) > hi1)) {
                auto [minh, maxh] = std::minmax(hi, hi1);
                bearing_area += dx * (maxh - heights(j)) / (maxh - minh);
            }
        }
        fractional_bearing_areas(j) = bearing_area / physical_size;
    }

    return fractional_bearing_areas;
}


Eigen::ArrayXd uniform1d_bearing_area(double dx, Eigen::Ref<Eigen::ArrayXd> topography_h, bool periodic,
                                      Eigen::Ref<Eigen::ArrayXd> heights) {
    /* The physical length of the line scan */
    const double physical_size{periodic ? dx * topography_h.size() : dx * (topography_h.size()-1)};

    /* Bearing area values for each input height */
    Eigen::ArrayXd fractional_bearing_areas(heights.size());

    /* Compute bearing areas */
    const auto maxi{periodic ? topography_h.size() : topography_h.size()-1};
    for (int j{0}; j < heights.size(); j++) {
        double bearing_area{0};
        for (int i{0}; i < maxi; i++) {
            const auto i1{i < topography_h.size()-1 ? i+1 : 0};
            const double hi{topography_h(i)}, hi1{topography_h(i1)};
            if (heights(j) < hi && heights(j) < hi1) {
                bearing_area += dx;
            } else if (!(heights(j) > hi && heights(j) > hi1)) {
                auto [minh, maxh] = std::minmax(hi, hi1);
                bearing_area += dx * (maxh - heights(j)) / (maxh - minh);
            }
        }
        fractional_bearing_areas(j) = bearing_area / physical_size;
    }

    return fractional_bearing_areas;
}


double _triangle(double h1_in, double h2_in, double h3_in, double h) {
    double h1{h1_in}, h2{h2_in}, h3{h3_in};
    double bearing_area{0};

    /* Sort h1, h2, h3 in ascending order */
    if (h1 > h2)  std::swap(h1, h2);
    if (h2 > h3)  std::swap(h2, h3);
    if (h1 > h2)  std::swap(h1, h2);

    /* Compute bearing area */
    if (h <= h1) {
        bearing_area += 1.0;
    } else if (h1 < h && h <= h2) {
        bearing_area += 1 - (h - h1) * (h - h1) / ((h2 - h1) * (h3 - h1));
    } else if (h2 < h && h <= h3) {
        bearing_area += (h3 - h) * (h3 - h) / ((h3 - h2) * (h3 - h1));
    }
    return bearing_area;
}


Eigen::ArrayXd uniform2d_bearing_area(double dx, double dy, Eigen::Ref<RowMajorXXd> topography_h, bool periodic,
                                      Eigen::Ref<Eigen::ArrayXd> heights) {
    /* Number of grid points for looping */
    const auto nx{periodic ? topography_h.rows() : topography_h.rows()-1};
    const auto ny{periodic ? topography_h.cols() : topography_h.cols()-1};

    /* The physical length of the line scan */
    const double triangle_area{dx * dy / 2};
    const double projected_area{dx * nx * dy * ny};

    /* Bearing area values for each input height */
    Eigen::ArrayXd fractional_bearing_areas(heights.size());

    /* Compute bearing areas */
    for (int j{0}; j < heights.size(); j++) {
        double bearing_area{0};
        /* This is assuming column-major storage */
        for (int x{0}; x < nx; x++) {
            const auto x1{x < topography_h.rows()-1 ? x+1 : 0};
            for (int y{0}; y < ny; y++) {
                const auto y1{y < topography_h.cols()-1 ? y+1 : 0};
                bearing_area += triangle_area * (
                    _triangle(topography_h(x, y), topography_h(x1, y), topography_h(x, y1), heights(j)) +
                    _triangle(topography_h(x1, y), topography_h(x1, y1), topography_h(x, y1), heights(j)));
            }
        }
        fractional_bearing_areas(j) = bearing_area / projected_area;
    }

    return fractional_bearing_areas;
}
