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
#include <iostream>

#include "bearing_area.h"


Eigen::ArrayXd nonuniform_bearing_area(Eigen::Ref<Eigen::ArrayXd> x, Eigen::Ref<Eigen::ArrayXd> h,
                                       Eigen::Ref<ArrayXl> el_sort_by_max, Eigen::Ref<Eigen::ArrayXd> heights) {
    if (x.size() != h.size()) {
        throw std::runtime_error("`x` and `h` must have the same size");
    }
    if (el_sort_by_max.size() != h.size()-1) {
        throw std::runtime_error("`el_sort_by_max` must have the same size one less than `h`");
    }

    /* The physical length of the line scan */
    const double physical_size{x.tail<1>().value() - x.head<1>().value()};

    /* Bearing area values for each input height */
    Eigen::ArrayXd fractional_bearing_areas(heights.size());

    /* Compute bearing areas */
    for (int j{0}; j < heights.size(); j++) {
        const double height{heights(j)};

        /* Find elements that cover the reference height */
        const auto lb{std::lower_bound(
            el_sort_by_max.begin(), el_sort_by_max.end(), height,
            [&h](const long& i, double value) {
                return std::max(h(i), h(i+1)) < value;
            })};

        /* Cumulative width for all elements completely below the input height */
        double bearing_area{0};

        for (auto k{lb}; k != el_sort_by_max.end(); k++) {
            const auto i{*k};
            const double hi{h(i)}, hi1{h(i+1)};
            const double dx{x(i+1) - x(i)};
            if (height < hi && height < hi1) {
                bearing_area += dx;
            } else if (!(height > hi && height > hi1)) {
                auto [minh, maxh] = std::minmax(hi, hi1);
                bearing_area += dx * (maxh - height) / (maxh - minh);
            }
        }
        fractional_bearing_areas(j) = bearing_area / physical_size;
    }

    return fractional_bearing_areas;
}


Eigen::ArrayXd uniform1d_bearing_area(Eigen::Ref<Eigen::ArrayXd> topography_h, bool periodic,
                                      Eigen::Ref<Eigen::ArrayXd> heights) {
    /* Bearing area values for each input height */
    Eigen::ArrayXd fractional_bearing_areas(heights.size());

    /* Compute bearing areas */
    const auto maxi{periodic ? topography_h.size() : topography_h.size()-1};
    for (int j{0}; j < heights.size(); j++) {
        double bearing_area{0};
        int physical_size{0};
        for (int i{0}; i < maxi; i++) {
            const auto i1{i < topography_h.size()-1 ? i+1 : 0};
            const double hi{topography_h(i)}, hi1{topography_h(i1)};
            /* Check for NaNs and only add if there are no NaNs */
            if (hi == hi && hi1 == hi1) {
                if (heights(j) < hi && heights(j) < hi1) {
                    bearing_area += 1;
                } else if (!(heights(j) > hi && heights(j) > hi1)) {
                    auto [minh, maxh] = std::minmax(hi, hi1);
                    bearing_area += (maxh - heights(j)) / (maxh - minh);
                }
                physical_size += 1;
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


Eigen::ArrayXd uniform2d_bearing_area(Eigen::Ref<RowMajorXXd> topography_h, bool periodic,
                                      Eigen::Ref<Eigen::ArrayXd> heights) {
    /* Number of grid points for looping */
    const auto nx{periodic ? topography_h.rows() : topography_h.rows()-1};
    const auto ny{periodic ? topography_h.cols() : topography_h.cols()-1};

    /* Bearing area values for each input height */
    Eigen::ArrayXd fractional_bearing_areas(heights.size());

    /* Compute bearing areas */
    for (int j{0}; j < heights.size(); j++) {
        double bearing_area{0};
        int projected_area{0};

        /* This is assuming column-major storage */
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
                    bearing_area += _triangle(h00, h10, h01, heights(j));
                    projected_area += 1;
                }
                /* Check for NaNs and only add triangle if there are no NaNs */
                if (h10 == h10 && h11 == h11 && h01 == h01) {
                    bearing_area += _triangle(h10, h11, h01, heights(j));
                    projected_area += 1;
                }
            }
        }
        fractional_bearing_areas(j) = bearing_area / projected_area;
    }

    return fractional_bearing_areas;
}
