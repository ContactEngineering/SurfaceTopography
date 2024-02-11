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

#include <iostream>

#include <algorithm>

#include <Eigen/Dense>

Eigen::ArrayXd nonuniform_bearing_area(Eigen::Ref<Eigen::ArrayXd> topography_x, Eigen::Ref<Eigen::ArrayXd> topography_h,
                                       Eigen::Ref<Eigen::ArrayXd> heights)
{
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
                                      Eigen::Ref<Eigen::ArrayXd> heights)
{
    /* The physical length of the line scan */
    double physical_size{dx * topography_h.size()};
    if (periodic) {
        physical_size += dx;
    }

    /* Bearing area values for each input height */
    Eigen::ArrayXd fractional_bearing_areas(heights.size());

    /* Compute bearing areas */
    for (int j{0}; j < heights.size(); j++) {
        double bearing_area{0};
        auto maxi{periodic ? topography_h.size() : topography_h.size()-1};
        for (int i{0}; i < maxi; i++) {
            const auto i1{i < maxi-1 ? i+1 : 0};
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
