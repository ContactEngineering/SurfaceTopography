/*
@file   autocorrelation.cpp

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   11 Jul 2019

@brief  Height-difference autocorrelation of nonuniform line scans

@section LICENCE

Copyright 2019-2025 Lars Pastewka

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
#include <cmath>
#include <stdexcept>

#include "autocorrelation.h"


std::tuple<Eigen::ArrayXd, Eigen::ArrayXd> nonuniform_autocorrelation(
    Eigen::Ref<Eigen::ArrayXd> x,
    Eigen::Ref<Eigen::ArrayXd> h,
    double physical_size,
    std::optional<Eigen::Ref<Eigen::ArrayXd>> distances_opt)
{
    const auto nb_grid_pts = x.size();

    if (h.size() != nb_grid_pts) {
        throw std::runtime_error("x- and h-arrays must contain identical number of data points.");
    }

    // Create or use provided distances array
    Eigen::ArrayXd distances;
    if (distances_opt) {
        distances = *distances_opt;
    } else {
        distances = Eigen::ArrayXd::LinSpaced(nb_grid_pts, 0.0, physical_size * (nb_grid_pts - 1) / nb_grid_pts);
    }

    const auto nb_distance_pts = distances.size();
    Eigen::ArrayXd acf = Eigen::ArrayXd::Zero(nb_distance_pts);

    for (Eigen::Index i = 0; i < nb_grid_pts - 1; ++i) {
        double x1 = x(i);
        double h1 = h(i);
        double s1 = (h(i + 1) - h1) / (x(i + 1) - x1);
        for (Eigen::Index j = 0; j < nb_grid_pts - 1; ++j) {
            // Determine lower and upper distance between segment i, i+1 and segment j, j+1
            double x2 = x(j);
            double h2 = h(j);
            double s2 = (h(j + 1) - h(j)) / (x(j + 1) - x(j));
            for (Eigen::Index k = 0; k < nb_distance_pts; ++k) {
                double b1 = std::max(x1, x2 - distances(k));
                double b2 = std::min(x(i + 1), x(j + 1) - distances(k));
                double b = (b1 + b2) / 2;
                double db = (b2 - b1) / 2;
                if (db > 0) {
                    // f1[x_] := (h1 + s1*(x - x1))
                    // f2[x_] := (h2 + s2*(x - x2))
                    // FullSimplify[Integrate[f1[x]*f2[x + d], {x, b - db, b + db}]]
                    //   = 2 * f1[b] * f2[b + d] * db + 2 * s1 * s2 * db ** 3 / 3
                    double z = h2 - s2 * x2 + (b + distances(k)) * s2 - h1 + s1 * x1 - b * s1;
                    double ds = s1 - s2;
                    acf(k) += (db * (3 * z * z + ds * ds * db * db)) / 3;
                }
            }
        }
    }

    for (Eigen::Index k = 0; k < nb_distance_pts; ++k) {
        acf(k) /= (physical_size - distances(k));
    }

    return {distances, acf};
}
