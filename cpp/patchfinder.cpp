/*
@file   patchfinder.cpp

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   10 Apr 2017

@brief  Analysis of contact patch geometries

@section LICENCE

Copyright 2015-2025 Till Junge, Lars Pastewka

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
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "patchfinder.h"
#include "stack.h"

// This is sufficient for typically 2048x2048
#define DEFAULT_STACK_SIZE 16*1024*1024

// Default 8-neighbor stencil
static const std::vector<std::pair<int, int>> default_stencil = {
    {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}, {0, 1}
};


void fill_patch(Eigen::Index nx, Eigen::Index ny, const RowMajorXXb &map,
                std::ptrdiff_t i0, std::ptrdiff_t j0, int p, bool periodic,
                const std::vector<std::pair<int, int>> &stencil, RowMajorXXi &id)
{
    Stack stack(DEFAULT_STACK_SIZE);

    stack.push(i0, j0);
    id(i0, j0) = p;

    while (!stack.is_empty()) {
        std::ptrdiff_t i, j;
        stack.pop_bottom(i, j);

        for (const auto &[di, dj] : stencil) {
            // Periodic boundary conditions
            std::ptrdiff_t jj = j + dj;
            if (periodic) {
                if (jj < 0) jj += ny;
                if (jj > ny - 1) jj -= ny;
            } else {
                if (jj < 0) continue;
                if (jj > ny - 1) continue;
            }

            // Periodic boundary conditions
            std::ptrdiff_t ii = i + di;
            if (periodic) {
                if (ii < 0) ii += nx;
                if (ii > nx - 1) ii -= nx;
            } else {
                if (ii < 0) continue;
                if (ii > nx - 1) continue;
            }

            if (map(ii, jj) && id(ii, jj) == 0) {
                stack.push(ii, jj);
                id(ii, jj) = p;
            }
        }
    }
}


std::tuple<int, RowMajorXXi> assign_patch_numbers(
    Eigen::Ref<RowMajorXXb> map,
    bool periodic,
    std::optional<Eigen::Ref<ArrayXl>> stencil_opt)
{
    const auto nx = map.rows();
    const auto ny = map.cols();

    // Build stencil
    std::vector<std::pair<int, int>> stencil;
    if (stencil_opt) {
        auto &s = *stencil_opt;
        if (s.size() % 2 != 0) {
            throw std::runtime_error("Stencil must have even number of elements.");
        }
        for (Eigen::Index i = 0; i < s.size() / 2; ++i) {
            stencil.push_back({static_cast<int>(s(2*i)), static_cast<int>(s(2*i + 1))});
        }
    } else {
        stencil = default_stencil;
    }

    RowMajorXXi id = RowMajorXXi::Zero(nx, ny);
    int p = 0;

    for (Eigen::Index i = 0; i < nx; ++i) {
        for (Eigen::Index j = 0; j < ny; ++j) {
            if (map(i, j) && id(i, j) == 0) {
                p++;
                fill_patch(nx, ny, map, i, j, p, periodic, stencil, id);
            }
        }
    }

    return {p, id};
}


void fill_segment(Eigen::Index ny, const RowMajorXXb &map, Eigen::Index row,
                  std::ptrdiff_t j, int p, RowMajorXXi &id)
{
    id(row, j) = p;

    std::ptrdiff_t jj = j + 1;
    // Periodic boundary conditions
    if (jj > ny - 1) jj -= ny;

    while (map(row, jj) && id(row, jj) == 0) {
        id(row, jj) = p;
        jj++;
        if (jj > ny - 1) jj -= ny;
    }

    jj = j - 1;
    if (jj < 0) jj += ny;

    while (map(row, jj) && id(row, jj) == 0) {
        id(row, jj) = p;
        jj--;
        // Periodic boundary conditions
        if (jj < 0) jj += ny;
    }
}


std::tuple<int, RowMajorXXi> assign_segment_numbers(Eigen::Ref<RowMajorXXb> map)
{
    const auto nx = map.rows();
    const auto ny = map.cols();

    RowMajorXXi id = RowMajorXXi::Zero(nx, ny);
    int p = 0;

    for (Eigen::Index i = 0; i < nx; ++i) {
        for (Eigen::Index j = 0; j < ny; ++j) {
            if (map(i, j) && id(i, j) == 0) {
                p++;
                fill_segment(ny, map, i, j, p, id);
            }
        }
    }

    return {p, id};
}


void track_distance(Eigen::Index nx, Eigen::Index ny, const RowMajorXXb &map,
                    RowMajorXXd &dist, RowMajorXXi &next)
{
    Stack stack(DEFAULT_STACK_SIZE);

    // Fill stack with all possible map points
    for (Eigen::Index i = 0; i < nx; ++i) {
        for (Eigen::Index j = 0; j < ny; ++j) {
            if (map(i, j)) {
                // Start tracking here with zero distance
                stack.push(static_cast<int>(i), static_cast<int>(j),
                           static_cast<int>(i), static_cast<int>(j));
            }
        }
    }

    // While there is something to look for
    while (!stack.is_empty()) {
        int i, j, i0, j0;
        stack.pop_bottom(i, j, i0, j0);

        int di = i - i0;
        int dj = j - j0;

        if (di > nx / 2) di = nx - di;
        if (dj > ny / 2) dj = ny - dj;
        if (di < -nx / 2) di = nx + di;
        if (dj < -ny / 2) dj = ny + dj;

        double d = sqrt(di * di + dj * dj);

        // Is i0, j0 closer than what is currently stored?
        if (d < dist(i, j)) {
            dist(i, j) = d;
            next(i, j) = i0 * ny + j0;

            // Loop over all neighbors
            for (int joff = -1; joff <= 1; ++joff) {
                // Periodic boundary conditions
                int jjj = j + joff;
                while (jjj < 0) jjj += ny;
                while (jjj >= ny) jjj -= ny;

                for (int ioff = -1; ioff <= 1; ++ioff) {
                    // Exclude middle
                    if (ioff != 0 || joff != 0) {
                        // Periodic boundary conditions
                        int iii = i + ioff;
                        while (iii < 0) iii += nx;
                        while (iii >= nx) iii -= nx;

                        // Push to stack if not on map
                        if (!map(iii, jjj)) {
                            stack.push(iii, jjj, i0, j0);
                        }
                    }
                }
            }
        }
    }
}


RowMajorXXd distance_map(Eigen::Ref<RowMajorXXb> map)
{
    const auto nx = map.rows();
    const auto ny = map.cols();

    // This stores the distance to the closest point on the contour
    RowMajorXXd dist = RowMajorXXd::Constant(nx, ny, nx * ny);

    // This stores the index of the closest point
    RowMajorXXi next = RowMajorXXi::Constant(nx, ny, nx * ny);

    // Track distances from contact edge
    track_distance(nx, ny, map, dist, next);

    return dist;
}


void track_closest_patch(Eigen::Index nx, Eigen::Index ny, const RowMajorXXi &map,
                         RowMajorXXd &dist, RowMajorXXi &next)
{
    Stack stack(DEFAULT_STACK_SIZE);

    // Fill stack with all possible map points
    for (Eigen::Index i = 0; i < nx; ++i) {
        for (Eigen::Index j = 0; j < ny; ++j) {
            if (map(i, j)) {
                // Start tracking here with zero distance
                stack.push(static_cast<int>(i), static_cast<int>(j),
                           static_cast<int>(i), static_cast<int>(j));
            }
        }
    }

    // While there is something to look for
    while (!stack.is_empty()) {
        int i, j, i0, j0;
        stack.pop_bottom(i, j, i0, j0);

        int di = i - i0;
        int dj = j - j0;

        if (di > nx / 2) di = nx - di;
        if (dj > ny / 2) dj = ny - dj;
        if (di < -nx / 2) di = nx + di;
        if (dj < -ny / 2) dj = ny + dj;

        double d = sqrt(di * di + dj * dj);

        // Is i0, j0 closer than what is currently stored?
        if (d < dist(i, j)) {
            dist(i, j) = d;
            next(i, j) = map(i0, j0);

            // Loop over all neighbors
            for (int joff = -1; joff <= 1; ++joff) {
                // Periodic boundary conditions
                int jjj = j + joff;
                while (jjj < 0) jjj += ny;
                while (jjj >= ny) jjj -= ny;

                for (int ioff = -1; ioff <= 1; ++ioff) {
                    // Exclude middle
                    if (ioff != 0 || joff != 0) {
                        // Periodic boundary conditions
                        int iii = i + ioff;
                        while (iii < 0) iii += nx;
                        while (iii >= nx) iii -= nx;

                        // Push to stack if not on map
                        if (!map(iii, jjj)) {
                            stack.push(iii, jjj, i0, j0);
                        }
                    }
                }
            }
        }
    }
}


RowMajorXXi closest_patch_map(Eigen::Ref<RowMajorXXi> map)
{
    const auto nx = map.rows();
    const auto ny = map.cols();

    // This stores the distance to the closest point on the contour
    RowMajorXXd dist = RowMajorXXd::Constant(nx, ny, nx * ny);

    // This stores the tag of the closest point
    RowMajorXXi next = RowMajorXXi::Constant(nx, ny, nx * ny);

    // Track distances from contact edge
    track_closest_patch(nx, ny, map, dist, next);

    return next;
}


RowMajorXXd shortest_distance(
    Eigen::Ref<RowMajorXXb> fromc,
    Eigen::Ref<RowMajorXXb> fromp,
    Eigen::Ref<RowMajorXXb> to,
    int maxd)
{
    const auto nx = fromc.rows();
    const auto ny = fromc.cols();

    if (fromp.rows() != nx || fromp.cols() != ny) {
        throw std::runtime_error("All three maps need to have identical dimensions.");
    }
    if (to.rows() != nx || to.cols() != ny) {
        throw std::runtime_error("All three maps need to have identical dimensions.");
    }

    RowMajorXXd dist = RowMajorXXd::Zero(nx, ny);

    // Make sure there is something to find
    bool found = false;
    for (Eigen::Index k = 0; k < nx * ny && !found; ++k) {
        if (to(k / ny, k % ny)) {
            found = true;
        }
    }
    if (!found) {
        throw std::runtime_error("No patches found in second map.");
    }

    // Find distance to patches in *to*
    if (maxd < 0) maxd = 2 * nx;
    double sqrt2 = sqrt(2.0);

    for (Eigen::Index j = 0; j < ny; ++j) {
        for (Eigen::Index i = 0; i < nx; ++i) {
            if (fromc(i, j)) {
                double d = maxd + 1.0;

                if (to(i, j)) {
                    // This is also the edge, set distance to 0
                    d = 0.0;
                } else {
                    int ter = maxd + 1;
                    bool on_some_patch = true;
                    for (int n = 1; n <= ter && on_some_patch; ++n) {
                        on_some_patch = false;

                        for (int joff = -n; joff <= n; ++joff) {
                            int jjj = j + joff;
                            while (jjj < 0) jjj += ny;
                            while (jjj >= static_cast<int>(ny)) jjj -= ny;

                            for (int ioff = -n; ioff <= n; ++ioff) {
                                if (abs(ioff) == n || abs(joff) == n) {
                                    int iii = i + ioff;
                                    while (iii < 0) iii += nx;
                                    while (iii >= static_cast<int>(nx)) iii -= nx;

                                    if (fromp(iii, jjj)) on_some_patch = true;

                                    if (to(iii, jjj)) {
                                        double curd = sqrt(ioff * ioff + joff * joff);
                                        if (curd < d) {
                                            d = curd;
                                            // This could be at distance sqrt(2)*n, hence we need to
                                            // go to rectangles with side length >sqrt(2)*n
                                            int newter = static_cast<int>(n * (sqrt2 + 1));
                                            if (newter < ter) ter = newter;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if (d < maxd) dist(i, j) = d;
            }
        }
    }

    return dist;
}


void fill_correlation_function(int max_lin_dist, int max_dist_sq, double map1val,
                               Eigen::Index i, Eigen::Index j, Eigen::Index nx, Eigen::Index ny,
                               const RowMajorXXd &map2, Eigen::ArrayXd &c, Eigen::ArrayXi &n)
{
    // Loop over all neighbors
    for (int joff = -max_lin_dist; joff <= max_lin_dist; ++joff) {
        int jjj = j + joff;
        while (jjj < 0) jjj += ny;
        while (jjj >= static_cast<int>(ny)) jjj -= ny;

        for (int ioff = -max_lin_dist; ioff <= max_lin_dist; ++ioff) {
            // Exclude middle
            if (ioff != 0 || joff != 0) {
                int iii = i + ioff;
                while (iii < 0) iii += nx;
                while (iii >= static_cast<int>(nx)) iii -= nx;

                int dist_sq = ioff * ioff + joff * joff;
                if (dist_sq <= max_dist_sq) {
                    c(dist_sq - 1) += map1val * map2(iii, jjj);
                    n(dist_sq - 1) += 1;
                }
            }
        }
    }
}


std::tuple<Eigen::ArrayXd, Eigen::ArrayXd, Eigen::ArrayXd> correlation_function(
    Eigen::Ref<RowMajorXXd> map1,
    Eigen::Ref<RowMajorXXd> map2,
    int max_dist)
{
    const auto nx = map1.rows();
    const auto ny = map1.cols();

    if (map2.rows() != nx || map2.cols() != ny) {
        throw std::runtime_error("Both maps need to have identical dimensions.");
    }

    int max_dist_sq = max_dist * max_dist;

    // Correlation function
    Eigen::ArrayXd c = Eigen::ArrayXd::Zero(max_dist_sq);

    // Number of points found at a certain distance
    Eigen::ArrayXi n = Eigen::ArrayXi::Zero(max_dist_sq);

    // Maximum search distance in x and y directions
    int max_lindist = static_cast<int>(floor(sqrt(max_dist_sq))) + 1;

    // Tracking algorithm
    for (Eigen::Index j = 0; j < ny; ++j) {
        for (Eigen::Index i = 0; i < nx; ++i) {
            // Start tracking here with zero distance
            fill_correlation_function(max_lindist, max_dist_sq, map1(i, j),
                                      i, j, nx, ny, map2, c, n);
        }
    }

    // Find nonzero entries
    int nz = 0;
    for (int k = 0; k < max_dist_sq; ++k) {
        if (n(k) > 0) nz++;
    }

    // Allocate correlation function of proper length
    Eigen::ArrayXd r(nz);     // Distance
    Eigen::ArrayXd cc(nz);    // Correlation function
    Eigen::ArrayXd Icc(nz);   // Integrated correlation function

    Icc.setZero();

    // Normalize and integrate
    nz = 0;
    for (int k = 0; k < max_dist_sq; ++k) {
        if (n(k) > 0) {
            double sqrtk = sqrt(1 + k);

            // Integrate
            double ival = c(k) / sqrtk;
            int inz = nz;
            for (int l = k; l < max_dist_sq; ++l) {
                if (n(l) > 0) {
                    Icc(inz) += ival;
                    inz++;
                }
            }

            // Normalize
            r(nz) = sqrtk;
            cc(nz) = c(k) / n(k);
            nz++;
        }
    }

    return {r, cc, Icc};
}


double perimeter_length(Eigen::Ref<RowMajorXXb> map)
{
    double sqrt2 = sqrt(2.0);
    const auto nx = map.rows();
    const auto ny = map.cols();

    double length = 0.0;

    for (Eigen::Index j = 0; j < ny; ++j) {
        for (Eigen::Index i = 0; i < nx; ++i) {
            if (map(i, j)) {
                int n1 = 0, nsqrt2 = 0;

                // Top right
                Eigen::Index ii = i + 1;
                if (ii > nx - 1) ii -= nx;
                Eigen::Index jj = j + 1;
                if (jj > ny - 1) jj -= ny;

                // Right
                if (map(ii, j)) n1++;
                // Top
                if (map(i, jj)) n1++;
                // Top right
                if (map(ii, jj)) nsqrt2++;

                // Top left
                ii = i - 1;
                if (ii < 0) ii += nx;

                // Left
                if (map(ii, j)) n1++;
                // Top left
                if (map(ii, jj)) nsqrt2++;

                // Bottom left
                jj = j - 1;
                if (jj < 0) jj += ny;

                // Bottom
                if (map(i, jj)) n1++;
                // Bottom left
                if (map(ii, jj)) nsqrt2++;

                // Bottom right
                ii = i + 1;
                if (ii > nx - 1) ii -= nx;
                // Bottom right
                if (map(ii, jj)) nsqrt2++;

                if (n1 >= 2) {
                    length += 1.0;
                } else if (n1 == 1) {
                    if (nsqrt2 >= 1) {
                        length += 0.5 * (1.0 + sqrt2);
                    }
                } else {
                    length += 0.5 * sqrt2;
                }
            }
        }
    }

    return length;
}
