/*
@file   patchfinder.h

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

#ifndef __PATCHFINDER_H
#define __PATCHFINDER_H

#include <optional>
#include <tuple>

#include "eigen_helper.h"

std::tuple<int, RowMajorXXi> assign_patch_numbers(
    Eigen::Ref<RowMajorXXb> map,
    bool periodic,
    std::optional<Eigen::Ref<ArrayXl>> stencil = std::nullopt);

std::tuple<int, RowMajorXXi> assign_segment_numbers(
    Eigen::Ref<RowMajorXXb> map);

RowMajorXXd distance_map(Eigen::Ref<RowMajorXXb> map);

RowMajorXXi closest_patch_map(Eigen::Ref<RowMajorXXi> map);

RowMajorXXd shortest_distance(
    Eigen::Ref<RowMajorXXb> fromc,
    Eigen::Ref<RowMajorXXb> fromp,
    Eigen::Ref<RowMajorXXb> to,
    int maxd = -1);

std::tuple<Eigen::ArrayXd, Eigen::ArrayXd, Eigen::ArrayXd> correlation_function(
    Eigen::Ref<RowMajorXXd> map1,
    Eigen::Ref<RowMajorXXd> map2,
    int max_dist);

double perimeter_length(Eigen::Ref<RowMajorXXb> map);

#endif
