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

#include "eigen_helper.h"

Eigen::ArrayXd nonuniform_bearing_area(Eigen::Ref<Eigen::ArrayXd> x, Eigen::Ref<Eigen::ArrayXd> h,
                                       Eigen::Ref<ArrayXl> el_sort_by_max, Eigen::Ref<Eigen::ArrayXd> heights);
Eigen::ArrayXd uniform1d_bearing_area(Eigen::Ref<Eigen::ArrayXd> topography_h, bool periodic,
                                      Eigen::Ref<Eigen::ArrayXd> heights);
Eigen::ArrayXd uniform2d_bearing_area(Eigen::Ref<RowMajorXXd> topography_h, bool periodic,
                                      Eigen::Ref<Eigen::ArrayXd> heights);
