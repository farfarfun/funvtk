#! /usr/bin/env python

######################################################################################
# MIT License
#
# Copyright (c) 2010-2021 Paulo A. Herrera
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
######################################################################################

# **************************************************************
# * 高层 poly_lines_to_vtk 函数示例。                          *
# **************************************************************
from pathlib import Path

import numpy as np

from funvtk.hl import poly_lines_to_vtk

FILE_PATH = "poly_lines"


def clean():
    Path(FILE_PATH + ".vtu").unlink(missing_ok=True)


def run():
    # 定义折线的点坐标
    npoints = 7
    x = np.zeros(npoints)
    y = np.zeros(npoints)
    z = np.zeros(npoints)

    # 第一条折线
    x[0], y[0], z[0] = 0.0, 0.0, 0.0
    x[1], y[1], z[1] = 1.0, 1.0, 0.0
    x[2], y[2], z[2] = 2.0, 0.0, 0.0
    x[3], y[3], z[3] = 3.0, -1.0, 0.0

    # 第二条折线
    x[4], y[4], z[4] = 0.0, 0.0, 3.0
    x[5], y[5], z[5] = 1.0, 1.0, 3.0
    x[6], y[6], z[6] = 2.0, 0.0, 3.0

    # 每条折线包含的点数
    points_per_line = np.zeros(2)
    points_per_line[0] = 4
    points_per_line[1] = 3

    # 数据变量：pressure/temp 为点数据，vel 为单元数据（每条折线一个值）
    pressure = np.random.rand(npoints)
    temp = np.random.rand(npoints)
    vel = np.array([1.0, 5.0])

    poly_lines_to_vtk(
        FILE_PATH,
        x,
        y,
        z,
        points_per_line=points_per_line,
        cell_data={"vel": vel},
        point_data={"temp": temp, "pressure": pressure},
    )


if __name__ == "__main__":
    run()
