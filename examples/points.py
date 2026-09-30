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
# * 高层 points_to_vtk 函数示例。                              *
# **************************************************************
from pathlib import Path

import numpy as np

from funvtk.hl import points_to_vtk, points_to_vtk_as_tin

FILE_PATH1 = "./rnd_points"
FILE_PATH2 = "./rnd_points_TIN"
FILE_PATH3 = "./line_points"
FILE_PATH4 = "./points_as_lists"


def clean():
    for path in (FILE_PATH1, FILE_PATH2, FILE_PATH3, FILE_PATH4):
        Path(path + ".vtu").unlink(missing_ok=True)


def run():
    # 示例 1：随机点
    npoints = 100
    x = np.random.rand(npoints)
    y = np.random.rand(npoints)
    z = np.random.rand(npoints)
    pressure = np.random.rand(npoints)
    temp = np.random.rand(npoints)
    comments = ["comment 1", "comment 2"]

    # 导出前会对键排序，可添加数字前缀控制顺序。
    points_to_vtk(
        FILE_PATH1,
        x,
        y,
        z,
        data={"1_temp": temp, "2_pressure": pressure},
        comments=comments,
    )

    # 示例 2：导出为 TIN
    ndim = 2  # 仅使用 x、y 坐标进行三角剖分
    points_to_vtk_as_tin(
        FILE_PATH2,
        x,
        y,
        z,
        ndim=ndim,
        data={"1_temp": temp, "2_pressure": pressure},
        comments=comments,
    )

    # 示例 3：规则点集
    x = np.arange(1.0, 10.0, 0.1)
    y = np.arange(1.0, 10.0, 0.1)
    z = np.arange(1.0, 10.0, 0.1)

    comments = ["comment 1", "comment 2"]
    points_to_vtk(FILE_PATH3, x, y, z, data={"elev": z}, comments=comments)

    # 示例 4：包含 5 个点的点集
    x = [0.0, 1.0, 0.5, 0.368, 0.4]
    y = [0.3, 2.0, 0.7, 0.1, 0.6]
    z = [1.0, 1.0, 0.3, 0.75, 0.9]
    pressure = [1.0, 2.0, 3.0, 4.0, 5.0]
    temp = [1.0, 2.0, 3.0, 4.0, 5.0]
    comments = ["comment 1", "comment 2"]

    # 导出前会对键排序，可添加数字前缀控制顺序。
    points_to_vtk(
        FILE_PATH4,
        x,
        y,
        z,
        data={"1_temp": temp, "2_pressure": pressure},
        comments=comments,
    )


if __name__ == "__main__":
    run()
