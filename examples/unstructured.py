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

# ************************************************************************
# * 高层 unstructured_grid_to_vtk 函数示例。                        *
# * This example shows how to export a unstructured grid give its        *
# * nodes and topology through a connectivity and offset lists.          *
# * Check the VTK file format for details of the unstructured grid.      *
# ************************************************************************
from pathlib import Path

import numpy as np

from funvtk.hl import unstructured_grid_to_vtk
from funvtk.vtk import VtkQuad, VtkTriangle

FILE_PATH = "./unstructured"


def clean():
    Path(FILE_PATH + ".vtu").unlink(missing_ok=True)


def run():

    # 定义顶点
    x = np.zeros(6)
    y = np.zeros(6)
    z = np.zeros(6)

    x[0], y[0], z[0] = 0.0, 0.0, 0.0
    x[1], y[1], z[1] = 1.0, 0.0, 0.0
    x[2], y[2], z[2] = 2.0, 0.0, 0.0
    x[3], y[3], z[3] = 0.0, 1.0, 0.0
    x[4], y[4], z[4] = 1.0, 1.0, 0.0
    x[5], y[5], z[5] = 2.0, 1.0, 0.0

    # 定义每个单元包含的顶点
    conn = np.zeros(10)

    conn[0], conn[1], conn[2] = 0, 1, 3  # 第一个三角形
    conn[3], conn[4], conn[5] = 1, 4, 3  # 第二个三角形
    conn[6], conn[7], conn[8], conn[9] = 1, 2, 5, 4  # 矩形

    # 定义每个单元最后一个顶点的偏移
    offset = np.zeros(3)
    offset[0] = 3
    offset[1] = 6
    offset[2] = 10

    # 定义单元类型

    ctype = np.zeros(3)
    ctype[0], ctype[1] = VtkTriangle.tid, VtkTriangle.tid
    ctype[2] = VtkQuad.tid

    cd = np.random.rand(3)
    cell_data = {"pressure": cd}

    pd = np.random.rand(6)
    point_data = {"ec": pd}

    comments = ["comment 1", "comment 2"]
    unstructured_grid_to_vtk(
        FILE_PATH,
        x,
        y,
        z,
        connectivity=conn,
        offsets=offset,
        cell_types=ctype,
        cell_data=cell_data,
        point_data=point_data,
        comments=comments,
    )


if __name__ == "__main__":
    run()
