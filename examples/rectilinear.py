#! /usr/bin/env python

# **************************************************************
# * 高层 rectilinear_to_vtk 函数示例：导出矩形网格。            *
# **************************************************************
from pathlib import Path

import numpy as np

from funvtk.hl import rectilinear_to_vtk

FILE_PATH = "./rectilinear"


def clean():
    Path(FILE_PATH + ".vtr").unlink(missing_ok=True)


def run():
    # 数据尺寸
    nx, ny, nz = 6, 6, 2
    lx, ly, lz = 1.0, 1.0, 1.0
    dx, dy, dz = lx / nx, ly / ny, lz / nz

    ncells = nx * ny * nz
    npoints = (nx + 1) * (ny + 1) * (nz + 1)

    # 坐标
    x = np.arange(0, lx + 0.1 * dx, dx, dtype="float64")
    y = np.arange(0, ly + 0.1 * dy, dy, dtype="float64")
    z = np.arange(0, lz + 0.1 * dz, dz, dtype="float64")

    # 数据变量
    pressure = np.random.rand(ncells).reshape((nx, ny, nz))
    temp = np.random.rand(npoints).reshape((nx + 1, ny + 1, nz + 1))

    comments = ["comment 1", "comment 2"]
    rectilinear_to_vtk(
        FILE_PATH,
        x,
        y,
        z,
        cell_data={"pressure": pressure},
        point_data={"temp": temp},
        comments=comments,
    )


if __name__ == "__main__":
    run()
