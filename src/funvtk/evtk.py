import struct
import sys
from typing import BinaryIO

import numpy as np
from numpy.typing import NDArray

# NumPy 数据类型到 struct 格式字符的映射
np_to_struct = {
    "int8": "b",
    "uint8": "B",
    "int16": "h",
    "uint16": "H",
    "int32": "i",
    "uint32": "I",
    "int64": "q",
    "uint64": "Q",
    "float32": "f",
    "float64": "d",
}


def _get_byte_order_char() -> str:
    # 格式说明见 https://docs.python.org/3/library/struct.html
    if sys.byteorder == "little":
        return "<"
    else:
        return ">"


# ================================
#        Python 接口
# ================================
def writeBlockSize(stream: BinaryIO, block_size: int) -> None:
    """将数据块大小写为 64 位无符号整数。"""
    fmt = _get_byte_order_char() + "Q"
    stream.write(struct.pack(fmt, block_size))


def writeArrayToFile(stream: BinaryIO, data: NDArray) -> None:
    """按 VTK 所需的 Fortran 顺序写入一个 NumPy 数组。"""
    assert data.ndim == 1 or data.ndim == 3
    fmt = _get_byte_order_char() + str(data.size) + np_to_struct[data.dtype.name]

    # VTK 要求 Fortran 顺序，多维 C 布局数组需要在此展开。
    dd = np.ravel(data, order="F")

    binary = struct.pack(fmt, *dd)
    stream.write(binary)


# ==============================================================================
def writeArraysToFile(stream: BinaryIO, x: NDArray, y: NDArray, z: NDArray) -> None:
    """交错写入三个形状和数据类型一致的 NumPy 数组。"""
    assert x.size == y.size == z.size, "Different array sizes."
    assert x.dtype.itemsize == y.dtype.itemsize == z.dtype.itemsize, (
        "Different item sizes."
    )

    nitems = x.size
    fmt = _get_byte_order_char() + "1" + np_to_struct[x.dtype.name]

    # VTK 要求 Fortran 顺序，多维 C 布局数组需要在此展开。
    xx = np.ravel(x, order="F")
    yy = np.ravel(y, order="F")
    zz = np.ravel(z, order="F")

    for i in range(nitems):
        bx = struct.pack(fmt, xx[i])
        by = struct.pack(fmt, yy[i])
        bz = struct.pack(fmt, zz[i])
        stream.write(bx)
        stream.write(by)
        stream.write(bz)
