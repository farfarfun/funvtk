﻿import struct
import sys
import numpy as np


# Map numpy dtype to struct format
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


def _get_byte_order_char():
    # Check format in https://docs.python.org/3.5/library/struct.html
    if sys.byteorder == "little":
        return "<"
    else:
        return ">"


# ================================
#        Python interface
# ================================
def writeBlockSize(stream, block_size):
    fmt = (
        _get_byte_order_char() + "Q"
    )  # Write size as unsigned long long == 64 bits unsigned integer
    stream.write(struct.pack(fmt, block_size))


def writeArrayToFile(stream, data):
    # stream.flush() # this should not be necessary
    assert data.ndim == 1 or data.ndim == 3
    fmt = (
        _get_byte_order_char() + str(data.size) + np_to_struct[data.dtype.name]
    )  # > for big endian

    # Check if array is contiguous
    # assert data.flags["C_CONTIGUOUS"] or data.flags["F_CONTIGUOUS"]

    # NOTE: VTK expects data in FORTRAN order
    # This is only needed when a multidimensional array has C-layout
    dd = np.ravel(data, order="F")

    bin = struct.pack(fmt, *dd)
    stream.write(bin)


# ==============================================================================
def writeArraysToFile(stream, x, y, z):
    # Check if arrays have same shape and data type
    assert x.size == y.size == z.size, "Different array sizes."
    assert (
        x.dtype.itemsize == y.dtype.itemsize == z.dtype.itemsize
    ), "Different item sizes."

    nitems = x.size
    itemsize = x.dtype.itemsize

    fmt = (
        _get_byte_order_char() + str(1) + np_to_struct[x.dtype.name]
    )  # > for big endian

    # Check if arrays are contiguous
    # assert x.flags["C_CONTIGUOUS"] or x.flags["F_CONTIGUOUS"]
    # assert y.flags["C_CONTIGUOUS"] or y.flags["F_CONTIGUOUS"]
    # assert z.flags["C_CONTIGUOUS"] or z.flags["F_CONTIGUOUS"]

    # NOTE: VTK expects data in FORTRAN order
    # This is only needed when a multidimensional array has C-layout
    xx = np.ravel(x, order="F")
    yy = np.ravel(y, order="F")
    zz = np.ravel(z, order="F")

    # eliminate this loop by creating a composed array.
    for i in range(nitems):
        bx = struct.pack(fmt, xx[i])
        by = struct.pack(fmt, yy[i])
        bz = struct.pack(fmt, zz[i])
        stream.write(bx)
        stream.write(by)
        stream.write(bz)
