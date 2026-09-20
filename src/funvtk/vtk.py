# **************************************
# *  将数据导出为二进制 VTK 文件的    *
# *  低层 Python 库。                  *
# **************************************

import os
import sys
from collections.abc import Iterable, Sequence
from typing import Any

from .evtk import writeArraysToFile, writeArrayToFile, writeBlockSize
from .xml import XmlWriter

# ================================
#            VTK 类型
# ================================


#     文件类型
class VtkFileType:
    """描述一种 VTK 文件格式及其扩展名。"""

    def __init__(self, name: str, ext: str) -> None:
        """创建文件类型描述。"""
        self.name = name
        self.ext = ext

    def __str__(self) -> str:
        """返回便于阅读的类型描述。"""
        return f"Name: {self.name}  Ext: {self.ext} \n"


VtkImageData = VtkFileType("ImageData", ".vti")
VtkPolyData = VtkFileType("PolyData", ".vtp")
VtkRectilinearGrid = VtkFileType("RectilinearGrid", ".vtr")
VtkStructuredGrid = VtkFileType("StructuredGrid", ".vts")
VtkUnstructuredGrid = VtkFileType("UnstructuredGrid", ".vtu")


#    数据类型
class VtkDataType:
    """描述一种 VTK 标量数据类型。"""

    def __init__(self, size: int, name: str) -> None:
        """创建数据类型描述。"""
        self.size = size
        self.name = name

    def __str__(self) -> str:
        """返回便于阅读的类型描述。"""
        return f"Type: {self.name}  Size: {self.size} \n"


VtkInt8 = VtkDataType(1, "Int8")
VtkUInt8 = VtkDataType(1, "UInt8")
VtkInt16 = VtkDataType(2, "Int16")
VtkUInt16 = VtkDataType(2, "UInt16")
VtkInt32 = VtkDataType(4, "Int32")
VtkUInt32 = VtkDataType(4, "UInt32")
VtkInt64 = VtkDataType(8, "Int64")
VtkUInt64 = VtkDataType(8, "UInt64")
VtkFloat32 = VtkDataType(4, "Float32")
VtkFloat64 = VtkDataType(8, "Float64")

# NumPy 到 VTK 数据类型的映射
np_to_vtk = {
    "int8": VtkInt8,
    "uint8": VtkUInt8,
    "int16": VtkInt16,
    "uint16": VtkUInt16,
    "int32": VtkInt32,
    "uint32": VtkUInt32,
    "int64": VtkInt64,
    "uint64": VtkUInt64,
    "float32": VtkFloat32,
    "float64": VtkFloat64,
}


#    单元类型
class VtkCellType:
    """描述一种 VTK 单元类型。"""

    def __init__(self, tid: int, name: str) -> None:
        """创建单元类型描述。"""
        self.tid = tid
        self.name = name

    def __str__(self) -> str:
        """返回便于阅读的类型描述。"""
        return f"VtkCellType( {self.name} ) \n"


VtkVertex = VtkCellType(1, "Vertex")
VtkPolyVertex = VtkCellType(2, "PolyVertex")
VtkLine = VtkCellType(3, "Line")
VtkPolyLine = VtkCellType(4, "PolyLine")
VtkTriangle = VtkCellType(5, "Triangle")
VtkTriangleStrip = VtkCellType(6, "TriangleStrip")
VtkPolygon = VtkCellType(7, "Polygon")
VtkPixel = VtkCellType(8, "Pixel")
VtkQuad = VtkCellType(9, "Quad")
VtkTetra = VtkCellType(10, "Tetra")
VtkVoxel = VtkCellType(11, "Voxel")
VtkHexahedron = VtkCellType(12, "Hexahedron")
VtkWedge = VtkCellType(13, "Wedge")
VtkPyramid = VtkCellType(14, "Pyramid")
VtkQuadraticEdge = VtkCellType(21, "Quadratic_Edge")
VtkQuadraticTriangle = VtkCellType(22, "Quadratic_Triangle")
VtkQuadraticQuad = VtkCellType(23, "Quadratic_Quad")
VtkQuadraticTetra = VtkCellType(24, "Quadratic_Tetra")
VtkQuadraticHexahedron = VtkCellType(25, "Quadratic_Hexahedron")


# ==============================
#       辅助函数
# ==============================
def _mix_extents(start: Sequence[int], end: Sequence[int]) -> str:
    assert len(start) == len(end) == 3
    string = f"{start[0]} {end[0]} {start[1]} {end[1]} {start[2]} {end[2]}"
    return string


def _array_to_string(a: Iterable[object]) -> str:
    s = "".join([repr(num) + " " for num in a])
    return s


def _get_byte_order() -> str:
    if sys.byteorder == "little":
        return "LittleEndian"
    else:
        return "BigEndian"


# ================================
#        VtkGroup 类
# ================================
class VtkGroup:
    """生成用于组织多个 VTK 数据文件的 PVD 集合文件。"""

    def __init__(self, filepath: str) -> None:
        """创建集合文件，``filepath`` 不包含扩展名。"""
        self.xml = XmlWriter(filepath + ".pvd")
        self.xml.openElement("VTKFile")
        self.xml.addAttributes(
            type="Collection", version="0.1", byte_order=_get_byte_order()
        )
        self.xml.openElement("Collection")
        self.root = os.path.dirname(filepath)

    def save(self) -> None:
        """结束集合并关闭文件。"""
        self.xml.closeElement("Collection")
        self.xml.closeElement("VTKFile")
        self.xml.close()

    def addFile(
        self,
        filepath: str,
        sim_time: int | float,
        group: str = "",
        part: str | int = "0",
    ) -> None:
        """将 VTK 文件及其模拟时间加入集合。"""
        filename = os.path.relpath(filepath, start=self.root)
        self.xml.openElement("DataSet")
        self.xml.addAttributes(timestep=sim_time, group=group, part=part, file=filename)
        self.xml.closeElement()


# ================================
#        VtkFile 类
# ================================
class VtkFile:
    """以 VTK XML 格式写入网格和二进制数组。"""

    def __init__(
        self, filepath: str, ftype: VtkFileType, largeFile: bool = False
    ) -> None:
        """创建 VTK 文件，``filepath`` 不包含扩展名。"""
        self.ftype = ftype
        self.filename = filepath + ftype.ext
        self.xml = XmlWriter(self.filename)
        self.offset = 0  # 二进制段起点后的字节偏移量
        self.appendedDataIsOpen = False
        self.xml.openElement("VTKFile").addAttributes(
            type=ftype.name,
            version="1.0",
            byte_order=_get_byte_order(),
            header_type="UInt64",
        )

    def addComments(self, comments: Iterable[str]) -> None:
        """将字符串序列写入 XML 头部作为注释。"""
        assert not self.appendedDataIsOpen
        for c in comments:
            self.xml.addComment(c)

    def getFileName(self) -> str:
        """返回输出文件的绝对路径。"""
        return os.path.abspath(self.filename)

    def openPiece(
        self,
        start: Sequence[int] | None = None,
        end: Sequence[int] | None = None,
        npoints: int | str | None = None,
        ncells: int | str | None = None,
        nverts: int | str | None = None,
        nlines: int | str | None = None,
        nstrips: int | str | None = None,
        npolys: int | str | None = None,
    ) -> "VtkFile":
        """打开网格分块并写入范围或元素数量，返回自身以支持链式调用。"""
        self.xml.openElement("Piece")
        if start and end:
            ext = _mix_extents(start, end)
            self.xml.addAttributes(Extent=ext)

        elif ncells and npoints:
            self.xml.addAttributes(NumberOfPoints=npoints, NumberOfCells=ncells)

        elif npoints or nverts or nlines or nstrips or npolys:
            if npoints is None:
                npoints = str(0)
            if nverts is None:
                nverts = str(0)
            if nlines is None:
                nlines = str(0)
            if nstrips is None:
                nstrips = str(0)
            if npolys is None:
                npolys = str(0)
            self.xml.addAttributes(
                NumberOfPoints=npoints,
                NumberOfVerts=nverts,
                NumberOfLines=nlines,
                NumberOfStrips=nstrips,
                NumberOfPolys=npolys,
            )
        else:
            assert False

        return self

    def closePiece(self) -> None:
        """关闭当前网格分块。"""
        self.xml.closeElement("Piece")

    def openData(
        self,
        nodeType: str,
        scalars: str | None = None,
        vectors: str | None = None,
        normals: str | None = None,
        tensors: str | None = None,
        tcoords: str | None = None,
    ) -> "VtkFile":
        """打开点数据或单元数据段，返回自身以支持链式调用。"""
        self.xml.openElement(nodeType + "Data")
        if scalars:
            self.xml.addAttributes(scalars=scalars)
        if vectors:
            self.xml.addAttributes(vectors=vectors)
        if normals:
            self.xml.addAttributes(normals=normals)
        if tensors:
            self.xml.addAttributes(tensors=tensors)
        if tcoords:
            self.xml.addAttributes(tcoords=tcoords)

        return self

    def closeData(self, nodeType: str) -> None:
        """关闭指定的点数据或单元数据段。"""
        self.xml.closeElement(nodeType + "Data")

    def openGrid(
        self,
        start: Sequence[int] | None = None,
        end: Sequence[int] | None = None,
        origin: Sequence[int | float] | None = None,
        spacing: Sequence[int | float] | None = None,
    ) -> "VtkFile":
        """打开网格段并写入范围、原点和间距，返回自身。"""
        gType = self.ftype.name
        self.xml.openElement(gType)
        if gType == VtkImageData.name:
            if not start or not end or not origin or not spacing:
                assert False
            ext = _mix_extents(start, end)
            self.xml.addAttributes(
                WholeExtent=ext,
                Origin=_array_to_string(origin),
                Spacing=_array_to_string(spacing),
            )

        elif gType == VtkStructuredGrid.name or gType == VtkRectilinearGrid.name:
            if not start or not end:
                assert False
            ext = _mix_extents(start, end)
            self.xml.addAttributes(WholeExtent=ext)

        return self

    def closeGrid(self) -> None:
        """关闭当前网格段。"""
        self.xml.closeElement(self.ftype.name)

    def addHeader(self, name: str, dtype: str, nelem: int, ncomp: int) -> "VtkFile":
        """向 XML 头部加入数组描述，返回自身以支持链式调用。"""
        dtype = np_to_vtk[dtype]

        self.xml.openElement("DataArray")
        self.xml.addAttributes(
            Name=name,
            NumberOfComponents=ncomp,
            type=dtype.name,
            format="appended",
            offset=self.offset,
        )
        self.xml.closeElement()

        self.offset += nelem * ncomp * dtype.size + 8
        return self

    def addData(self, name: str, data: Any) -> None:
        """根据 NumPy 数组或三分量数组向 XML 头部加入描述。"""
        if type(data).__name__ == "tuple":  # 向量数据
            assert len(data) == 3
            x = data[0]
            self.addHeader(name, x.dtype.name, x.size, 3)
        elif type(data).__name__ == "ndarray":
            if data.ndim == 1 or data.ndim == 3:
                self.addHeader(name, data.dtype.name, data.size, 1)
            else:
                assert False, "Bad array shape: " + str(data.shape)
        else:
            assert False, "Argument must be a Numpy array"

    def appendHeader(self, dtype: str, nelem: int, ncomp: int) -> None:
        """写入待追加数据块的大小；调用后应立即写入数据。"""
        self.openAppendedData()
        dsize = np_to_vtk[dtype].size
        block_size = dsize * ncomp * nelem
        writeBlockSize(self.xml.stream, block_size)

    def appendData(self, data: Any) -> "VtkFile":
        """向二进制段追加数组数据，返回自身以支持链式调用。"""
        self.openAppendedData()

        if type(data).__name__ == "tuple":  # 三个 NumPy 数组
            ncomp = len(data)
            assert ncomp == 3
            dsize = data[0].dtype.itemsize
            nelem = data[0].size
            block_size = ncomp * nelem * dsize
            writeBlockSize(self.xml.stream, block_size)
            x, y, z = data[0], data[1], data[2]
            writeArraysToFile(self.xml.stream, x, y, z)

        elif type(data).__name__ == "ndarray" and (
            data.ndim == 1 or data.ndim == 3
        ):  # 单个 NumPy 数组
            ncomp = 1
            dsize = data.dtype.itemsize
            nelem = data.size
            block_size = ncomp * nelem * dsize
            writeBlockSize(self.xml.stream, block_size)
            writeArrayToFile(self.xml.stream, data)

        else:
            assert False

        return self

    def openAppendedData(self) -> None:
        """打开二进制追加数据段。"""
        if not self.appendedDataIsOpen:
            self.xml.openElement("AppendedData").addAttributes(encoding="raw").addText(
                "_"
            )
            self.appendedDataIsOpen = True

    def closeAppendedData(self) -> None:
        """关闭二进制追加数据段。"""
        self.xml.closeElement("AppendedData")

    def openElement(self, tagName: str) -> None:
        """打开 Coordinates、Points 或 Verts 等 XML 元素。"""
        self.xml.openElement(tagName)

    def closeElement(self, tagName: str) -> None:
        """关闭指定 XML 元素。"""
        self.xml.closeElement(tagName)

    def save(self) -> None:
        """结束文档并关闭文件。"""
        if self.appendedDataIsOpen:
            self.xml.closeElement("AppendedData")
        self.xml.closeElement("VTKFile")
        self.xml.close()
