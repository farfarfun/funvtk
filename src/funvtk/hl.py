import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import Delaunay

from .vtk import (
    VtkFile,
    VtkImageData,
    VtkLine,
    VtkPixel,
    VtkPolyLine,
    VtkRectilinearGrid,
    VtkStructuredGrid,
    VtkTetra,
    VtkTriangle,
    VtkUnstructuredGrid,
    VtkVertex,
)

_Data = dict[str, ArrayLike]


# =================================
#       辅助函数
# =================================
def _add_data_to_file(vtk_file, cell_data, point_data):
    # 点数据
    if point_data:
        keys = sorted(list(point_data.keys()))
        vtk_file.openData("Point", scalars=keys[0])
        for key in keys:
            data = point_data[key]
            vtk_file.addData(key, data)
        vtk_file.closeData("Point")

    # 单元数据
    if cell_data:
        keys = sorted(list(cell_data.keys()))
        vtk_file.openData("Cell", scalars=keys[0])
        for key in keys:
            data = cell_data[key]
            vtk_file.addData(key, data)
        vtk_file.closeData("Cell")


def _append_data_to_file(vtk_file, cell_data, point_data):
    # 将数据追加到二进制区段
    if point_data is not None:
        keys = sorted(list(point_data.keys()))
        for key in keys:
            data = point_data[key]
            vtk_file.appendData(data)

    if cell_data is not None:
        keys = sorted(list(cell_data.keys()))
        for key in keys:
            data = cell_data[key]
            vtk_file.appendData(data)


def _convert_list_to_array(list_1d):
    """将列表或元组转为 NumPy 数组，数组原样返回。"""
    if (list_1d is not None) and (not type(list_1d).__name__ == "ndarray"):
        assert isinstance(list_1d, (list, tuple))
        return np.array(list_1d)
    else:
        return list_1d


def _convert_dict_lists_to_arrays(data):
    """将字典中的列表或元组值转为 NumPy 数组。"""
    if data is not None:
        dict = {}
        for k, list_1d in data.items():
            dict[k] = _convert_list_to_array(list_1d)
        return dict
    else:
        return data  # None


def _validate_data_lengths(data, expected_length, argument_name):
    """校验标量数据数组与对应点或单元的数量一致。"""
    if data is None:
        return
    for name, values in data.items():
        if values.size != expected_length:
            raise ValueError(
                f"{argument_name}[{name!r}] 的元素数量必须为 {expected_length}"
            )


# =================================
#       高层函数
# =================================
def image_to_vtk(
    path: str,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    cell_data: _Data | None = None,
    point_data: _Data | None = None,
    comments: list[str] | None = None,
) -> str:
    """将规则图像数据写入 VTK 文件。

    ``path`` 不含扩展名；``origin`` 和 ``spacing`` 分别指定网格原点与间距。
    ``cellData``、``pointData`` 是名称到数据数组的映射，至少提供其中一个以推断
    网格尺寸。``comments`` 会写入 XML 头部。返回生成文件的绝对路径。
    """
    if cell_data is None and point_data is None:
        raise ValueError("必须提供 cell_data 或 point_data")

    # 推断网格尺寸
    start = (0, 0, 0)
    end = None
    if cell_data is not None:
        if not cell_data:
            raise ValueError("cell_data 不能为空")
        keys = list(cell_data.keys())
        data = cell_data[keys[0]]
        end = data.shape
    elif point_data is not None:
        if not point_data:
            raise ValueError("point_data 不能为空")
        keys = list(point_data.keys())
        data = point_data[keys[0]]
        end = data.shape
        end = (end[0] - 1, end[1] - 1, end[2] - 1)

    # 写入文件
    w = VtkFile(path, VtkImageData)
    if comments:
        w.addComments(comments)
    w.openGrid(start=start, end=end, origin=origin, spacing=spacing)
    w.openPiece(start=start, end=end)
    _add_data_to_file(w, cell_data, point_data)
    w.closePiece()
    w.closeGrid()
    _append_data_to_file(w, cell_data, point_data)
    w.save()
    return w.getFileName()


# ==============================================================================
def rectilinear_to_vtk(
    path: str,
    x: NDArray,
    y: NDArray,
    z: NDArray,
    cell_data: _Data | None = None,
    point_data: _Data | None = None,
    comments: list[str] | None = None,
) -> str:
    """将直角坐标网格写入 VTK 文件。

    ``x``、``y``、``z`` 必须是一维节点坐标数组。可通过 ``cellData`` 和
    ``pointData`` 写入单元及点数据，通过 ``comments`` 写入头部注释。
    返回生成文件的绝对路径。
    """
    if not (x.ndim == 1 and y.ndim == 1 and z.ndim == 1):
        raise ValueError("x、y、z 必须是一维数组")
    ftype = VtkRectilinearGrid
    nx, ny, nz = x.size - 1, y.size - 1, z.size - 1
    # 推断网格尺寸
    start = (0, 0, 0)
    end = (nx, ny, nz)

    w = VtkFile(path, ftype)
    if comments:
        w.addComments(comments)
    w.openGrid(start=start, end=end)
    w.openPiece(start=start, end=end)

    w.openElement("Coordinates")
    w.addData("x_coordinates", x)
    w.addData("y_coordinates", y)
    w.addData("z_coordinates", z)
    w.closeElement("Coordinates")

    _add_data_to_file(w, cell_data, point_data)
    w.closePiece()
    w.closeGrid()
    # 写入坐标
    w.appendData(x).appendData(y).appendData(z)
    # 写入数据
    _append_data_to_file(w, cell_data, point_data)
    w.save()
    return w.getFileName()


def structured_to_vtk(
    path: str,
    x: NDArray,
    y: NDArray,
    z: NDArray,
    cell_data: _Data | None = None,
    point_data: _Data | None = None,
    comments: list[str] | None = None,
) -> str:
    """将逻辑结构网格写入 VTK 文件。

    ``x``、``y``、``z`` 必须是形状相同的三维节点坐标数组。可通过
    ``cellData`` 和 ``pointData`` 写入单元及点数据。返回生成文件的绝对路径。
    """
    if not (x.ndim == 3 and y.ndim == 3 and z.ndim == 3):
        raise ValueError("x、y、z 必须是三维数组")
    if not (x.shape == y.shape == z.shape):
        raise ValueError("x、y、z 的形状必须相同")

    ftype = VtkStructuredGrid
    s = x.shape
    nx, ny, nz = s[0] - 1, s[1] - 1, s[2] - 1
    start = (0, 0, 0)
    end = (nx, ny, nz)

    w = VtkFile(path, ftype)
    if comments:
        w.addComments(comments)
    w.openGrid(start=start, end=end)
    w.openPiece(start=start, end=end)
    w.openElement("Points")
    w.addData("points", (x, y, z))
    w.closeElement("Points")

    _add_data_to_file(w, cell_data, point_data)
    w.closePiece()
    w.closeGrid()
    w.appendData((x, y, z))
    _append_data_to_file(w, cell_data, point_data)
    w.save()
    return w.getFileName()


def grid_to_vtk(
    path: str,
    x: NDArray,
    y: NDArray,
    z: NDArray,
    cell_data: _Data | None = None,
    point_data: _Data | None = None,
    comments: list[str] | None = None,
) -> str:
    """根据坐标维度写入直角坐标网格或逻辑结构网格。

    一维 ``x``、``y``、``z`` 生成直角坐标网格，三维数组生成逻辑结构网格。
    ``cellData``、``pointData`` 分别提供单元和点数据。返回生成文件的绝对路径。
    """
    # 推断网格尺寸
    start = (0, 0, 0)
    nx = ny = nz = 0

    if x.ndim == 1 and y.ndim == 1 and z.ndim == 1:
        nx, ny, nz = x.size - 1, y.size - 1, z.size - 1
        isRect = True
        ftype = VtkRectilinearGrid
    elif x.ndim == 3 and y.ndim == 3 and z.ndim == 3:
        s = x.shape
        nx, ny, nz = s[0] - 1, s[1] - 1, s[2] - 1
        isRect = False
        ftype = VtkStructuredGrid
    else:
        raise ValueError("x、y、z 必须全部是一维数组或全部是三维数组")

    if not isRect and not (x.shape == y.shape == z.shape):
        raise ValueError("x、y、z 的形状必须相同")

    end = (nx, ny, nz)
    w = VtkFile(path, ftype)
    if comments:
        w.addComments(comments)
    w.openGrid(start=start, end=end)
    w.openPiece(start=start, end=end)

    if isRect:
        w.openElement("Coordinates")
        w.addData("x_coordinates", x)
        w.addData("y_coordinates", y)
        w.addData("z_coordinates", z)
        w.closeElement("Coordinates")
    else:
        w.openElement("Points")
        w.addData("points", (x, y, z))
        w.closeElement("Points")

    _add_data_to_file(w, cell_data, point_data)
    w.closePiece()
    w.closeGrid()
    # 写入坐标
    if isRect:
        w.appendData(x).appendData(y).appendData(z)
    else:
        w.appendData((x, y, z))
    # 写入数据
    _append_data_to_file(w, cell_data, point_data)
    w.save()
    return w.getFileName()


# ==============================================================================
def points_to_vtk(
    path: str,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    data: _Data | None = None,
    comments: list[str] | None = None,
) -> str:
    """将离散点及其数据写入 VTK 非结构网格。

    ``x``、``y``、``z`` 是长度相同的一维坐标序列，``data`` 是点数据名称到
    等长序列的映射。返回生成文件的绝对路径。
    """
    if not (len(x) == len(y) == len(z)):
        raise ValueError("x、y、z 的长度必须相同")
    x = _convert_list_to_array(x)
    y = _convert_list_to_array(y)
    z = _convert_list_to_array(z)
    data = _convert_dict_lists_to_arrays(data)

    npoints = len(x)
    _validate_data_lengths(data, npoints, "data")

    # 构造网格拓扑数组
    offsets = np.arange(
        start=1, stop=npoints + 1, dtype="int32"
    )  # 每个单元最后一个节点的偏移
    connectivity = np.arange(npoints, dtype="int32")  # 每个点仅与自身连接
    cell_types = np.empty(npoints, dtype="uint8")

    cell_types[:] = VtkVertex.tid

    w = VtkFile(path, VtkUnstructuredGrid)
    if comments:
        w.addComments(comments)
    w.openGrid()
    w.openPiece(ncells=npoints, npoints=npoints)

    w.openElement("Points")
    w.addData("points", (x, y, z))
    w.closeElement("Points")
    w.openElement("Cells")
    w.addData("connectivity", connectivity)
    w.addData("offsets", offsets)
    w.addData("types", cell_types)
    w.closeElement("Cells")

    _add_data_to_file(w, cell_data=None, point_data=data)

    w.closePiece()
    w.closeGrid()
    w.appendData((x, y, z))
    w.appendData(connectivity).appendData(offsets).appendData(cell_types)

    _append_data_to_file(w, cell_data=None, point_data=data)

    w.save()
    return w.getFileName()


# ==============================================================================
def points_to_vtk_as_tin(
    path: str,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    data: _Data | None = None,
    comments: list[str] | None = None,
    ndim: int = 2,
) -> str:
    """对离散点执行 Delaunay 剖分并写入 VTK 非结构网格。

    ``x``、``y``、``z`` 是长度相同的坐标序列。``ndim`` 为 2 时使用 x/y
    三角剖分，为 3 时使用 x/y/z 四面体剖分。未提供 ``data`` 时自动写入高程。
    返回生成文件的绝对路径。
    """

    if not (len(x) == len(y) == len(z)):
        raise ValueError("x、y、z 的长度必须相同")
    if ndim not in (2, 3):
        raise ValueError("ndim 必须为 2 或 3")
    x = _convert_list_to_array(x)
    y = _convert_list_to_array(y)
    z = _convert_list_to_array(z)
    data = _convert_dict_lists_to_arrays(data)

    npts = len(x)

    points = np.zeros((npts, ndim))  # Delaunay 所需的二维或三维临时坐标
    for i in range(npts):
        points[i, 0] = x[i]
        points[i, 1] = y[i]
        if ndim > 2:
            points[i, 2] = z[i]

    tri = Delaunay(points)

    ncells, npoints_per_cell = tri.simplices.shape
    conn = tri.simplices.ravel()
    offset = np.arange(1, ncells + 1) * npoints_per_cell
    vtk_type = VtkTriangle.tid if ndim == 2 else VtkTetra.tid
    cell_type = np.full(ncells, vtk_type, dtype="uint8")

    if not data:
        data = {"Elevation": z}
    return unstructured_grid_to_vtk(
        path,
        x,
        y,
        z,
        connectivity=conn,
        offsets=offset,
        cell_types=cell_type,
        cell_data=None,
        point_data=data,
        comments=comments,
    )


# ==============================================================================
def lines_to_vtk(
    path: str,
    x: NDArray,
    y: NDArray,
    z: NDArray,
    cell_data: _Data | None = None,
    point_data: _Data | None = None,
    comments: list[str] | None = None,
) -> str:
    """将两点一组的线段及关联数据写入 VTK 文件。

    ``x``、``y``、``z`` 必须是长度相同且元素数为偶数的一维数组。
    ``cellData`` 和 ``pointData`` 分别提供线段和顶点数据。返回生成文件的绝对路径。
    """
    if not (x.size == y.size == z.size):
        raise ValueError("x、y、z 的长度必须相同")
    if x.size % 2 != 0:
        raise ValueError("坐标数量必须为偶数")

    x = _convert_list_to_array(x)
    y = _convert_list_to_array(y)
    z = _convert_list_to_array(z)
    cell_data = _convert_dict_lists_to_arrays(cell_data)
    point_data = _convert_dict_lists_to_arrays(point_data)

    npoints = len(x)
    ncells = int(len(x) / 2.0)
    _validate_data_lengths(cell_data, ncells, "cell_data")
    _validate_data_lengths(point_data, npoints, "point_data")

    # 构造网格拓扑数组
    offsets = np.arange(
        start=2, step=2, stop=npoints + 1, dtype="int32"
    )  # 每个单元最后一个节点的偏移
    connectivity = np.arange(npoints, dtype="int32")  # 每个点仅与自身连接
    cell_types = np.empty(npoints, dtype="uint8")

    cell_types[:] = VtkLine.tid

    w = VtkFile(path, VtkUnstructuredGrid)
    if comments:
        w.addComments(comments)
    w.openGrid()
    w.openPiece(ncells=ncells, npoints=npoints)

    w.openElement("Points")
    w.addData("points", (x, y, z))
    w.closeElement("Points")
    w.openElement("Cells")
    w.addData("connectivity", connectivity)
    w.addData("offsets", offsets)
    w.addData("types", cell_types)
    w.closeElement("Cells")

    _add_data_to_file(w, cell_data=cell_data, point_data=point_data)

    w.closePiece()
    w.closeGrid()
    w.appendData((x, y, z))
    w.appendData(connectivity).appendData(offsets).appendData(cell_types)

    _append_data_to_file(w, cell_data=cell_data, point_data=point_data)

    w.save()
    return w.getFileName()


# ==============================================================================
def poly_lines_to_vtk(
    path: str,
    x: NDArray,
    y: NDArray,
    z: NDArray,
    points_per_line: NDArray,
    cell_data: _Data | None = None,
    point_data: _Data | None = None,
    comments: list[str] | None = None,
) -> str:
    """将包含不同点数的折线及关联数据写入 VTK 文件。

    ``x``、``y``、``z`` 保存连续排列的顶点坐标，``pointsPerLine`` 指定每条
    折线的点数。``cellData`` 和 ``pointData`` 分别提供折线和顶点数据。
    返回生成文件的绝对路径。
    """
    if not (x.size == y.size == z.size):
        raise ValueError("x、y、z 的长度必须相同")

    x = _convert_list_to_array(x)
    y = _convert_list_to_array(y)
    z = _convert_list_to_array(z)
    cell_data = _convert_dict_lists_to_arrays(cell_data)
    point_data = _convert_dict_lists_to_arrays(point_data)

    npoints = len(x)
    ncells = points_per_line.size
    if np.any(points_per_line < 2) or points_per_line.sum() != npoints:
        raise ValueError("points_per_line 必须至少为 2，且总和必须等于坐标数量")
    _validate_data_lengths(cell_data, ncells, "cell_data")
    _validate_data_lengths(point_data, npoints, "point_data")

    # 构造网格拓扑数组
    offsets = np.zeros(ncells, dtype="int32")  # 每个单元最后一个节点的偏移
    ii = 0
    for i in range(ncells):
        ii += points_per_line[i]
        offsets[i] = ii

    connectivity = np.arange(npoints, dtype="int32")  # 每条折线连接连续排列的点

    cell_types = np.empty(npoints, dtype="uint8")
    cell_types[:] = VtkPolyLine.tid

    w = VtkFile(path, VtkUnstructuredGrid)
    if comments:
        w.addComments(comments)
    w.openGrid()
    w.openPiece(ncells=ncells, npoints=npoints)

    w.openElement("Points")
    w.addData("points", (x, y, z))
    w.closeElement("Points")
    w.openElement("Cells")
    w.addData("connectivity", connectivity)
    w.addData("offsets", offsets)
    w.addData("types", cell_types)
    w.closeElement("Cells")

    _add_data_to_file(w, cell_data=cell_data, point_data=point_data)

    w.closePiece()
    w.closeGrid()
    w.appendData((x, y, z))
    w.appendData(connectivity).appendData(offsets).appendData(cell_types)

    _append_data_to_file(w, cell_data=cell_data, point_data=point_data)

    w.save()
    return w.getFileName()


# ==============================================================================
def unstructured_grid_to_vtk(
    path: str,
    x: NDArray,
    y: NDArray,
    z: NDArray,
    connectivity: ArrayLike,
    offsets: ArrayLike,
    cell_types: ArrayLike,
    cell_data: _Data | None = None,
    point_data: _Data | None = None,
    comments: list[str] | None = None,
) -> str:
    """将非结构网格及关联数据写入 VTK 文件。

    ``x``、``y``、``z`` 保存顶点坐标；``connectivity``、``offsets`` 和
    ``cell_types`` 描述单元拓扑。``cellData`` 与 ``pointData`` 分别提供单元和
    顶点数据。返回生成文件的绝对路径。
    """
    if not (x.size == y.size == z.size):
        raise ValueError("x、y、z 的长度必须相同")
    x = _convert_list_to_array(x)
    y = _convert_list_to_array(y)
    z = _convert_list_to_array(z)
    connectivity = _convert_list_to_array(connectivity)
    offsets = _convert_list_to_array(offsets)
    cell_types = _convert_list_to_array(cell_types)
    cell_data = _convert_dict_lists_to_arrays(cell_data)
    point_data = _convert_dict_lists_to_arrays(point_data)

    npoints = x.size
    ncells = cell_types.size
    if offsets.size != ncells:
        raise ValueError("offsets 与 cell_types 的长度必须相同")
    _validate_data_lengths(cell_data, ncells, "cell_data")
    _validate_data_lengths(point_data, npoints, "point_data")

    w = VtkFile(path, VtkUnstructuredGrid)
    if comments:
        w.addComments(comments)
    w.openGrid()
    w.openPiece(ncells=ncells, npoints=npoints)

    w.openElement("Points")
    w.addData("points", (x, y, z))
    w.closeElement("Points")
    w.openElement("Cells")
    w.addData("connectivity", connectivity)
    w.addData("offsets", offsets)
    w.addData("types", cell_types)
    w.closeElement("Cells")

    _add_data_to_file(w, cell_data=cell_data, point_data=point_data)

    w.closePiece()
    w.closeGrid()
    w.appendData((x, y, z))
    w.appendData(connectivity).appendData(offsets).appendData(cell_types)

    _append_data_to_file(w, cell_data=cell_data, point_data=point_data)

    w.save()
    return w.getFileName()


# ==============================================================================
def cylinder_to_vtk(
    path: str,
    x0: float,
    y0: float,
    z0: float,
    z1: float,
    radius: float,
    nlayers: int,
    npilars: int = 16,
    cell_data: _Data | None = None,
    point_data: _Data | None = None,
    comments: list[str] | None = None,
) -> str:
    """将竖直圆柱侧面写入 VTK 非结构网格。

    ``x0``、``y0`` 指定圆心，``z0``、``z1`` 指定高度范围，``radius`` 指定
    半径。``nlayers`` 是竖直分层数，``npilars`` 是每层圆周点数。
    返回生成文件的绝对路径。
    """
    import math as m

    # 根据极坐标计算 x、y 坐标
    dpi = 2.0 * m.pi / npilars
    angles = np.arange(0.0, 2.0 * m.pi, dpi)

    x = radius * np.cos(angles) + x0
    y = radius * np.sin(angles) + y0

    dz = (z1 - z0) / nlayers
    z = np.arange(z0, z1 + dz, step=dz)

    npoints = npilars * (nlayers + 1)
    ncells = npilars * nlayers

    xx = np.zeros(npoints)
    yy = np.zeros(npoints)
    zz = np.zeros(npoints)

    ii = 0
    for k in range(nlayers + 1):
        for p in range(npilars):
            xx[ii] = x[p]
            yy[ii] = y[p]
            zz[ii] = z[k]
            ii = ii + 1

    # 构造连接关系
    conn = np.zeros(4 * ncells, dtype=np.int64)
    ii = 0
    for layer in range(nlayers):
        for p in range(npilars):
            p0 = p
            if p + 1 == npilars:
                p1 = 0
            else:
                p1 = p + 1  # 闭合圆周

            n0 = p0 + layer * npilars
            n1 = p1 + layer * npilars
            n2 = n0 + npilars
            n3 = n1 + npilars

            conn[ii + 0] = n0
            conn[ii + 1] = n1
            conn[ii + 2] = n3
            conn[ii + 3] = n2
            ii = ii + 4

    # 构造偏移
    offsets = np.zeros(ncells, dtype=np.int64)
    for i in range(ncells):
        offsets[i] = (i + 1) * 4

    # 构造单元类型
    ctype = np.ones(ncells) + VtkPixel.tid

    return unstructured_grid_to_vtk(
        path,
        xx,
        yy,
        zz,
        connectivity=conn,
        offsets=offsets,
        cell_types=ctype,
        cell_data=cell_data,
        point_data=point_data,
        comments=comments,
    )


def _deprecated_api(old_name, new_function):
    """为旧版 CamelCase API 创建兼容包装器。"""

    keyword_names = {
        "cellData": "cell_data",
        "pointData": "point_data",
        "pointsPerLine": "points_per_line",
    }

    def wrapper(*args, **kwargs):
        warnings.warn(
            f"{old_name} 已弃用，请改用 {new_function.__name__}",
            DeprecationWarning,
            stacklevel=2,
        )
        for old_keyword, new_keyword in keyword_names.items():
            if old_keyword in kwargs:
                kwargs[new_keyword] = kwargs.pop(old_keyword)
        return new_function(*args, **kwargs)

    wrapper.__name__ = old_name
    wrapper.__doc__ = f"已弃用；请改用 :func:`{new_function.__name__}`。"
    return wrapper


# 旧名称至少保留一个次版本周期，便于现有调用方平滑迁移。
imageToVTK = _deprecated_api("imageToVTK", image_to_vtk)
rectilinearToVTK = _deprecated_api("rectilinearToVTK", rectilinear_to_vtk)
structuredToVTK = _deprecated_api("structuredToVTK", structured_to_vtk)
gridToVTK = _deprecated_api("gridToVTK", grid_to_vtk)
pointsToVTK = _deprecated_api("pointsToVTK", points_to_vtk)
pointsToVTKAsTIN = _deprecated_api("pointsToVTKAsTIN", points_to_vtk_as_tin)
linesToVTK = _deprecated_api("linesToVTK", lines_to_vtk)
polyLinesToVTK = _deprecated_api("polyLinesToVTK", poly_lines_to_vtk)
unstructuredGridToVTK = _deprecated_api(
    "unstructuredGridToVTK", unstructured_grid_to_vtk
)
cylinderToVTK = _deprecated_api("cylinderToVTK", cylinder_to_vtk)
