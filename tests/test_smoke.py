"""funvtk 的轻量级冒烟测试。

funvtk 是 pyevtk 的纯 Python 分支，用于将数值网格和点数据写入 VTK XML 文件。
它不依赖 VTK/ParaView 的 C++ 库，也不需要显示设备、GPU、网络或凭据，因此可通过
向临时目录写入小文件来完成端到端测试。
"""

import os

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 导入测试
# ---------------------------------------------------------------------------


def test_import_top_level_package():
    import funvtk

    assert funvtk is not None


def test_top_level_public_api_symbols():
    import funvtk

    for name in (
        "points_to_vtk",
        "points_to_vtk_as_tin",
        "poly_lines_to_vtk",
        "grid_to_vtk",
    ):
        assert hasattr(funvtk, name), f"funvtk.{name} missing"
        assert callable(getattr(funvtk, name))
    assert set(funvtk.__all__) == {
        "points_to_vtk",
        "points_to_vtk_as_tin",
        "poly_lines_to_vtk",
        "grid_to_vtk",
        "pointsToVTK",
        "pointsToVTKAsTIN",
        "polyLinesToVTK",
        "gridToVTK",
    }


@pytest.mark.parametrize("submodule", ["hl", "vtk", "evtk", "xml"])
def test_import_submodules(submodule):
    mod = pytest.importorskip(f"funvtk.{submodule}")
    assert mod is not None


def test_import_version_submodule():
    from funvtk.version import PYEVTK_VERSION

    assert PYEVTK_VERSION == "2.0.0"


# ---------------------------------------------------------------------------
# 高层 API 冒烟测试
# ---------------------------------------------------------------------------


def test_points_to_vtk_writes_file(tmp_path):
    from funvtk import points_to_vtk

    npoints = 10
    x = np.random.rand(npoints)
    y = np.random.rand(npoints)
    z = np.random.rand(npoints)
    pressure = np.random.rand(npoints)

    out_path = str(tmp_path / "points")
    result = points_to_vtk(out_path, x, y, z, data={"pressure": pressure})

    assert result == out_path + ".vtu"
    assert os.path.isfile(result)
    assert os.path.getsize(result) > 0


def test_points_to_vtk_as_tin_writes_file(tmp_path):
    """覆盖 scipy.spatial.Delaunay 依赖。"""
    from funvtk import points_to_vtk_as_tin

    x = np.array([0.0, 1.0, 1.0, 0.0, 0.5])
    y = np.array([0.0, 0.0, 1.0, 1.0, 0.5])
    z = np.arange(5, dtype="float64")

    out_path = str(tmp_path / "tin")
    result = points_to_vtk_as_tin(out_path, x, y, z, ndim=2)

    assert result == out_path + ".vtu"
    assert os.path.isfile(result)
    assert os.path.getsize(result) > 0


def test_points_to_vtk_as_3d_tin_writes_file(tmp_path):
    from funvtk import points_to_vtk_as_tin

    x = np.array([0.0, 1.0, 0.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 0.0, 1.0])
    z = np.array([0.0, 0.0, 0.0, 1.0, 1.0])

    out_path = str(tmp_path / "tin_3d")
    result = points_to_vtk_as_tin(out_path, x, y, z, ndim=3)

    assert result == out_path + ".vtu"
    assert os.path.getsize(result) > 0


def test_grid_to_vtk_structured_writes_file(tmp_path):
    """三维坐标数组覆盖逻辑结构网格分支。"""
    from funvtk import grid_to_vtk

    nx, ny, nz = 3, 3, 3
    x, y, z = np.meshgrid(
        np.arange(nx, dtype="float64"),
        np.arange(ny, dtype="float64"),
        np.arange(nz, dtype="float64"),
        indexing="ij",
    )

    out_path = str(tmp_path / "grid_structured")
    result = grid_to_vtk(out_path, x, y, z)

    assert os.path.isfile(result)
    assert os.path.getsize(result) > 0


def test_grid_to_vtk_rectilinear_writes_file(tmp_path):
    from funvtk import grid_to_vtk

    x = np.arange(0, 5, dtype="float64")
    y = np.arange(0, 4, dtype="float64")
    z = np.arange(0, 3, dtype="float64")
    point_data = {"value": np.zeros((x.size, y.size, z.size))}

    out_path = str(tmp_path / "grid_rect")
    result = grid_to_vtk(out_path, x, y, z, point_data=point_data)

    assert result == out_path + ".vtr"
    assert os.path.isfile(result)
    assert os.path.getsize(result) > 0


def test_poly_lines_to_vtk_writes_file(tmp_path):
    from funvtk import poly_lines_to_vtk

    # 两条折线：第一条包含 3 个点，第二条包含 2 个点。
    x = np.array([0.0, 1.0, 2.0, 5.0, 6.0])
    y = np.array([0.0, 1.0, 0.0, 2.0, 3.0])
    z = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    points_per_line = np.array([3, 2])

    out_path = str(tmp_path / "polylines")
    result = poly_lines_to_vtk(out_path, x, y, z, points_per_line=points_per_line)

    assert os.path.isfile(result)
    assert os.path.getsize(result) > 0


def test_image_to_vtk_writes_file(tmp_path):
    from funvtk.hl import image_to_vtk

    point_data = {"val": np.random.rand(3, 4, 5)}
    out_path = str(tmp_path / "image")
    result = image_to_vtk(out_path, point_data=point_data)

    assert os.path.isfile(result)
    assert os.path.getsize(result) > 0


def test_lines_to_vtk_writes_file(tmp_path):
    from funvtk.hl import lines_to_vtk

    npoints = 4
    x = np.array([0.0, 1.0, 0.0, -1.0])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    z = np.array([0.0, 1.0, 0.0, 1.0])
    vel = np.zeros(2)
    temp = np.random.rand(npoints)

    out_path = str(tmp_path / "lines")
    result = lines_to_vtk(
        out_path, x, y, z, cell_data={"vel": vel}, point_data={"temp": temp}
    )

    assert os.path.isfile(result)
    assert os.path.getsize(result) > 0


# ---------------------------------------------------------------------------
# 低层 API 冒烟测试
# ---------------------------------------------------------------------------


def test_vtk_file_and_group_low_level(tmp_path):
    from funvtk.vtk import VtkFile, VtkGroup, VtkImageData

    path = str(tmp_path / "lowlevel")
    w = VtkFile(path, VtkImageData)
    w.openGrid(
        start=(0, 0, 0), end=(0, 0, 0), origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0)
    )
    w.openPiece(start=(0, 0, 0), end=(0, 0, 0))
    w.closePiece()
    w.closeGrid()
    w.save()

    assert os.path.isfile(path + ".vti")

    group_path = str(tmp_path / "group")
    group = VtkGroup(group_path)
    group.addFile(path + ".vti", sim_time=0.0)
    group.save()

    assert os.path.isfile(group_path + ".pvd")


def test_xml_writer_low_level(tmp_path):
    from funvtk.xml import XmlWriter

    path = str(tmp_path / "raw.xml")
    xml = XmlWriter(path)
    xml.openElement("Root")
    xml.addAttributes(foo="bar")
    xml.closeElement("Root")
    xml.close()

    assert os.path.isfile(path)
    content = open(path, "rb").read()
    assert b"Root" in content
    assert b"foo" in content


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------


def test_no_cli_entry_point_declared():
    """项目未声明控制台入口，因此没有可执行的 CLI 冒烟测试。"""
    pytest.skip("funvtk declares no CLI entry point in pyproject.toml")
