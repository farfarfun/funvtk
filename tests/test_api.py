"""高层公开 API 的兼容性、正常路径和输入边界测试。"""

import numpy as np
import pytest

from funvtk import points_to_vtk, pointsToVTK
from funvtk.hl import (
    cylinder_to_vtk,
    grid_to_vtk,
    image_to_vtk,
    lines_to_vtk,
    points_to_vtk_as_tin,
    poly_lines_to_vtk,
    rectilinear_to_vtk,
    structured_to_vtk,
    unstructured_grid_to_vtk,
)


def test_deprecated_api_remains_compatible(tmp_path):
    path = str(tmp_path / "legacy")
    coordinates = np.array([0.0])

    with pytest.warns(DeprecationWarning, match="points_to_vtk"):
        result = pointsToVTK(path, coordinates, coordinates, coordinates)

    assert result == path + ".vtu"


def test_direct_grid_writers(tmp_path):
    axis = np.arange(2.0)
    rectilinear = rectilinear_to_vtk(str(tmp_path / "rectilinear"), axis, axis, axis)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    structured = structured_to_vtk(str(tmp_path / "structured"), x, y, z)

    assert rectilinear.endswith(".vtr")
    assert structured.endswith(".vts")


def test_unstructured_and_cylinder_writers(tmp_path):
    coordinates = np.array([0.0, 1.0, 0.0])
    unstructured = unstructured_grid_to_vtk(
        str(tmp_path / "unstructured"),
        coordinates,
        coordinates,
        coordinates,
        connectivity=np.array([0, 1, 2]),
        offsets=np.array([3]),
        cell_types=np.array([5], dtype="uint8"),
    )
    cylinder = cylinder_to_vtk(
        str(tmp_path / "cylinder"), 0.0, 0.0, 0.0, 1.0, 1.0, 1, 4
    )

    assert unstructured.endswith(".vtu")
    assert cylinder.endswith(".vtu")


def test_image_requires_data(tmp_path):
    with pytest.raises(ValueError, match="cell_data 或 point_data"):
        image_to_vtk(str(tmp_path / "image"))


def test_coordinate_lengths_must_match(tmp_path):
    with pytest.raises(ValueError, match="长度必须相同"):
        points_to_vtk(str(tmp_path / "points"), [0.0], [0.0, 1.0], [0.0])

    with pytest.raises(ValueError, match="长度必须相同"):
        lines_to_vtk(
            str(tmp_path / "lines"),
            np.zeros(2),
            np.zeros(4),
            np.zeros(2),
        )


def test_point_data_length_must_match(tmp_path):
    coordinates = np.arange(3.0)
    with pytest.raises(ValueError, match=r"data\['value'\].*必须为 3"):
        points_to_vtk(
            str(tmp_path / "points"),
            coordinates,
            coordinates,
            coordinates,
            data={"value": np.arange(2.0)},
        )


def test_tin_dimension_must_be_supported(tmp_path):
    coordinates = np.arange(4.0)
    with pytest.raises(ValueError, match="ndim 必须为 2 或 3"):
        points_to_vtk_as_tin(
            str(tmp_path / "tin"), coordinates, coordinates, coordinates, ndim=4
        )


def test_grid_dimensions_and_shapes_must_match(tmp_path):
    with pytest.raises(ValueError, match="一维数组或全部是三维数组"):
        grid_to_vtk(
            str(tmp_path / "grid"), np.zeros(2), np.zeros((2, 2)), np.zeros(2)
        )

    with pytest.raises(ValueError, match="形状必须相同"):
        structured_to_vtk(
            str(tmp_path / "structured"),
            np.zeros((2, 2, 2)),
            np.zeros((2, 2, 3)),
            np.zeros((2, 2, 2)),
        )


def test_topology_lengths_must_match(tmp_path):
    coordinates = np.arange(3.0)
    with pytest.raises(ValueError, match="总和必须等于坐标数量"):
        poly_lines_to_vtk(
            str(tmp_path / "polyline"),
            coordinates,
            coordinates,
            coordinates,
            points_per_line=np.array([2]),
        )

    with pytest.raises(ValueError, match="offsets 与 cell_types"):
        unstructured_grid_to_vtk(
            str(tmp_path / "unstructured"),
            coordinates,
            coordinates,
            coordinates,
            connectivity=np.array([0, 1, 2]),
            offsets=np.array([], dtype="int32"),
            cell_types=np.array([5], dtype="uint8"),
        )
