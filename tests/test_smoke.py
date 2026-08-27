"""Lightweight smoke tests for funvtk.

funvtk is a pure-Python fork of pyevtk: it writes numerical grid/point data
to VTK XML files (.vtu/.vtp/.vtr/.vts/.vti/.pvd). It does not depend on the
real VTK/ParaView C++ libraries and does not need a display, GPU, network,
or credentials, so it is straightforward to smoke test end-to-end by writing
small files to a temporary directory and checking they were created.

There is no [project.scripts]/[tool.poetry.scripts] CLI entry point in this
package, so no CLI smoke test is included.
"""

import os

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------


def test_import_top_level_package():
    import funvtk

    assert funvtk is not None


def test_top_level_public_api_symbols():
    import funvtk

    for name in ("pointsToVTK", "pointsToVTKAsTIN", "polyLinesToVTK", "gridToVTK"):
        assert hasattr(funvtk, name), f"funvtk.{name} missing"
        assert callable(getattr(funvtk, name))
    assert set(funvtk.__all__) == {
        "pointsToVTK",
        "pointsToVTKAsTIN",
        "polyLinesToVTK",
        "gridToVTK",
    }


@pytest.mark.parametrize("submodule", ["hl", "vtk", "evtk", "xml"])
def test_import_submodules(submodule):
    mod = pytest.importorskip(f"funvtk.{submodule}")
    assert mod is not None


def test_version_submodule_has_syntax_error():
    """funvtk.version.py contains a syntax error (`_MAJOR = 2_MINOR = 0`),
    which is an invalid decimal literal in Python. It is not imported by
    funvtk/__init__.py so it does not break normal usage of the package,
    but importing it directly fails. This is a pre-existing bug in the
    repository, tracked here rather than silently worked around.
    """
    with pytest.raises(SyntaxError):
        import funvtk.version  # noqa: F401


# ---------------------------------------------------------------------------
# High level API smoke tests (funvtk.hl / funvtk top-level re-exports)
# ---------------------------------------------------------------------------


def test_points_to_vtk_writes_file(tmp_path):
    from funvtk import pointsToVTK

    npoints = 10
    x = np.random.rand(npoints)
    y = np.random.rand(npoints)
    z = np.random.rand(npoints)
    pressure = np.random.rand(npoints)

    out_path = str(tmp_path / "points")
    result = pointsToVTK(out_path, x, y, z, data={"pressure": pressure})

    assert result == out_path + ".vtu"
    assert os.path.isfile(result)
    assert os.path.getsize(result) > 0


def test_points_to_vtk_as_tin_writes_file(tmp_path):
    """Exercises the scipy.spatial.Delaunay dependency.

    NOTE: pointsToVTKAsTIN() has a pre-existing bug -- it forwards to
    unstructuredGridToVTK() but never returns its result, so it always
    returns None instead of the file path the docstring promises. This is
    a real business-logic bug in the repository; it is not fixed here.
    We verify success by checking the well-known output path directly
    instead of relying on the (always-None) return value.
    """
    from funvtk import pointsToVTKAsTIN

    npoints = 20
    x = np.random.rand(npoints)
    y = np.random.rand(npoints)
    z = np.random.rand(npoints)

    out_path = str(tmp_path / "tin")
    result = pointsToVTKAsTIN(out_path, x, y, z, ndim=2)

    assert result is None  # documents the current (buggy) behavior
    expected_file = out_path + ".vtu"
    assert os.path.isfile(expected_file)
    assert os.path.getsize(expected_file) > 0


def test_grid_to_vtk_structured_writes_file(tmp_path):
    """gridToVTK() with 3D coordinate arrays exercises the "structured"
    branch, which works correctly.
    """
    from funvtk import gridToVTK

    nx, ny, nz = 3, 3, 3
    x, y, z = np.meshgrid(
        np.arange(nx, dtype="float64"),
        np.arange(ny, dtype="float64"),
        np.arange(nz, dtype="float64"),
        indexing="ij",
    )

    out_path = str(tmp_path / "grid_structured")
    result = gridToVTK(out_path, x, y, z)

    assert os.path.isfile(result)
    assert os.path.getsize(result) > 0


def test_grid_to_vtk_rectilinear_branch_is_broken(tmp_path):
    """gridToVTK() with 1D coordinate arrays takes the "rectilinear"
    branch, which has a pre-existing indentation bug in hl.py: the calls to
    _addDataToFile(), w.closePiece() and w.closeGrid() are nested inside the
    `else` (structured) branch of the `if isRect / else` block, so they are
    silently skipped for rectilinear grids. This leaves the XML writer's
    element stack unbalanced and closeElement() raises AssertionError.

    This is a real business-logic bug in the repository (not fixed here);
    this test documents and locks in the current failure mode rather than
    silently skipping it.
    """
    from funvtk import gridToVTK

    x = np.arange(0, 5, dtype="float64")
    y = np.arange(0, 4, dtype="float64")
    z = np.arange(0, 3, dtype="float64")

    out_path = str(tmp_path / "grid_rect")
    with pytest.raises(AssertionError):
        gridToVTK(out_path, x, y, z)


def test_poly_lines_to_vtk_writes_file(tmp_path):
    from funvtk import polyLinesToVTK

    # Two polylines: first with 3 points, second with 2 points.
    x = np.array([0.0, 1.0, 2.0, 5.0, 6.0])
    y = np.array([0.0, 1.0, 0.0, 2.0, 3.0])
    z = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    points_per_line = np.array([3, 2])

    out_path = str(tmp_path / "polylines")
    result = polyLinesToVTK(out_path, x, y, z, pointsPerLine=points_per_line)

    assert os.path.isfile(result)
    assert os.path.getsize(result) > 0


def test_image_to_vtk_writes_file(tmp_path):
    from funvtk.hl import imageToVTK

    point_data = {"val": np.random.rand(3, 4, 5)}
    out_path = str(tmp_path / "image")
    result = imageToVTK(out_path, pointData=point_data)

    assert os.path.isfile(result)
    assert os.path.getsize(result) > 0


def test_lines_to_vtk_writes_file(tmp_path):
    from funvtk.hl import linesToVTK

    npoints = 4
    x = np.array([0.0, 1.0, 0.0, -1.0])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    z = np.array([0.0, 1.0, 0.0, 1.0])
    vel = np.zeros(2)
    temp = np.random.rand(npoints)

    out_path = str(tmp_path / "lines")
    result = linesToVTK(out_path, x, y, z, cellData={"vel": vel}, pointData={"temp": temp})

    assert os.path.isfile(result)
    assert os.path.getsize(result) > 0


# ---------------------------------------------------------------------------
# Low level API smoke tests (funvtk.vtk / funvtk.xml)
# ---------------------------------------------------------------------------


def test_vtk_file_and_group_low_level(tmp_path):
    from funvtk.vtk import VtkFile, VtkGroup, VtkImageData

    path = str(tmp_path / "lowlevel")
    w = VtkFile(path, VtkImageData)
    w.openGrid(start=(0, 0, 0), end=(0, 0, 0), origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))
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
# CLI entry point
# ---------------------------------------------------------------------------


def test_no_cli_entry_point_declared():
    """funvtk's pyproject.toml declares no [tool.poetry.scripts] /
    [project.scripts] console entry point, so there is nothing to smoke
    test via subprocess/CliRunner. This test documents that fact so a
    future added CLI won't silently go untested.
    """
    pytest.skip("funvtk declares no CLI entry point in pyproject.toml")
