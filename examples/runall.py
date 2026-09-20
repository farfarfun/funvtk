import shutil

import group
import image
import lines
import lowlevel
import points
import poly_lines
import rectilinear
import structured
import unstructured


def clean_all():
    group.clean()
    image.clean()
    lines.clean()
    points.clean()
    poly_lines.clean()
    rectilinear.clean()
    structured.clean()
    unstructured.clean()
    lowlevel.clean()
    try:
        shutil.rmtree("__pycache__")
    except FileNotFoundError:
        pass


def test_all():
    group.run()
    image.run()
    lines.run()
    points.run()
    poly_lines.run()
    rectilinear.run()
    structured.run()
    unstructured.run()
    lowlevel.run()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        opt = sys.argv[1]
    else:
        opt = "-"

    if opt == "run":
        test_all()
    elif opt == "clean":
        clean_all()
    else:
        raise SystemExit(f"未知选项：{opt}\n用法：python runall.py [run|clean]")
