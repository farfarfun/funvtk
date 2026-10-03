# funvtk

funvtk 是一个纯 Python VTK XML 文件导出库，可将 NumPy 数组写为结构化和非结构化网格、点集与折线，无需安装 VTK。

## 安装

```bash
pip install funvtk
```

## 使用

```python
from funvtk import points_to_vtk

output = points_to_vtk(
    "points",
    x=[0.0, 1.0],
    y=[0.0, 1.0],
    z=[0.0, 0.0],
    data={"value": [1.0, 2.0]},
)
```

运行后将在当前目录生成 `points.vtu`。

## 已弃用的 API

早期版本使用 CamelCase 命名（如 `pointsToVTK`）。这些名称已弃用，调用时会触发
`DeprecationWarning`，计划在 `2.0.0` 移除，请尽快迁移到对应的 snake_case 新接口：

| 旧名称（已弃用） | 新名称 |
|---|---|
| `imageToVTK` | `image_to_vtk` |
| `rectilinearToVTK` | `rectilinear_to_vtk` |
| `structuredToVTK` | `structured_to_vtk` |
| `gridToVTK` | `grid_to_vtk` |
| `pointsToVTK` | `points_to_vtk` |
| `pointsToVTKAsTIN` | `points_to_vtk_as_tin` |
| `linesToVTK` | `lines_to_vtk` |
| `polyLinesToVTK` | `poly_lines_to_vtk` |
| `unstructuredGridToVTK` | `unstructured_grid_to_vtk` |
| `cylinderToVTK` | `cylinder_to_vtk` |

新接口的关键字参数名也做了同步调整（如 `cellData` → `cell_data`），旧接口仍兼容两种写法。

---

## 关于 farfarfun

[farfarfun](https://github.com/farfarfun) 是一个专注于实用工具库的开源组织，
涵盖云存储、数据处理、AI、多媒体与开发工具链等方向。

- 🏠 组织主页：<https://github.com/farfarfun>
- 📦 PyPI：<https://pypi.org/user/niuliangtao/>
- 📧 联系：farfarfun@qq.com

本项目基于 [MIT](LICENSE) 协议开源。
