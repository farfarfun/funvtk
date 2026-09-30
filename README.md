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

---

## 关于 farfarfun

[farfarfun](https://github.com/farfarfun) 是一个专注于实用工具库的开源组织，
涵盖云存储、数据处理、AI、多媒体与开发工具链等方向。

- 🏠 组织主页：<https://github.com/farfarfun>
- 📦 PyPI：<https://pypi.org/user/niuliangtao/>
- 📧 联系：farfarfun@qq.com

本项目基于 [MIT](LICENSE) 协议开源。
