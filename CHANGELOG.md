# Changelog

## 未发布

### 修复

- 弃用警告补充计划移除版本号（`2.0.0`），README 新增 CamelCase 旧 API 到 snake_case
  新 API 的迁移对照表。
- 兼容包装器 `_deprecated_api`/`wrapper` 补充类型标注。
- 示例中残留的英文说明性注释改为中文（许可证原文保留）。
- 冒烟测试改用固定种子的 `numpy.random.default_rng(0)`，避免随机输入导致偶发失败。
- `examples/poly_lines.py` 的单元数据 `vel` 长度与折线数（`ncells`）不一致，
  触发新增的长度校验后报错；改为每条折线一个值，和 `points_per_line` 对齐。

### 废弃

- `imageToVTK`/`rectilinearToVTK`/`structuredToVTK`/`gridToVTK`/`pointsToVTK`/
  `pointsToVTKAsTIN`/`linesToVTK`/`polyLinesToVTK`/`unstructuredGridToVTK`/
  `cylinderToVTK` 计划在 `2.0.0` 移除，请改用对应的 snake_case 新接口（详见 README）。

## 1.0.10（未发布）

### 新增

- 增加 `uv.lock`、Ruff 配置及公开 API 正常路径测试。

### 修复

- 修复版本模块导入、规则网格导出和 TIN 导出返回值。
- 示例不再静默吞掉文件删除及执行异常。

### 变更

- 构建后端由 Poetry 迁移至 Hatchling，依赖管理迁移至 uv，Python 下限调整为 3.10。
- 补充公开 API 的类型标注、中文文档及项目说明。

### 废弃

（无）

## 1.0.9 及更早版本

早期版本未维护 CHANGELOG，具体变更参见 git 提交历史。
