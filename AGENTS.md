# AGENTS.md

本文件适用于本仓库根目录及其所有子目录。后续任何 agent 或开发者在本仓库工作时，都应先阅读并遵守这里的说明。

不要在本文档、代码或配置中写入个人电脑上的绝对路径。需要表达仓库位置时，统一使用“仓库根目录”或 `<repo-root>`；需要表达本地数据、缓存或输出路径时，优先通过配置文件、命令行参数或环境变量传入。

## 核心目标

本仓库的主要目标是进行 alpha 因子增强：以一个已有 alpha 因子为核心信号，结合多个风格因子、风险暴露或辅助解释变量，构建组合后的增强模型，使最终信号在稳定性、解释性、风险控制或预测效果上优于单一 alpha 因子。

实现时应优先服务于因子研究和建模流程，而不是交易执行系统、展示页或无关平台功能。后续代码应围绕以下问题展开：

- 如何读取、对齐和清洗 alpha 因子与多个风格因子。
- 如何处理日期、股票代码、行业、市值、估值、缺失值和异常值。
- 如何构建训练样本、验证样本和 out-of-sample 评估。
- 如何比较原始 alpha 与增强后 alpha 的效果。
- 如何保存可复现的模型配置、特征组合和评估结果。

## Python 虚拟环境

本项目建议使用 Python 虚拟环境，并优先使用 `uv` 创建和管理环境。虚拟环境目录统一放在仓库根目录下的 `.venv`。

当前仓库在撰写本文件时仍处于早期状态，不能假设已经存在 `pyproject.toml`、`src/`、`tests/` 或 `.venv`。如果本地还没有虚拟环境，先进入仓库根目录：

```bash
cd <repo-root>
uv venv
```

如果需要指定 Python 版本，可使用：

```bash
uv venv --python 3.12
```

创建成功后，`uv` 通常会在仓库根目录生成 `.venv`，并提示对应系统下的激活命令。

## 进入虚拟环境

请根据当前操作系统和 shell 选择对应命令。

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python --version
```

如果 PowerShell 脚本执行策略阻止激活虚拟环境，先在当前 PowerShell 进程中临时放开执行策略，再激活：

```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
.\.venv\Scripts\Activate.ps1
python --version
```

这只影响当前 PowerShell 进程，不会永久修改系统策略。

Windows cmd：

```bat
.\.venv\Scripts\activate.bat
python --version
```

macOS/Linux shell：

```bash
source .venv/bin/activate
python --version
```

## 退出虚拟环境

任意系统下，退出当前虚拟环境：

```bash
deactivate
```

## 不激活环境时运行 Python 命令

如果使用 `uv`，很多命令可以不手动激活虚拟环境，直接通过 `uv run` 在项目环境中执行：

```bash
uv run python --version
uv run pytest
```

后续如果初始化 Python package，优先使用：

```bash
uv init --package
uv add pandas numpy scikit-learn
uv add --dev pytest
uv run pytest
```

## Windows 下 NumPy/Pandas DLL 访问被拒绝

在 Windows PowerShell 或受限执行环境中，即使已经使用仓库根目录下的 `.venv`，也可能在导入 NumPy、Pandas、LightGBM 等带本地扩展的包时遇到类似错误：

```text
ImportError: DLL load failed while importing _multiarray_umath: 拒绝访问。
```

这类错误通常发生在 Python 已经启动、但本地 DLL 被当前进程权限或沙箱策略阻止加载时。它不同于 `Activate.ps1` 被 PowerShell 执行策略阻止；`Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process` 只用于临时放开激活脚本权限，通常不能解决 DLL 加载的“拒绝访问”。

遇到该问题时，先保持同一个 `.venv` 和同一条 Python 命令不变，在当前工具或运行环境中申请最小必要的权限提升后重试。不要第一时间重装 NumPy、Pandas 或重建 `.venv`，除非在非受限权限下仍然稳定复现导入失败。运行评估或训练脚本时，输出仍应限定在仓库根目录下的配置输出目录，例如 `analysis/outputs/...`。

## 开发约定

- Python 代码应使用类型标注。
- 路径处理优先使用 `pathlib.Path`。
- 因子、收益率、股票代码和日期字段必须保持可追踪、可复现。
- 股票代码、指数代码等金融标识不要当作数字处理。
- 缺失值、停牌、退市、极端值和行业/风格暴露应显式处理，不要静默丢弃。
- 模型评估应区分训练集、验证集和样本外测试，避免未来函数和数据泄露。
- 每次实现或修改核心逻辑后，应尽量运行自动化测试或最小 smoke check。

## 推荐验证方式

如果项目已经有测试：

```bash
uv run pytest
```

如果还没有测试框架，至少验证 Python 环境可用：

```bash
uv run python --version
```

或在已激活虚拟环境中运行：

```bash
python --version
```

不要在没有任何验证的情况下宣称功能已经完成。
