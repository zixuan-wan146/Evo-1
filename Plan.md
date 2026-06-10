# Evo-1 工程化改造记录

更新时间：2026-06-10

## 当前状态

- 实际 Git 仓库位于 `/home/myser/Project/Evo/Evo-1`。
- 外层 `/home/myser/Project/Evo` 不是有效 Git 仓库，里面的 `.git` 是空目录。
- 当前工作区已有未提交改动，工程改造已按用户要求暂停。
- 本地环境目前没有安装 `torch`，因此尚不能做完整训练、推理、模型加载或 GPU 验证。

## 已完成的改动

### 工程配置与轻量测试骨架

- 新增 `pyproject.toml`
  - 配置 `pytest` 的测试路径和 `pythonpath`。
  - 配置 `ruff` 的基础 lint 规则。
- 新增 `requirements-dev.txt`
  - 包含轻量开发依赖：`numpy`、`pytest`、`ruff`。
- 新增 `.github/workflows/ci.yml`
  - GitHub Actions 轻量 CI。
  - 安装开发依赖。
  - 运行单元测试。
  - 运行 Python 源码编译检查。
- 新增 `Evo_1/__init__.py`
  - 让 `Evo_1` 成为明确的 Python package。
- 新增 `tests/test_runtime_config.py`
  - 覆盖 `pad_1d`、`build_action_mask`、`normalize_mask` 的基础行为。
- 新增 `tests/test_flow_matching_config.py`
  - 覆盖 `FlowmatchingActionHead` 无 `config` 构造路径。
  - 如果本地没有 `torch`，该测试会自动跳过。

### 已修复的代码问题

- `Evo_1/model/action_head/flow_matching.py`
  - 修复 `FlowmatchingActionHead(config=None, ...)` 时访问 `config.per_action_dim` 和 `config.action_dim` 导致失败的问题。
  - 将 `per_action_dim` 写入默认 `SimpleNamespace`。
  - 增加 `action_dim == horizon * per_action_dim` 的显式校验。

- `Evo_1/scripts/train.py`
  - 在创建 dataloader 前检查 dataset 是否为空。
  - 在 `drop_last=True` 导致 dataloader 没有 batch 时显式报错。
  - 训练 forward 时传入 `embodiment_ids`，避免多 embodiment 配置下该信息被静默丢弃。
  - 将 `assert pred_velocity.shape == target_velocity.shape` 改成显式 `ValueError`。

- `Evo_1/scripts/Evo1_server.py`
  - 将请求 shape 校验从 `assert` 改成运行时 `ValueError`。
  - 修复 `with torch.no_grad() and torch.amp.autocast(...)` 的上下文管理写法，改为 `with torch.no_grad(), autocast_context:`。
  - CPU 设备下不再强行启用 CUDA autocast。

- `Evo_1/dataset/simulation_dataset.py`
  - 将样本读取失败时的无限递归式重试改为有限重试。
  - 新增 `max_sample_retries = 10`。
  - 抽出 `_load_sample()`，读取 cache 和视频失败后由 `__getitem__()` 统一处理。

## 尚未完成但计划继续做的事情

### 评估脚本参数化

- `LIBERO_evaluation/libero_client_4tasks.py`
  - 将 `SERVER_URL` 默认值从 `ws://0.0.0.0:9000` 改成更适合作为客户端目标的 `ws://127.0.0.1:9000`。
  - 支持通过环境变量覆盖 `SERVER_URL`、episode 数、seed、日志目录、视频目录等。
  - 减少 `print()`，统一使用 logging。

- `MetaWorld_evaluation/mt50_evo1_client_prompt.py`
  - 支持通过环境变量覆盖 server URL、episode 数、horizon、seed、是否保存视频、日志目录等。
  - 保留默认行为，避免破坏 README 中的复现路径。

### 训练脚本稳健性

- 将 CUDA/bfloat16 autocast 根据 `--device` 自动选择，避免 CPU 环境或非 CUDA 环境直接失败。
- 检查 checkpoint 保存逻辑中对 `model_engine.save_checkpoint()` 的假设。
- 对 `resume_path`、`dataset_config_path`、`save_dir` 做更明确的路径校验和错误信息。
- 考虑把训练配置从长命令迁移到 YAML/JSON 配置文件，减少复现实验命令长度。

### Dataset 与缓存治理

- 对 cache 版本加入元信息，避免 dataset 配置变化后复用旧 cache。
- 检查 `_pad_tensor()` 是否需要显式拒绝超过 `max_dim` 的数据，而不是运行时 tensor assignment 报错。
- 对视频读取失败、缺失 view、timestamp 越界增加更清楚的错误上下文。

### 依赖与复现

- 细化 `requirements.txt` 中的浮动依赖版本。
- 拆分运行依赖、训练依赖、评估依赖和开发依赖。
- 补充服务器环境说明，包括 CUDA、PyTorch、flash-attn、DeepSpeed 的匹配版本。

### 测试与验证

- 本地轻量验证：
  - `python3 -m pytest`
  - `python3 -m compileall -q Evo_1 MetaWorld_evaluation LIBERO_evaluation`
  - `python3 -m ruff check .`
- 服务器重型验证：
  - 安装完整依赖。
  - 下载 checkpoint。
  - 启动 `Evo1_server.py`。
  - 跑一条 MetaWorld smoke eval。
  - 跑一条 LIBERO smoke eval。
  - 如有数据集，跑短步数训练 smoke test。

## 服务器建议

当前不必须立刻租服务器。建议先完成本地工程化改造和轻量验证，再用服务器做重型验证。

推荐服务器配置：

- Ubuntu 22.04 或接近环境。
- Python 3.10。
- CUDA 与 PyTorch 2.5.1 匹配。
- GPU 显存 24GB 起步，48GB 更稳；如果要全量微调 VLM，80GB 更合适。
- 磁盘至少 100GB。
- 网络可访问 GitHub、Hugging Face、PyPI。

## 恢复工作时的建议顺序

1. 先运行轻量测试和编译检查，确认当前未提交改动没有语法问题。
2. 继续完成评估脚本参数化。
3. 再处理训练脚本的设备/autocast/checkpoint 边界。
4. 更新 README，记录新的开发验证命令和服务器验证流程。
5. 在服务器上做完整依赖安装和 smoke test。
