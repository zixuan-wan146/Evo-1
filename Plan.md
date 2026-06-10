# Evo-1 工程化改造记录

更新时间：2026-06-11（服务器时间）

## 当前状态

- 本地仓库：`/home/myser/Project/Evo/Evo-1`
- 服务器仓库：`Evo:/root/autodl-tmp/Evo-1`
- 外层目录 `/home/myser/Project/Evo` 不是有效 Git 仓库，里面的 `.git` 是空目录。
- 本地已创建 SSH host：
  - `Host Evo`
  - `HostName connect.bjb2.seetacloud.com`
  - `Port 53983`
  - `User root`
  - `IdentityFile /home/myser/.ssh/id_ed25519_autodl`
- GitHub 推送状态：
  - 本地提交已完成，但本地 `main` 仍领先 `origin/main`。
  - `git push origin main` 失败，原因是当前凭据 `myserendipity137` 没有 `zixuan-wan146/Evo-1.git` 写权限。
  - 需要给该账号写权限，或提供有权限的新 remote。
- 服务器状态：
  - LIBERO 部署和 smoke 已完成。
  - 服务器已按要求关机，后续 SSH 检查表现为 connection refused。
- 下载源说明：
  - 服务器部署时曾按环境建议使用 Hugging Face mirror 加速模型/assets 下载。
  - 后续不再修改全局下载源；如需临时加速，应限制在单条命令或单个环境变量内，避免影响国内资源下载。

## 已完成提交

本地提交：

- `fb2ffad Improve engineering baseline for Evo-1`
- `c97cfa8 Document remote setup and checkpoint loading`
- `a2888f2 Add practical lint gate`
- `Complete LIBERO remote smoke setup`（本轮新增，记录 LIBERO 环境、assets、smoke 结果和客户端修复）
- `Add reproducible LIBERO workflow scripts`
- `Add repository preflight checks`
- `Harden flow matching shape validation`
- `Test LIBERO client configuration`
- `Validate LIBERO action responses`
- `Test training configuration validation`
- `Make dataset config path resolution explicit`（本轮新增，记录训练数据配置路径解析规则）

服务器对应提交：

- `8e7edc8 Improve engineering baseline for Evo-1`
- `3fb5b49 Document remote setup and checkpoint loading`
- `ee64c12 Add practical lint gate`
- `Complete LIBERO remote smoke setup`（服务器本轮新增，内容与本地等价）

服务器提交哈希不同是因为通过 `git format-patch | git am` 应用，内容等价但提交对象不同。
最新一次 `git push origin main` 仍失败：当前 GitHub 凭据 `myserendipity137` 对 `zixuan-wan146/Evo-1.git` 没有写权限。

## 已完成的工程改造

### 工程配置与 CI

- 新增 `pyproject.toml`
  - 配置 `pytest`。
  - 配置第一阶段实用 `ruff` 门禁：`E`/`F`，忽略长行。
- 新增 `requirements-dev.txt`
  - `numpy`
  - `pytest`
  - `ruff`
- 新增 `.github/workflows/ci.yml`
  - 安装轻量开发依赖。
  - 运行 `pytest`。
  - 运行 `ruff check .`。
  - 运行 `compileall`。
- 新增 `Evo_1/__init__.py`。
- 新增基础测试：
  - `tests/test_runtime_config.py`
  - `tests/test_flow_matching_config.py`

### 代码稳健性修复

- `Evo_1/model/action_head/flow_matching.py`
  - 修复 `FlowmatchingActionHead(config=None, ...)` 构造路径。
  - 显式校验 `action_dim == horizon * per_action_dim`。
  - 将 action sequence horizon 与训练 `action_mask` shape 校验从 `assert` 改为显式 `ValueError`。

- `Evo_1/scripts/train.py`
  - 增加训练配置校验。
  - 将训练配置校验抽到 `Evo_1/training_config.py`，可在无 torch/accelerate 的本地环境中测试。
  - 新增 `--dataset_config_base_dir`，训练数据 YAML 内的相对路径不再依赖启动目录。
  - 检查空 dataset 和空 dataloader。
  - 训练 forward 时传入 `embodiment_ids`。
  - 将 shape `assert` 改为显式 `ValueError`。
  - autocast 根据设备选择，非 CUDA 不强制 CUDA autocast。
  - checkpoint 保存/恢复处显式要求 DeepSpeed checkpoint support。
  - 非有限数值检测改为显式 `FloatingPointError`。

- `Evo_1/scripts/Evo1_server.py`
  - 增加设备可用性校验。
  - 请求字段和 shape 校验从 `assert` 改为运行时错误。
  - 修复 `torch.no_grad()` 与 autocast 上下文写法。
  - 单请求失败时通过 websocket 返回 JSON error。
  - `torch.load(..., weights_only=False)` 显式声明 DeepSpeed checkpoint 可信加载假设。

- `Evo_1/dataset/simulation_dataset.py`
  - 样本失败读取从无限递归改为有限重试。
  - 空 dataset 显式报错。
  - `_pad_tensor()` 显式拒绝超过 `max_dim` 的张量。

- `Evo_1/dataset/config_utils.py`
  - 新增 dataset config 结构校验。
  - 新增 dataset path 解析工具。
  - 训练入口和 preflight 共用同一套路径解析逻辑。

### 评估脚本配置化

- `LIBERO_evaluation/libero_client_4tasks.py`
  - 兼容 Python 3.8 类型注解。
  - 将 `MUJOCO_GL` 设置提前到 LIBERO/robosuite 导入前。
  - 默认使用 `osmesa` headless 渲染，可通过 `EVO1_MUJOCO_GL=egl` 切换到 EGL。
  - 将环境变量解析抽到 `LIBERO_evaluation/libero_client_config.py`，可在无 LIBERO/robosuite 的本地环境中测试。
  - 将 websocket action 响应解析抽到 `LIBERO_evaluation/libero_action_protocol.py`，显式校验服务端 error、horizon 和动作维度。
  - 客户端默认地址从 `ws://0.0.0.0:9000` 修正为 `ws://127.0.0.1:9000`。
  - 支持单值 `EVO1_LIBERO_MAX_STEPS` 自动扩展到多个 task suite。
  - 新增 `EVO1_LIBERO_TASK_LIMIT`，用于轻量 smoke 测试只跑前 N 个 task。
  - 支持环境变量：
    - `EVO1_SERVER_URI`
    - `EVO1_LIBERO_SERVER_URL`
    - `EVO1_MUJOCO_GL`
    - `EVO1_LIBERO_HORIZON`
    - `EVO1_LIBERO_MAX_STEPS`
    - `EVO1_LIBERO_TASK_SUITES`
    - `EVO1_LIBERO_TASK_LIMIT`
    - `EVO1_LIBERO_EPISODES`
    - `EVO1_LIBERO_SEED`
    - `EVO1_LIBERO_LOG_DIR`
    - `EVO1_LIBERO_VIDEO_DIR`
    - `EVO1_LIBERO_LOG_FILE`

- `MetaWorld_evaluation/mt50_evo1_client_prompt.py`
  - 支持环境变量：
    - `EVO1_SERVER_URI`
    - `EVO1_MT50_SERVER_URL`
    - `EVO1_MT50_EPISODES`
    - `EVO1_MT50_EPISODE_HORIZON`
    - `EVO1_MT50_HORIZON`
    - `EVO1_MT50_SEED`
    - `EVO1_MT50_SAVE_VIDEO`
    - `EVO1_MT50_LOG_DIR`
    - `EVO1_MT50_VIDEO_DIR`
    - `EVO1_MT50_TARGET_LEVEL`
    - `EVO1_MAX_MESSAGE_SIZE`

### README

- 新增本地开发检查说明。
- 新增 MetaWorld/LIBERO 环境变量使用示例。
- 新增远程服务器部署说明。
- 新增训练数据路径解析说明：`dataset/config.yaml` 内的相对路径从 `--dataset_config_base_dir` 解析。
- 记录 `HF_HOME`、`HUGGINGFACE_HUB_CACHE`、`PIP_CACHE_DIR`、`TMPDIR` 等数据盘路径建议。
- 记录 `flash-attn` cross-device link 安装问题的处理方式。

### 复现脚本

- 新增 `scripts/preflight.py`
  - 默认检查仓库关键文件、脚本执行权限、shell 脚本语法和数据配置结构。
  - 可选检查 checkpoint 目录：`--checkpoint /path/to/Evo1_LIBERO`。
  - 可选检查运行时依赖：`--check-imports evo1|libero|all`。
  - 可选严格检查数据文件：`--strict-data`。
- 新增 `scripts/setup_libero_env.sh`
  - 创建/复用 LIBERO Python 3.8.13 conda prefix env。
  - 安装 `libero==0.1.1`、`websockets`、`imageio`。
  - 下载 LIBERO assets 到数据盘。
  - 写入 `~/.libero/config.yaml`。
  - 将包内 `assets` 软链接到数据盘 assets 目录。
  - root + apt 环境下自动安装 `libegl1`、`libosmesa6`、`libglu1-mesa`。
- 新增 `scripts/start_evo1_server.sh`
  - 用环境变量或参数启动 Evo1 websocket server。
  - 支持 `EVO1_PYTHON`、`EVO1_CKPT_DIR`、`EVO1_HOST`、`EVO1_PORT`、`EVO1_DEVICE`、`EVO1_INFERENCE_STEPS`。
- 新增 `scripts/run_libero_smoke.sh`
  - 固化 1 task / 1 episode / 1 step 的 LIBERO smoke 默认配置。
  - 支持通过环境变量扩展到更长 eval。
- CI 新增 shell 脚本语法检查、`scripts/preflight.py` 和 `compileall scripts`。

## 服务器部署状态

服务器硬件：

- GPU：NVIDIA GeForce RTX 4090 D
- 显存：24564 MiB
- 数据盘：`/root/autodl-tmp`，50GB

已安装：

- Miniforge：`/root/autodl-tmp/miniforge3`
- Conda env：`Evo1`
- Python：3.10.20
- Torch：2.5.1+cu124
- Torchvision：0.20.1
- flash-attn：2.8.3
- 其余 `Evo_1/requirements.txt` 依赖
- 轻量开发依赖：`pytest`、`ruff`

LIBERO 评估环境：

- Conda prefix env：`/root/autodl-tmp/envs/libero`
- Python：3.8.13
- 安装包：`libero==0.1.1`、`robosuite==1.4.0`、`robomimic==0.2.0`、`mujoco==3.2.3`、`websockets`、`imageio`
- 系统 headless 渲染库：`libegl1`、`libosmesa6`、`libglu1-mesa`
- LIBERO assets：`/root/autodl-tmp/libero/assets`
- LIBERO datasets 目录：`/root/autodl-tmp/libero/datasets`
- LIBERO 配置：`/root/.libero/config.yaml`
- 已将 Python 包内 assets 路径软链接到数据盘：`site-packages/libero/libero/assets -> /root/autodl-tmp/libero/assets`
- 说明：官方 GitHub 仓库在服务器上 full/sparse clone 速度很慢；本次使用 PyPI 的 `libero==0.1.1`，再从 Hugging Face 下载 assets。

关键环境变量建议：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf-home
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf-cache
export PIP_CACHE_DIR=/root/autodl-tmp/pip-cache
export TMPDIR=/root/autodl-tmp/tmp
```

已下载 checkpoint：

- `/root/autodl-tmp/checkpoints/Evo1_MetaWorld`
- `/root/autodl-tmp/checkpoints/Evo1_LIBERO`

已缓存基座模型：

- `OpenGVLab/InternVL3-1B`，缓存位于 `/root/autodl-tmp/hf-cache`

## 已验证

本地验证：

```bash
python3 -m pytest
python3 scripts/preflight.py
bash -n scripts/*.sh
PYTHONPYCACHEPREFIX=/tmp/evo1_pycache python3 -m compileall -q Evo_1 MetaWorld_evaluation LIBERO_evaluation scripts tests
git diff --check
```

本地结果：

- `pytest`：34 passed, 3 skipped
- `scripts/preflight.py`：通过；仅提示默认训练数据路径不存在的 WARN（本地未放完整训练数据，非失败）
- `bash -n scripts/*.sh`：通过
- `compileall`：通过
- `git diff --check`：通过
- `python3 -m ruff check .`：未运行成功，本地 Python 环境未安装 `ruff`

服务器验证：

```bash
python -m pytest
python -m ruff check .
python -m compileall -q Evo_1 MetaWorld_evaluation LIBERO_evaluation tests
```

服务器结果：

- `pytest`：6 passed
- `ruff check .`：通过
- `compileall`：通过
- `torch.cuda.is_available()`：True
- GPU：NVIDIA GeForce RTX 4090 D

重型 smoke 验证：

- MetaWorld checkpoint 加载成功。
- LIBERO checkpoint 加载成功。
- 使用 MetaWorld checkpoint 跑 dummy JSON inference 成功。
- 输出动作维度：`50 x 24`。
- LIBERO 真实环境 smoke 成功：
  - Evo1 websocket 服务：`ws://127.0.0.1:9000`
  - checkpoint：`/root/autodl-tmp/checkpoints/Evo1_LIBERO`
  - suite：`libero_spatial`
  - task limit：1
  - episodes：1
  - max steps：1
  - horizon：1
  - 渲染后端：`osmesa`
  - 输出视频：`/root/autodl-tmp/evo1_libero_smoke/videos_osmesa/libero_spatial/task1_episode1.mp4`
  - 结果：流程跑通，任务失败是预期的，因为只执行 1 个决策步用于 smoke。

## 尚未完成

- GitHub push 尚未完成，需要仓库写权限或新的目标 remote。
- MetaWorld 已按用户要求停止，不再继续安装或评估。
- 尚未跑完整 LIBERO suite eval，目前只跑了 1 task/1 episode/1 step 的 smoke。
- 尚未下载训练数据集并跑短步数训练 smoke test。
- `requirements.txt` 仍有部分浮动依赖，后续可进一步锁定版本。
- `flash-attn` 目前按服务器实际环境安装成功，尚未写入主 requirements，避免无 GPU/无匹配 wheel 环境被强绑定。

## 下一步建议

1. 处理 GitHub 权限：
   - 给 `myserendipity137` 写权限，或
   - 添加一个有写权限的新 remote。
2. 需要评估效果时，在服务器跑完整 LIBERO eval，例如逐个 suite 设置合理 `EVO1_LIBERO_MAX_STEPS` 和 `EVO1_LIBERO_HORIZON=14`。
3. 下载小规模训练数据或抽样数据，跑 `max_steps=1` 到 `10` 的训练 smoke test。
4. 根据真实 eval/training 结果继续修复工程问题。
5. 在下一次服务器可用时，用新脚本从空环境重跑一次 LIBERO 安装和 smoke，确认脚本级端到端复现。
