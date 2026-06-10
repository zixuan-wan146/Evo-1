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
  - 本地提交已完成，但本地 `main` 仍领先 `origin/main` 30 个提交。
  - 最新本地提交主题：`Validate LIBERO run manifests`。
  - 最新未推送补丁包：`exports/unpushed_commits_20260610T185000Z`。
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
- `Record LIBERO evaluation summaries`（本轮新增，记录 LIBERO 评估结果 JSON、失败原因和统计汇总）
- `Summarize LIBERO result files`（本轮新增，聚合多个 LIBERO result JSON 为 Markdown/CSV 对比表）
- `Record LIBERO run metadata`（本轮新增，在 result JSON 和对比表中记录 git/命令/环境元数据）
- `Validate Evo-1 inference requests`（本轮新增，抽出服务端 JSON 请求协议校验并测试）
- `Validate LIBERO result artifacts`（本轮新增，preflight 可检查 LIBERO result JSON 结构）
- `Validate checkpoint metadata`（本轮新增，preflight 可检查 checkpoint config/norm_stats 结构）
- `Preflight checkpoint before server start`（本轮新增，启动 Evo-1 server 前自动运行 checkpoint preflight）
- `Check LIBERO result consistency`（本轮新增，preflight 会校验 summary 与 episode 明细一致）
- `Add full LIBERO eval launcher`（本轮新增，将正式 eval 入口与 smoke 入口分离）
- `Export unpushed commits as patches`（本轮新增，GitHub 权限阻塞时导出可迁移 patch bundle）
- `Pin LIBERO environment entrypoints`（本轮新增，固定 LIBERO 评估环境顶层依赖并加入 setup dry-run）
- `Audit requirements drift`（本轮新增，新增 requirements 策略审计，防止未登记浮动依赖继续扩散）
- `Add consolidated repository check`（本轮新增，用 `scripts/check_repo.sh` 统一本地和 CI 轻量门禁）
- `Validate training dataset structure`（本轮新增，新增训练数据结构验证脚本和 strict-data 结构检查）
- `Add LIBERO checkpoint downloader`（本轮新增，固化 LIBERO checkpoint 下载入口，默认不设置镜像源）
- `Group LIBERO run artifacts`（本轮新增，支持 `EVO1_LIBERO_RUN_DIR` 统一管理 smoke/eval 输出）
- `Record LIBERO run manifests`（本轮新增，LIBERO smoke/eval 启动前写出 run manifest）
- `Validate LIBERO run manifests`（本轮新增，preflight 可校验 run manifest 结构和敏感字段）

服务器对应提交：

- `8e7edc8 Improve engineering baseline for Evo-1`
- `3fb5b49 Document remote setup and checkpoint loading`
- `ee64c12 Add practical lint gate`
- `Complete LIBERO remote smoke setup`（服务器本轮新增，内容与本地等价）

服务器提交哈希不同是因为通过 `git format-patch | git am` 应用，内容等价但提交对象不同。
最新一次 `git push origin main` 在提交 `Validate LIBERO run manifests` 后仍失败：当前 GitHub 凭据 `myserendipity137` 对 `zixuan-wan146/Evo-1.git` 没有写权限。

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
  - 使用 `Evo_1/server_protocol.py` 统一校验 websocket JSON 请求。
  - 修复 `torch.no_grad()` 与 autocast 上下文写法。
  - 单请求失败时通过 websocket 返回 JSON error。
  - `torch.load(..., weights_only=False)` 显式声明 DeepSpeed checkpoint 可信加载假设。

- `Evo_1/server_protocol.py`
  - 新增纯 Python 请求协议校验，单测不依赖 torch/cv2/model。
  - 校验必填字段、3 路 RGB 图像、0..255 有限像素、有限 state、0/1 mask、mask 非全空和自动 padding。

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
  - 评估后写出机器可读 JSON summary，包含 episode 明细、suite 成功率和失败原因。
  - 动作解析失败、非法动作、步数耗尽都会记录明确 failure reason。
  - 每个 task 的 LIBERO env 用完后显式 close，避免长评估资源泄漏。
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
    - `EVO1_LIBERO_RESULT_FILE`

- `LIBERO_evaluation/libero_eval_summary.py`
  - 新增 `EpisodeResult` 结构。
  - 新增成功率、平均步数、suite 分组统计。
  - 新增结果 JSON 写盘函数，可在无 LIBERO/robosuite 的本地环境中测试。
  - 新增运行元数据：UTC 时间、命令、Python 版本、平台、git commit/branch/dirty 状态、非敏感环境变量。

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
- 新增 Evo-1 websocket inference request JSON 契约说明。
- 新增 LIBERO result summary JSON 的说明和路径覆盖示例。
- 新增 LIBERO 多次运行结果汇总为 Markdown/CSV 的说明。
- 新增 LIBERO result JSON 中运行元数据和汇总表 git 列的说明。
- 新增用 `scripts/preflight.py --libero-result` 校验评估结果文件的说明。
- 新增 LIBERO result 校验会比较 overall/per-suite summary 与 episode 明细一致性的说明。
- 新增用 `scripts/preflight.py --libero-manifest` 校验 LIBERO run manifest 的说明。
- 新增 checkpoint preflight 会检查 `config.json` 关键维度和 `norm_stats.json` min/max 结构的说明。
- 新增 `scripts/start_evo1_server.sh` 默认先跑 checkpoint preflight、可用 `EVO1_SKIP_PREFLIGHT=1` 跳过的说明。
- 新增 `scripts/run_libero_eval.sh` 正式 LIBERO eval 入口和 dry-run 说明。
- 新增 `requirements-libero.txt`，记录 LIBERO 评估环境顶层依赖入口，避免和 Evo1 主环境混用。
- 新增 `scripts/setup_libero_env.sh` dry-run 说明，可在不创建 conda 环境、不下载 assets 的情况下检查路径解析。
- 新增 `requirements-policy.json` 和 `scripts/audit_requirements.py`，要求新增 requirements 文件和未固定依赖必须登记策略理由。
- 新增 `scripts/check_repo.sh`，README 和 CI 都改为使用同一个仓库检查入口。
- 新增 `scripts/validate_training_dataset.py`，训练前可检查 meta、stats、parquet 和视频路径。
- 新增 `scripts/download_libero_checkpoint.sh`，替代手写 `hf download MINT-SJTU/Evo1_LIBERO`。
- 新增 `EVO1_LIBERO_RUN_DIR` 说明，将 LIBERO run 的日志、视频和结果 JSON 归档到同一目录。
- 新增 LIBERO run manifest 说明，评估启动前写出 Git、命令、环境和输出路径上下文。
- 记录 `HF_HOME`、`HUGGINGFACE_HUB_CACHE`、`PIP_CACHE_DIR`、`TMPDIR` 等数据盘路径建议。
- 记录 `flash-attn` cross-device link 安装问题的处理方式。

### 复现脚本

- 新增 `scripts/preflight.py`
  - 默认检查仓库关键文件、脚本执行权限、shell 脚本语法和数据配置结构。
  - 可选检查 checkpoint 目录：`--checkpoint /path/to/Evo1_LIBERO`，包括必备文件、`config.json` 维度和 `norm_stats.json` 结构。
  - 可选检查运行时依赖：`--check-imports evo1|libero|all`。
  - 可选严格检查数据文件：`--strict-data`。
  - 可选检查 LIBERO result JSON 文件、目录或 glob：`--libero-result path_or_glob`，包括 schema 和 summary/episode 一致性。
  - 可选检查 LIBERO run manifest 文件、目录或 glob：`--libero-manifest path_or_glob`，包括 run kind、关键环境变量、Git metadata 和敏感字段过滤。
- 新增 `scripts/setup_libero_env.sh`
  - 创建/复用 LIBERO Python 3.8.13 conda prefix env。
  - 默认从 `requirements-libero.txt` 安装 LIBERO 顶层依赖。
  - 支持 `EVO1_SETUP_LIBERO_DRY_RUN=1` 打印解析后的路径而不创建环境或下载资源。
  - 支持 `EVO1_LIBERO_REQUIREMENTS=/path/to/requirements.txt` 测试替代依赖集合。
  - 下载 LIBERO assets 到数据盘。
  - 写入 `~/.libero/config.yaml`。
  - 将包内 `assets` 软链接到数据盘 assets 目录。
  - root + apt 环境下自动安装 `libegl1`、`libosmesa6`、`libglu1-mesa`。
- 新增 `scripts/start_evo1_server.sh`
  - 用环境变量或参数启动 Evo1 websocket server。
  - 支持 `EVO1_PYTHON`、`EVO1_CKPT_DIR`、`EVO1_HOST`、`EVO1_PORT`、`EVO1_DEVICE`、`EVO1_INFERENCE_STEPS`。
  - 默认启动前运行 `scripts/preflight.py --checkpoint`，支持 `EVO1_SKIP_PREFLIGHT=1` 跳过。
- 新增 `scripts/run_libero_smoke.sh`
  - 固化 1 task / 1 episode / 1 step 的 LIBERO smoke 默认配置。
  - 支持通过环境变量扩展到更长 eval。
  - 默认写出 `${EVO1_LIBERO_CKPT_NAME}_results.json`。
  - 支持 `EVO1_LIBERO_RUN_DIR=/path/to/run`，将日志、视频、结果 JSON 和 run manifest 分别写入 `logs/`、`videos/`、`results/` 与 `run_manifest.json`。
  - 支持 `EVO1_LIBERO_DRY_RUN=1` 打印环境变量而不运行客户端。
- 新增 `scripts/run_libero_eval.sh`
  - 固化正式 eval 默认配置：4 个 LIBERO suite、10 episodes、horizon 14、max steps `25,25,25,95`。
  - 支持 `EVO1_LIBERO_DRY_RUN=1` 打印环境变量而不运行客户端。
  - 支持 `EVO1_LIBERO_RUN_DIR=/path/to/run`，将日志、视频、结果 JSON 和 run manifest 分别写入 `logs/`、`videos/`、`results/` 与 `run_manifest.json`。
- 新增 `scripts/write_libero_run_manifest.py`
  - 在 LIBERO smoke/eval 启动客户端前写出 manifest。
  - 记录 run kind、Git commit/branch/dirty 状态、命令、Python 版本、输出路径和非敏感环境变量。
  - 即使客户端崩溃或评估中断，也能留下本次运行的上下文。
- 新增 `scripts/download_libero_checkpoint.sh`
  - 默认下载 `MINT-SJTU/Evo1_LIBERO` 到 `$EVO1_DATA_ROOT/checkpoints/Evo1_LIBERO`。
  - 默认不设置 `HF_ENDPOINT`，避免影响国内资源下载。
  - 如需外网 Hugging Face 加速，只通过单条命令设置 `EVO1_HF_ENDPOINT=https://hf-mirror.com`。
  - 支持 `EVO1_DOWNLOAD_LIBERO_CHECKPOINT_DRY_RUN=1` 打印最终下载命令而不联网。
- 新增 `scripts/export_unpushed_commits.sh`
  - 默认以 `origin/main` 为基线导出本地领先提交。
  - 默认输出到 `exports/unpushed_commits_<UTC time>/`。
  - 输出 `patches/*.patch`、`manifest.json` 和应用说明 `README.md`。
  - 适用于当前 GitHub remote 无写权限时，把本地工程化改造迁移到新 remote 或服务器。
- 新增 `scripts/audit_requirements.py`
  - 检查仓库内 `requirements*.txt` 是否都被 `requirements-policy.json` 覆盖。
  - 未固定版本的依赖如果没有策略理由会失败。
  - 已固定的 LIBERO 依赖保持严格检查；Evo1 主环境当前浮动项以 WARN 暴露并登记为后续锁版本任务。
- 新增 `scripts/check_repo.sh`
  - 统一运行 requirements 审计、pytest、ruff、shell 语法、preflight、LIBERO setup dry-run、LIBERO checkpoint download dry-run、compileall 和 whitespace 检查。
  - 本地默认缺少 `ruff` 时 WARN 后继续，CI 使用 `EVO1_CHECK_REQUIRE_RUFF=1` 强制失败。
  - 支持 `EVO1_CHECK_DRY_RUN=1`、`EVO1_CHECK_SKIP_PYTEST=1`、`EVO1_CHECK_SKIP_COMPILE=1`、`EVO1_CHECK_SKIP_RUFF=1`。
- 新增 `scripts/validate_training_dataset.py`
  - 复用 `Evo_1/dataset/validation.py`。
  - 检查 `tasks.jsonl`、`episodes.jsonl`、`stats.json` 或 `episodes_stats.jsonl`。
  - 检查 `data/*/*.parquet` 以及由 `view_map` 推导出的 `videos/<episode>/<view>/<trajectory>.mp4`。
  - `scripts/preflight.py --strict-data` 也会使用同一套训练数据结构验证逻辑。
- 新增 `scripts/summarize_libero_results.py`
  - 支持输入 result JSON 文件、目录或 glob。
  - 输出 overall 和 per-suite 行。
  - 支持 Markdown 和 CSV，用于多次复现/改进结果对比。
  - 输出 run name、git commit、dirty 状态和生成时间，便于追踪实验来源。
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
- 主训练环境依赖补充：`PyYAML==6.0.2`

LIBERO 依赖规格：

- 仓库内新增：`requirements-libero.txt`
- 固定顶层包：`libero==0.1.1`、`robosuite==1.4.0`、`robomimic==0.2.0`、`mujoco==3.2.3`
- 固定辅助包：`websockets==12.0`、`imageio==2.34.2`、`huggingface_hub==0.23.4`

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
export HF_HOME=/root/autodl-tmp/hf-home
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf-cache
export PIP_CACHE_DIR=/root/autodl-tmp/pip-cache
export TMPDIR=/root/autodl-tmp/tmp
```

`HF_ENDPOINT=https://hf-mirror.com` 只建议在单次 Hugging Face 外网下载命令前临时设置，不写入全局
shell 启动文件或系统环境，避免拖慢国内资源下载。

已下载 checkpoint：

- `/root/autodl-tmp/checkpoints/Evo1_MetaWorld`
- `/root/autodl-tmp/checkpoints/Evo1_LIBERO`

已缓存基座模型：

- `OpenGVLab/InternVL3-1B`，缓存位于 `/root/autodl-tmp/hf-cache`

## 已验证

本地验证：

```bash
scripts/check_repo.sh
```

本地结果：

- `scripts/check_repo.sh`：通过
- `pytest`：92 passed, 3 skipped
- `scripts/audit_requirements.py`：通过；当前 Evo1 主环境和 dev 环境的浮动依赖都以 WARN 暴露，并已在 `requirements-policy.json` 登记理由
- `scripts/preflight.py`：通过；仅提示默认训练数据路径不存在的 WARN（本地未放完整训练数据，非失败）
- `bash -n scripts/*.sh`：通过
- `EVO1_SETUP_LIBERO_DRY_RUN=1 scripts/setup_libero_env.sh`：通过，能在不创建 conda 环境、不下载 assets 的情况下打印解析后的路径
- `EVO1_DOWNLOAD_LIBERO_CHECKPOINT_DRY_RUN=1 scripts/download_libero_checkpoint.sh`：通过，默认不设置 `HF_ENDPOINT`
- `EVO1_LIBERO_DRY_RUN=1 EVO1_LIBERO_RUN_DIR=run_outputs/test_eval scripts/run_libero_eval.sh`：通过，输出路径解析到 `logs/`、`videos/`、`results/`
- `python3 scripts/write_libero_run_manifest.py --output /tmp/... --run-kind smoke --repo-root "$PWD"`：通过，能写出非敏感运行上下文
- `python3 -m pytest tests/test_preflight.py tests/test_write_libero_run_manifest.py`：通过，覆盖 LIBERO run manifest 校验
- `python3 scripts/preflight.py --dataset-config "" --libero-manifest /tmp/evo1_manifest_check/run_manifest.json`：通过，能校验真实写出的 manifest
- `compileall`：通过
- `git diff --check`：通过
- `python3 -m ruff check .`：本地 Python 环境未安装 `ruff`；`scripts/check_repo.sh` 已按本地默认策略 WARN 后跳过，CI 会强制要求 `ruff`

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
- 最新的 LIBERO result summary JSON 改动只完成了本地单元测试，服务器已关机，尚未在真实 LIBERO 环境上重跑 smoke。
- 尚未下载训练数据集并跑短步数训练 smoke test；已新增训练数据结构验证脚本，当前本地默认数据路径不存在时会明确失败。
- `Evo_1/requirements.txt` 仍有部分浮动依赖；目前已由 `requirements-policy.json` 审计和显式登记，后续可根据服务器真实环境进一步锁定主模型/训练环境版本。
- `flash-attn` 目前按服务器实际环境安装成功，尚未写入主 requirements，避免无 GPU/无匹配 wheel 环境被强绑定。

## 下一步建议

1. 处理 GitHub 权限：
   - 给 `myserendipity137` 写权限，或
   - 添加一个有写权限的新 remote。
2. 需要评估效果时，在服务器跑完整 LIBERO eval，例如逐个 suite 设置合理 `EVO1_LIBERO_MAX_STEPS` 和 `EVO1_LIBERO_HORIZON=14`。
3. 下载小规模训练数据或抽样数据，跑 `max_steps=1` 到 `10` 的训练 smoke test。
4. 根据真实 eval/training 结果继续修复工程问题。
5. 在下一次服务器可用时，用新脚本从空环境重跑一次 LIBERO 安装和 smoke，确认脚本级端到端复现。
