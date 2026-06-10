# Evo-1 工程化改造记录

更新时间：2026-06-10

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
  - 本地提交已完成。
  - `git push origin main` 失败，原因是当前凭据 `myserendipity137` 没有 `zixuan-wan146/Evo-1.git` 写权限。
  - 需要给该账号写权限，或提供有权限的新 remote。

## 已完成提交

本地提交：

- `fb2ffad Improve engineering baseline for Evo-1`
- `c97cfa8 Document remote setup and checkpoint loading`
- `a2888f2 Add practical lint gate`

服务器对应提交：

- `8e7edc8 Improve engineering baseline for Evo-1`
- `3fb5b49 Document remote setup and checkpoint loading`
- `ee64c12 Add practical lint gate`

服务器提交哈希不同是因为通过 `git format-patch | git am` 应用，内容等价但提交对象不同。

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

- `Evo_1/scripts/train.py`
  - 增加训练配置校验。
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

### 评估脚本配置化

- `LIBERO_evaluation/libero_client_4tasks.py`
  - 客户端默认地址从 `ws://0.0.0.0:9000` 修正为 `ws://127.0.0.1:9000`。
  - 支持环境变量：
    - `EVO1_SERVER_URI`
    - `EVO1_LIBERO_SERVER_URL`
    - `EVO1_LIBERO_HORIZON`
    - `EVO1_LIBERO_MAX_STEPS`
    - `EVO1_LIBERO_TASK_SUITES`
    - `EVO1_LIBERO_EPISODES`
    - `EVO1_LIBERO_SEED`
    - `EVO1_LIBERO_LOG_DIR`
    - `EVO1_LIBERO_VIDEO_DIR`

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
- 记录 `HF_HOME`、`HUGGINGFACE_HUB_CACHE`、`PIP_CACHE_DIR`、`TMPDIR` 等数据盘路径建议。
- 记录 `flash-attn` cross-device link 安装问题的处理方式。

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
PYTHONPYCACHEPREFIX=/tmp/evo1_pycache python3 -m compileall -q Evo_1 MetaWorld_evaluation LIBERO_evaluation tests
git diff --check
```

本地结果：

- `pytest`：5 passed, 1 skipped（本地无 torch，跳过 torch 相关测试）
- `compileall`：通过
- `git diff --check`：通过

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

## 尚未完成

- GitHub push 尚未完成，需要仓库写权限或新的目标 remote。
- 尚未安装 MetaWorld 评估环境并跑真实 MT50 smoke eval。
- 尚未安装 LIBERO 评估环境并跑真实 LIBERO smoke eval。
- 尚未下载训练数据集并跑短步数训练 smoke test。
- `requirements.txt` 仍有部分浮动依赖，后续可进一步锁定版本。
- `flash-attn` 目前按服务器实际环境安装成功，尚未写入主 requirements，避免无 GPU/无匹配 wheel 环境被强绑定。

## 下一步建议

1. 处理 GitHub 权限：
   - 给 `myserendipity137` 写权限，或
   - 添加一个有写权限的新 remote。
2. 在服务器继续安装 MetaWorld 环境，跑 `EVO1_MT50_EPISODES=1` 的真实 smoke eval。
3. 安装 LIBERO 环境，跑 `EVO1_LIBERO_EPISODES=1` 的真实 smoke eval。
4. 下载小规模训练数据或抽样数据，跑 `max_steps=1` 到 `10` 的训练 smoke test。
5. 根据真实 eval/training 结果继续修复工程问题。
