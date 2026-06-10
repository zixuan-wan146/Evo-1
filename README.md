# Evo-1 Simulation Workspace

This repository has been trimmed to the simulation workflow for Evo-1:

- MetaWorld evaluation
- LIBERO evaluation
- Evo-1 model server
- training on simulation datasets

Real-robot examples and external robot-integration code have been removed from this workspace.

## Project Layout

```text
Evo_1/
  dataset/                  Training dataset loader and dataset config
  model/                    Evo-1 model components
  runtime_config.py         Shared runtime constants
  scripts/
    Evo1_server.py          WebSocket inference server for simulation clients
    train.py                Training entry point
  utils/                    Shared helpers

MetaWorld_evaluation/       MetaWorld simulation evaluation client
LIBERO_evaluation/          LIBERO simulation evaluation client
deepspeed_setup_example.txt Accelerate/DeepSpeed setup reference
```

## Installation

```bash
conda create -n Evo1 python=3.10 -y
conda activate Evo1

cd Evo_1
pip install -r requirements.txt

# Adjust MAX_JOBS for your machine if needed.
MAX_JOBS=64 pip install -v flash-attn --no-build-isolation
```

## Development Checks

Install the lightweight development dependencies from the repository root:

```bash
pip install -r requirements-dev.txt
scripts/check_repo.sh
```

The lightweight tests avoid downloading model weights. Tests that require PyTorch are skipped when
PyTorch is not installed.

`scripts/check_repo.sh` runs the local quality gate: dependency policy audit, unit tests, optional
`ruff`, shell syntax checks, repository preflight, LIBERO setup dry-run, LIBERO checkpoint download
dry-run, `compileall`, and `git diff --check`. Set `EVO1_CHECK_REQUIRE_RUFF=1` in CI or a fully
prepared dev environment to make missing `ruff` fail instead of warn.

`scripts/audit_requirements.py` fails when a new `requirements*.txt` file is not covered by
`requirements-policy.json`, or when a dependency is left unpinned without an explicit reason.
Existing unpinned Evo-1 runtime dependencies are recorded as known follow-up work until the exact
GPU server wheel set is captured.

If the GitHub remote is temporarily unavailable or your account lacks write permission, export the
local commits as a portable patch bundle:

```bash
scripts/export_unpushed_commits.sh
```

The export is written under `exports/` and can be applied to another clone with
`git am /path/to/export/patches/*.patch`.

## MetaWorld Evaluation

Create a separate environment for MetaWorld:

```bash
conda create -n metaworld python=3.10 -y
conda activate metaworld
pip install mujoco metaworld websockets opencv-python packaging huggingface_hub
```

Download the checkpoint:

```bash
hf download MINT-SJTU/Evo1_MetaWorld --local-dir /path/to/checkpoint
```

Start the Evo-1 server:

```bash
conda activate Evo1
cd Evo_1
python scripts/Evo1_server.py --ckpt_dir /path/to/checkpoint --port 9000
```

The WebSocket request must be a JSON object with:

- `image`: exactly 3 RGB image arrays with pixel values in `0..255`; images are resized by the server.
- `state`: a non-empty finite numeric vector with length at most 24.
- `image_mask`: 0/1 mask with length at most 3; shorter masks are padded with zeros.
- `action_mask`: 0/1 mask with length at most 24; shorter masks are padded with zeros and at least one dimension must be active.
- `prompt`: optional task instruction string.

Run the MetaWorld client:

```bash
conda activate metaworld
cd MetaWorld_evaluation
python mt50_evo1_client_prompt.py
```

The MetaWorld client uses `ws://127.0.0.1:9000` by default. Change `SERVER_URL` in
`MetaWorld_evaluation/mt50_evo1_client_prompt.py` if you run the server elsewhere.

Common MetaWorld client settings can also be overridden without editing source code:

```bash
export EVO1_SERVER_URI=ws://127.0.0.1:9000
export EVO1_MT50_EPISODES=1
export EVO1_MT50_EPISODE_HORIZON=100
export EVO1_MT50_SAVE_VIDEO=false
python mt50_evo1_client_prompt.py
```

## LIBERO Evaluation

Recommended setup for a server with a data disk:

```bash
EVO1_DATA_ROOT=/root/autodl-tmp \
CONDA_BIN=/root/miniconda3/bin/conda \
scripts/setup_libero_env.sh
```

The setup script creates a Python 3.8.13 LIBERO environment, installs `libero==0.1.1`,
downloads LIBERO assets, configures `~/.libero/config.yaml`, and installs the headless
MuJoCo system libraries when run as root on Ubuntu.

The script installs top-level LIBERO packages from `requirements-libero.txt`. To validate resolved
paths without creating a conda environment or downloading assets:

```bash
EVO1_SETUP_LIBERO_DRY_RUN=1 scripts/setup_libero_env.sh
```

Use `EVO1_LIBERO_REQUIREMENTS=/path/to/requirements.txt` only when deliberately testing another
LIBERO dependency set.

Download the checkpoint:

```bash
EVO1_DATA_ROOT=/root/autodl-tmp scripts/download_libero_checkpoint.sh
```

Start the Evo-1 server:

```bash
EVO1_PYTHON=/root/autodl-tmp/miniforge3/envs/Evo1/bin/python \
scripts/start_evo1_server.sh /path/to/checkpoint
```

`scripts/start_evo1_server.sh` runs a lightweight checkpoint preflight before loading the model.
Set `EVO1_SKIP_PREFLIGHT=1` only when deliberately bypassing that check for debugging.
`scripts/download_libero_checkpoint.sh` writes to `$EVO1_DATA_ROOT/checkpoints/Evo1_LIBERO` by
default. It does not set a Hugging Face mirror by default; if a single external download needs one,
use `EVO1_HF_ENDPOINT=https://hf-mirror.com` only on that command.

Run the minimal LIBERO smoke client from another shell:

```bash
LIBERO_PYTHON=/root/autodl-tmp/envs/libero/bin/python \
scripts/run_libero_smoke.sh
```

To keep one run's logs, videos, and result JSON together, set a run directory:

```bash
EVO1_LIBERO_RUN_DIR=/root/autodl-tmp/evo1_runs/libero_smoke_001 \
LIBERO_PYTHON=/root/autodl-tmp/envs/libero/bin/python \
scripts/run_libero_smoke.sh
```

The run directory layout is `logs/`, `videos/`, `results/`, and `run_manifest.json`.
The manifest is written before the client starts, so failed or interrupted runs still keep the
resolved LIBERO settings, output paths, Git commit, dirty state, command, Python version, and
selected non-secret environment variables.

Run the full default LIBERO evaluation when you are ready to collect comparable numbers:

```bash
LIBERO_PYTHON=/root/autodl-tmp/envs/libero/bin/python \
scripts/run_libero_eval.sh
```

`scripts/run_libero_eval.sh` defaults to all four LIBERO suites, `EVO1_LIBERO_HORIZON=14`,
`EVO1_LIBERO_EPISODES=10`, and max steps `25,25,25,95`. Set `EVO1_LIBERO_DRY_RUN=1` to print the
resolved eval environment without running the client.
Set `EVO1_LIBERO_RUN_DIR=/path/to/run` to use the same grouped output layout as smoke runs.

The LIBERO client stores logs, videos, and a machine-readable result summary under
`LIBERO_evaluation/`.

Common LIBERO client settings can be overridden without editing source code:

```bash
export EVO1_SERVER_URI=ws://127.0.0.1:9000
export EVO1_MUJOCO_GL=osmesa
export EVO1_LIBERO_EPISODES=1
export EVO1_LIBERO_TASK_SUITES=libero_spatial
export EVO1_LIBERO_TASK_LIMIT=1
export EVO1_LIBERO_MAX_STEPS=25
export EVO1_LIBERO_RESULT_FILE="$PWD/LIBERO_evaluation/log_file/libero_spatial_results.json"
export EVO1_LIBERO_MANIFEST_FILE="$PWD/LIBERO_evaluation/log_file/libero_spatial_run_manifest.json"
LIBERO_PYTHON=/root/autodl-tmp/envs/libero/bin/python scripts/run_libero_smoke.sh
```

The result summary JSON contains run metadata, evaluated episodes, per-suite success rates, and
failure reasons such as action parsing errors or step-limit exhaustion. Metadata includes the
current Git commit, dirty state, command, Python version, and selected non-secret environment
variables.

To compare one or more LIBERO runs after evaluation:

```bash
python scripts/summarize_libero_results.py LIBERO_evaluation/log_file/*_results.json \
  --output outputs/libero_results.md
python scripts/summarize_libero_results.py LIBERO_evaluation/log_file/*_results.json \
  --format csv \
  --output outputs/libero_results.csv
```

The comparison table includes the run name, Git commit, dirty state, overall metrics, and per-suite
metrics when present in the result JSON.

To inventory grouped run directories, including interrupted runs that only have a manifest:

```bash
python scripts/summarize_libero_results.py /root/autodl-tmp/evo1_runs \
  --table runs \
  --output outputs/libero_run_inventory.md
```

The run inventory table reports each run directory, completeness status, manifest settings, result
path, Git metadata, and overall success metrics when a result JSON exists.

To gate a candidate result before treating it as an improvement:

```bash
python scripts/check_libero_metrics.py /root/autodl-tmp/evo1_runs/candidate \
  --min-success-rate 0.10 \
  --min-total-episodes 10
python scripts/check_libero_metrics.py /root/autodl-tmp/evo1_runs/candidate \
  --baseline /root/autodl-tmp/evo1_runs/baseline \
  --max-regression 0.02
```

The metric gate defaults to the `overall` scope. Add `--scope suite:libero_spatial` or repeat
`--scope` to gate suite-level metrics.

To generate a report bundle for a set of runs:

```bash
python scripts/report_libero_runs.py /root/autodl-tmp/evo1_runs \
  --output-dir outputs/libero_report \
  --min-success-rate 0.10 \
  --min-total-episodes 10
```

The report directory contains run inventory tables, result summary tables, a report manifest, and a
metric gate log when gate options are provided.

For headless smoke tests, `EVO1_MUJOCO_GL=osmesa` is the more stable default. Use
`EVO1_MUJOCO_GL=egl` on GPU servers when EGL cleanup warnings are acceptable and
faster rendering is preferred.

Before running a longer evaluation, use the lightweight preflight checks:

```bash
python scripts/preflight.py \
  --checkpoint /path/to/checkpoint \
  --check-imports evo1
```

The checkpoint check validates required files, basic `config.json` dimensions, and `norm_stats.json`
state/action min-max structure without loading model weights.

After evaluation, validate result JSON files and run manifests before summarizing or syncing them:

```bash
python scripts/preflight.py \
  --dataset-config "" \
  --libero-result "LIBERO_evaluation/log_file/*_results.json" \
  --libero-manifest "LIBERO_evaluation/log_file/*_run_manifest.json"
```

The result check verifies both schema and consistency between overall/per-suite summaries and the
episode records. The manifest check verifies run kind, key LIBERO settings, Git metadata, and that
recorded environment variables do not include common secret fields.

If the run used `EVO1_LIBERO_RUN_DIR`, validate the whole run directory instead:

```bash
python scripts/preflight.py \
  --dataset-config "" \
  --libero-run-dir /root/autodl-tmp/evo1_runs/libero_smoke_001
```

The run-directory check validates both files and verifies that the manifest points to the result
JSON from the same run, with matching checkpoint name and Git metadata.

For a strict training-data check, add `--strict-data` after downloading the dataset.

## Training

Download the example MetaWorld dataset:

```bash
mkdir Evo1_training_dataset
cd Evo1_training_dataset
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/MINT-SJTU/Evo1_MetaWorld_Dataset
cd Evo1_MetaWorld_Dataset
git lfs pull
```

Configure the dataset path in `Evo_1/dataset/config.yaml`. The default path expects:

```text
Evo1_training_dataset/Evo1_MetaWorld_Dataset
```

Before starting a training run on a new dataset copy, validate the dataset structure from the
repository root:

```bash
python scripts/validate_training_dataset.py \
  --dataset-config Evo_1/dataset/config.yaml \
  --dataset-base-dir Evo_1
```

The validator checks `tasks.jsonl`, `episodes.jsonl`, `stats.json` or `episodes_stats.jsonl`,
`data/*/*.parquet`, and expected video paths derived from the dataset `view_map`.

Run stage 1 training:

```bash
conda activate Evo1
cd Evo_1

accelerate launch --num_processes 1 --num_machines 1 --deepspeed_config_file ds_config.json scripts/train.py \
  --run_name Evo1_metaworld_stage1 \
  --action_head flowmatching \
  --use_augmentation \
  --lr 1e-5 \
  --dropout 0.2 \
  --weight_decay 1e-3 \
  --batch_size 16 \
  --image_size 448 \
  --max_steps 5000 \
  --log_interval 10 \
  --ckpt_interval 2500 \
  --warmup_steps 1000 \
  --grad_clip_norm 1.0 \
  --num_layers 8 \
  --horizon 50 \
  --finetune_action_head \
  --disable_wandb \
  --vlm_name OpenGVLab/InternVL3-1B \
  --dataset_config_path dataset/config.yaml \
  --dataset_config_base_dir . \
  --per_action_dim 24 \
  --state_dim 24 \
  --save_dir /path/to/checkpoints/stage1
```

Run stage 2 training:

```bash
conda activate Evo1
cd Evo_1

accelerate launch --num_processes 1 --num_machines 1 --deepspeed_config_file ds_config.json scripts/train.py \
  --run_name Evo1_metaworld_stage2 \
  --action_head flowmatching \
  --use_augmentation \
  --lr 1e-5 \
  --dropout 0.2 \
  --weight_decay 1e-3 \
  --batch_size 16 \
  --image_size 448 \
  --max_steps 80000 \
  --log_interval 10 \
  --ckpt_interval 2500 \
  --warmup_steps 1000 \
  --grad_clip_norm 1.0 \
  --num_layers 8 \
  --horizon 50 \
  --finetune_vlm \
  --finetune_action_head \
  --disable_wandb \
  --vlm_name OpenGVLab/InternVL3-1B \
  --dataset_config_path dataset/config.yaml \
  --dataset_config_base_dir . \
  --per_action_dim 24 \
  --state_dim 24 \
  --save_dir /path/to/checkpoints/stage2 \
  --resume \
  --resume_pretrain \
  --resume_path /path/to/checkpoints/stage1/step_5000
```

Relative dataset paths inside `dataset/config.yaml` are resolved from `--dataset_config_base_dir`.
The examples above use `.` because they run from `Evo_1/`; if launching from the repository root,
use `--dataset_config_path Evo_1/dataset/config.yaml --dataset_config_base_dir Evo_1`.
Use `--cache_dir /path/to/cache` if you want the generated training cache outside the project directory.

## Remote Deployment Notes

On remote servers with a separate data disk, keep code, checkpoints, caches, and outputs outside the
system disk. For example:

```bash
cd /root/autodl-tmp
git clone https://github.com/zixuan-wan146/Evo-1.git
export HF_HOME=/root/autodl-tmp/hf-home
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf-cache
export PIP_CACHE_DIR=/root/autodl-tmp/pip-cache
```

Only set `HF_ENDPOINT=https://hf-mirror.com` for a single Hugging Face download command when that
specific external download benefits from it. Do not put it in shell startup files or global env
configuration, because it can slow down downloads from domestic resources.

Download checkpoints to the data disk:

```bash
hf download MINT-SJTU/Evo1_MetaWorld --local-dir /root/autodl-tmp/checkpoints/Evo1_MetaWorld --max-workers 1
hf download MINT-SJTU/Evo1_LIBERO --local-dir /root/autodl-tmp/checkpoints/Evo1_LIBERO --max-workers 1
```

If `flash-attn` installation fails with a cross-device link error, set `TMPDIR` to the same data disk:

```bash
mkdir -p /root/autodl-tmp/tmp /root/autodl-tmp/pip-cache
export TMPDIR=/root/autodl-tmp/tmp
export PIP_CACHE_DIR=/root/autodl-tmp/pip-cache
pip install flash-attn --no-build-isolation
```

## Citation

```bibtex
@article{lin2025evo,
  title={Evo-1: Lightweight Vision-Language-Action Model with Preserved Semantic Alignment},
  author={Lin, Tao and Zhong, Yilei and Du, Yuxin and Zhang, Jingjing and Liu, Jiting and Chen, Yinxinyu and Gu, Encheng and Liu, Ziyan and Cai, Hongyi and Zou, Yanwen and others},
  journal={arXiv preprint arXiv:2511.04555},
  year={2025}
}
```
