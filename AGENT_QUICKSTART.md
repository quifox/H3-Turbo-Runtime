# 给 AI 的部署提示词

> 先读完这份文件，再开始。目标是用最少的步骤，在这台机器上跑通一条 5 秒示例视频；不做目标之外的优化、训练或改造。

## 目标

- 能复用就复用，缺什么才装什么。
- 只生成一条 5s / 832×480 / 24fps / 带原生音轨的 MP4。
- 不做 15 秒、Ref2VA、性能调优、CozyUI 等额外工作。

## 1. 先检测（只读）

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true
df -h .
command -v uv ffmpeg git python3
echo "H3_RUNTIME_COMFY=${H3_RUNTIME_COMFY:-}"
test -f "${H3_RUNTIME_COMFY:-/nonexistent}/nodes.py" && echo "ComfyUI: found" || echo "ComfyUI: not found"
ls weights 2>/dev/null | head
ls workbench/condition-models 2>/dev/null | head
test -f workbench/ClipProj/__init__.py && echo "ClipProj: found" || echo "ClipProj: missing"
```

判断：

- 没有 GPU 或 `nvidia-smi` 失败 → 停，报告。
- 磁盘剩余 < 40GB → 停，报告需要约 40GB。
- 已有 ComfyUI、权重、ClipProj → 记下来，后面跳过对应步骤。

## 2. 只在缺的时候问

以下都是非平凡操作，做之前逐项确认；已有就跳过、不问：

1. 权重约 23.67GB，是否允许下载？放哪个目录（剩余 ≥ 40GB）？
2. 没有 ComfyUI，是否允许克隆固定版本并装依赖？
3. 现在是否允许跑 GPU 生成？

## 3. 最小流程

### 3.1 仓库与依赖

```bash
git clone https://github.com/quifox/H3-Turbo-Runtime.git
cd H3-Turbo-Runtime
uv sync
uv run pytest -q
```

测试不过就停，贴原始报错。

### 3.2 权重（缺才做）

```bash
uv run python scripts/fetch_weights.py --fetch
```

6 份，约 23.67GB；文本编码器和 ClipProj 权重会落到 `workbench/condition-models/`。校验失败就停。

### 3.3 ComfyUI（缺才做）

```bash
export H3_RUNTIME_COMFY=/path/to/ComfyUI
uv run python deploy.py check --comfy "$H3_RUNTIME_COMFY" --models weights
```

需要新装时，按 `runtime/models/manifest.json` 的 `runtime.comfyui_revision` 准备；先确认再装。`check` 报错就停。

### 3.4 ClipProj 插件（缺才做）

```bash
git clone https://github.com/nicolab28/ComfyUI-ClipProj.git workbench/ClipProj
git -C workbench/ClipProj checkout c01ba8fb8f41b4f2094dbd0b185cdc238fb6134c
```

它是外部 MIT 插件，不随仓库；节点默认从 `workbench/ClipProj` 加载。

### 3.5 配置与启动

```bash
export H3_RUNTIME_MODELS="$PWD/weights"
export H3_RUNTIME_WORKBENCH="$PWD/workbench"
export H3_RUNTIME_PYTHON="$PWD/.venv/bin/python"
uv run python deploy.py configure-comfy --comfy "$H3_RUNTIME_COMFY"

cd "$H3_RUNTIME_COMFY"
python main.py --listen 127.0.0.1 --port 8188
```

显存紧张时 ComfyUI 可加 `--cpu`，生成子进程会用自己的 Python 跑 GPU。确认 `object_info` 里包含 `H3TurboRuntime`。

### 3.6 出片

```bash
cd /path/to/H3-Turbo-Runtime
python3 - <<'PY'
import json
wf = json.load(open("integrations/comfyui/complete-5s-api.json"))
json.dump({"prompt": wf}, open("/tmp/h3-sample.json", "w"))
PY
curl -s -X POST http://127.0.0.1:8188/prompt \
  -H 'Content-Type: application/json' -d @/tmp/h3-sample.json
```

记下 `prompt_id`，轮询 `/history/<prompt_id>`；MP4 在 `$H3_RUNTIME_COMFY/output/h3-turbo/`。

如果 compute capability 不是 8.9，先停下问用户；不要自动把 `fast` 改成 `stream`。

### 3.7 验证

```bash
ffprobe -v error -show_entries format=duration \
  -show_entries stream=codec_type,width,height,r_frame_rate \
  -of json /path/to/output.mp4
```

期望：约 5.000 秒、832×480、24/1、有音频流。

## 4. 报告

- 设备与版本（GPU、驱动、ComfyUI revision、仓库 commit）
- 部署路径（weights、workbench、ComfyUI）
- 输出文件路径 + ffprobe 结果
- 通过/失败；失败贴原始报错
- 未验证项（如有）

## 5. 规则

- 下载、安装、GPU 生成前先确认；不要连续自动做完。
- 失败保留原始错误和日志，不包装成新错误。
- 不静默切换 `fast` / `stream`；选择要显式。
- 不把权重、`runs/`、日志、本地路径提交进仓库。
