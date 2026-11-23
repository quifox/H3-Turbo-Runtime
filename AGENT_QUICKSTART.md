# 给 AI 的部署提示词

> 先读完这份文件，再开始。目标是用最少的步骤，在本地用 CLI 出通一条 5 秒示例视频；出片成功即视为部署成功。ComfyUI 界面是可选的，需要时再配。

## 目标

- **CLI 出片是成功标准。**
- 能复用就复用，缺什么才装什么。
- 不做 15 秒、Ref2VA、性能调优、CozyUI 等额外工作。

## 1. 先检测（只读）

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true
df -h .
command -v uv ffmpeg git python3

# ComfyUI 源码（CLI 也要用它，但不一定要起服务）
echo "H3_RUNTIME_COMFY=${H3_RUNTIME_COMFY:-}"
test -f "${H3_RUNTIME_COMFY:-/nonexistent}/nodes.py" && echo "ComfyUI source: found" || echo "ComfyUI source: not found"

# 权重与示例条件
ls weights 2>/dev/null | head
test -f runtime/examples/nora/condition.safetensors && echo "example condition: found" || echo "example condition: missing"
```

判断：

- 没有 GPU 或 `nvidia-smi` 失败 → 停，报告。
- 磁盘剩余 < 30GB → 停（CLI 只取 4 份主权重约 18.27GB，另留输出与缓存）。
- 已有 ComfyUI 源码、`weights/`、示例条件 → 记下来，后面跳过对应步骤。

## 2. 先停下来问用户

**不要先做后问。** 把检测结果和缺的项一次列给用户，等用户逐项确认后再继续；已经有的项跳过、不问。

要问的项：

1. 只跑 CLI 需要 4 份主权重，约 18.27GB；如果也要 ComfyUI 文本条件，是 6 份、约 23.67GB。选哪种？
2. ComfyUI 源码是否已有？没有的话，是否允许准备一份？
3. 现在是否允许跑 GPU 生成？

可以直接这样问：

> 本机检测到：<GPU / 显存 / 磁盘 / ComfyUI / 权重 / 示例条件状态>。
> 缺 <X / Y / Z>，需要你确认：
> 1) 只取 CLI 的 4 份主权重（约 18.27GB），还是连 ComfyUI 条件一起取 6 份（约 23.67GB）？
> 2) 是否允许准备 ComfyUI 源码？用已有的还是新装？
> 3) 现在是否允许跑 GPU 生成？
> 你确认后我再继续。

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

只跑 CLI：

```bash
uv run python scripts/fetch_weights.py --fetch --skip-conditioning
```

4 份主权重，约 18.27GB，落到 `weights/`。之后要用 ComfyUI 文本条件，再跑一次不带 `--skip-conditioning` 的 `--fetch`，补 2 份。

### 3.3 ComfyUI 源码（缺才做，先确认）

```bash
export H3_RUNTIME_COMFY=/path/to/ComfyUI
uv run python deploy.py check --comfy "$H3_RUNTIME_COMFY" --models weights
```

CLI 也需要这份源码（运行时会导入 ComfyUI 模块），但不需要启动 ComfyUI 服务。需要新装时，按 `runtime/models/manifest.json` 的 `runtime.comfyui_revision` 准备；先确认再装。`check` 报错就停。

### 3.4 CLI 出片

```bash
uv run python deploy.py render --comfy "$H3_RUNTIME_COMFY" --models weights \
  --out runs/cli-sample --steps 4 --strategy fast
```

- 不传 `--condition` 时，使用仓库自带的 `runtime/examples/nora/condition.safetensors`；
- 输出在 `runs/cli-sample/output.mp4`，记录在 `runs/cli-sample/result.json`。

### 3.5 验证

```bash
ffprobe -v error -show_entries format=duration \
  -show_entries stream=codec_type,width,height,r_frame_rate \
  -of json runs/cli-sample/output.mp4
python3 - <<'PY'
import json
r = json.load(open("runs/cli-sample/result.json"))
c = json.load(open("runtime/examples/nora/case.json"))
print("latents match:", r["stages"]["sample"]["latent_sha256"] == c["expected_latent_sha256"])
PY
```

期望：约 5.000 秒、832×480、24/1、有音频流；`latents match: True`。

### 3.6 ComfyUI 界面（可选）

只有用户需要界面/新提示词时才做：

- 按 `integrations/comfyui/README.md` 安装外部 ClipProj 插件、配置节点、启动 ComfyUI；
- 之后可以用 `complete-5s-api.json` 从文本出片，或换首尾帧/参考素材。

这不是部署成功的必要条件。

## 4. 报告

- 设备与版本（GPU、驱动、ComfyUI revision、仓库 commit）
- 部署路径（weights、ComfyUI）
- 输出文件路径 + ffprobe 结果 + `latents match`
- 通过/失败；失败贴原始报错
- 未验证项（如有）

## 5. 规则

- 下载、安装、GPU 生成前先确认；不要连续自动做完。
- 失败保留原始错误和日志，不包装成新错误。
- 不静默切换 `fast` / `stream`；选择要显式。
- 不把权重、`runs/`、日志、本地路径提交进仓库。
