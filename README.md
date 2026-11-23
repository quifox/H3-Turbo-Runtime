# H3 Turbo Runtime

简体中文 · [English](README.en.md)

MiniMax H3 Turbo 的低显存原生音视频推理运行时。支持**文本生成视频、首尾帧生成视频、可配置时长、多时间关键帧、原生音频和视频同步输出**，默认使用 5 秒、4 步配置。**RTX 4060 Laptop 8 GiB 实测（基于 `uv` 统一管理的 Python 3.14 环境）：5 秒生成约 2 分 39 秒（总成片 159.18 秒，去噪 118.92 秒）、整卡显存峰值 5.86 GiB；15 秒单次生成去噪 586.79 秒、显存峰值约 6.83 GiB。** 保留模型原生的视频与音频生成，无需把 15 秒拆成三段 5 秒视频。默认展示首尾帧（FL2VA）模式。

[上游模型清单与规格](MODELS.md) · [完整 5 秒 ComfyUI 工作流](integrations/comfyui/complete-5s-workflow.json) · [ComfyUI 全量参数接口](integrations/comfyui/README.md) · [CozyUI 接入](integrations/cozyui/README.md) · [公开有声样片](demos/README.md) · [贡献指南](CONTRIBUTING.md) · [第三方许可归属](THIRD_PARTY_NOTICES.md)

## 案例画廊 (Showcase Gallery)

基于 H3 Turbo Runtime 的多模态能力与代表性生成案例（全部视频已包含在 [`demos/`](demos/README.md) 中可直接查验播放）：

### 1. 诺拉回身 (Nora Turn) · 开源基线 vs 我方优化对比

| 任务模式 | 规格与步数 | 核心特性说明 | 样片原文件 |
|---|---|---|---|
| 首尾帧引导 (FL2VA) | 832×480, 5s (120帧), 4 步 | 输入相同首尾帧（种子 4106），测试官方基线与我方优化的生成一致性、速度与显存。 | [上游基线](demos/videos/A03-诺拉-4步-上游T4.mp4) · [我方优化](demos/videos/A04-诺拉-4步-我方优化4060.mp4) |

<table>
<tr>
<th width="50%" align="center">公开开源基线 (LightX2V Turbo · Tesla T4)<br>4 步耗时 1,257.67 秒 · 显存 14.31 GiB</th>
<th width="50%" align="center">H3 Turbo Runtime (我方优化 · RTX 4060)<br>4 步耗时 159.18 秒 · 显存 5.86 GiB</th>
</tr>
<tr>
<td align="center">

https://github.com/user-attachments/assets/dc8de87b-22d6-4176-b886-feda1fd203c9

</td>
<td align="center">

https://github.com/user-attachments/assets/b42f312d-52a6-4ad5-95a4-8a05eb729217

</td>
</tr>
</table>

### 2. 单镜头 15 秒原生生成 (再来一次)

| 任务模式 | 规格与步数 | 核心特性说明 | 样片原文件 |
|---|---|---|---|
| 4 关键帧时序引导 (FL2VA) | 832×480, 15s (360帧), 4 步 Stream | 单次联合去噪生成完整的 15 秒连贯音视频（360 帧），由 4 处时序关键帧约束镜头节奏。 | [样片](demos/videos/C01-再来一次-最终15秒-480p有声.mp4) |

https://github.com/user-attachments/assets/6e4717a6-9945-484d-9175-64ad68eb49d2

### 3. 飞船舰长 (Starship Captain)

| 任务模式 | 规格与步数 | 核心特性说明 | 样片原文件 |
|---|---|---|---|
| 纯文生音视频 (T2VA) | 832×480, 10s (240帧), 4 步 Fast | 纯文本提示词驱动生成，呈现舱内光影变换与背景空间环境音。 | [样片](demos/videos/B02-飞船舰长-我方Turbo.mp4) |

https://github.com/user-attachments/assets/1737891d-8ef4-4cfa-8b7e-f7779edf75ef

### 4. 拉面馆家庭 (Ramen Family)

| 任务模式 | 规格与步数 | 核心特性说明 | 样片原文件 |
|---|---|---|---|
| 时间轴多关键帧引导 | 832×480, 8s (192帧), 4 步 Fast | 在时间轴指定位置嵌入关键帧，引导室内多人物互动动作与背景声效。 | [样片](demos/videos/B04-拉面家庭-我方Turbo.mp4) |

https://github.com/user-attachments/assets/3bafdd3e-0efd-4135-bdfb-5b141d4d36ca

### 5. 奶龙笑声 (Ref2VA)

| 任务模式 | 规格与步数 | 核心特性说明 | 样片原文件 |
|---|---|---|---|
| 跨模态角色与音色参考 | 832×480, 5s (120帧), 8 步 Ref2VA | 输入单张角色参考图与参考音频，生成保持角色外形与笑声声线的动态片段。 | [样片](demos/videos/C02-奶龙-Ref2VA-有声.mp4) |

https://github.com/user-attachments/assets/b38c0cb4-e39c-4289-855c-61866b0f57e7

### 6. 漫剧微电影 (赎命)

| 任务模式 | 规格与步数 | 核心特性说明 | 样片原文件 |
|---|---|---|---|
| 多镜头叙事剪辑 (12 镜头) | 原生 832×480 (5s×12镜头分段生成)，44.96s | 由 12 个 5 秒原生 832×480 镜头分段生成，经定剪组合为 44.96 秒连贯漫剧，全片保留 12 段原生有声音轨。 | [样片](demos/videos/C03-赎命-最终44秒-480p原声.mp4) |

<details>
<summary>展开 44.96 秒完整实测成片在线播放器（减轻页面初始加载）</summary>

https://github.com/user-attachments/assets/dfae256b-6761-4013-9d57-f74e7e467e4c

</details>

## 性能对比

使用 **832×480、24 fps、首尾帧引导（FL2VA）** 的代表性任务。公开基线为 LightX2V 发布的 **INT8 ConvRot Turbo 工作流**；本项目使用社区 W4A8 主干与同一份 Turbo LoRA，并非两套完全相同的量化部署。表内显存均为 **GPU 整卡峰值**，不是 PyTorch 分配量；“去噪/采样”与“生成到成片”的计时边界不同，详细记录见[跨设备 Benchmark](benchmarks/README.md)。

### 5 秒：主要性能基准

| 实现 | GPU | 步数 | 去噪 / 采样 | 生成到成片 | 显存峰值 |
|---|---|---:|---:|---:|---:|
| 公开 LightX2V Turbo（INT8 ConvRot） | Tesla T4 16GB | 4 | 1,099.73 秒 | 1,257.67 秒 | 14.31 GiB |
| **H3 Turbo Runtime** | Tesla T4 16GB | 4 | **168.92 秒** | **~250 秒*** | **3.95 GiB** |
| **H3 Turbo Runtime** | RTX 4060 Laptop 8GB | 4 | **118.92 秒** | **159.18 秒** | **5.86 GiB** |
| 公开 LightX2V Turbo（INT8 ConvRot） | Tesla T4 16GB | 8 | 2,126.70 秒 | 2,287.90 秒 | 14.31 GiB |
| **H3 Turbo Runtime** | Tesla T4 16GB | 8 | **558.19 秒** | **670.12 秒** | **4.08 GiB** |
| **H3 Turbo Runtime** | RTX 4060 Laptop 8GB | 8 | **223.22 秒** | **269.87 秒** | **5.33 GiB** |

<small>说明：四步 4060 的 118.92 秒是采样阶段耗时，159.18 秒为从条件就绪到有声成片的完整耗时。*四步 T4 采样为 168.92 秒，历史成片实测仅包含无声画面轨约 248 秒（按同环境 VAE 解码约 80 秒折算，若含音频混流完整成片约 250 秒左右）。上游和优化版在环境、量化主干及部分计时边界上不同，表中加速属于部署级观测对比。</small>

### 15 秒：长视频、单次原生生成

| 实现 | GPU | 步数 | 四步去噪 | 采样阶段 | 显存峰值 |
|---|---|---:|---:|---:|---:|
| **H3 Turbo Runtime** | Tesla T4 16GB | 4 | **986.90 秒** | 1,017.26 秒 | **6.17 GiB** |
| **H3 Turbo Runtime** | RTX 4060 Laptop 8GB | 4 | **586.79 秒** | **592.81 秒** | **6.83 GiB** |

两条记录均已完成 15 秒原生音视频解码与 MP4 导出。**15 秒是一次联合去噪，而非 3×5 秒拼接。** 表格使用同一首尾帧测试镜头；额外的多时间关键帧演示与独立计时见[完整验证记录](benchmarks/README.md)。

<small>说明：15 秒去噪时间包含四次模型计算，采样阶段包含准备与调度装载（4060 实测 592.81 秒，含后续音视频解码总成片约 10 分 38 秒）。RTX 4060 四关键帧示例带条件编码全流程合计约 11 分 38 秒。</small>

### 功能与可配置时长

- **输入模式**：文生视频、单图引导、首尾帧引导及多个指定时间位置的关键帧，沿用 H3 的原生条件接口。
- **时长**：使用 `--duration SECONDS` 按需指定，常用的 **4–15 秒**无需切换模型或重新准备另一套权重；默认 5 秒。原生帧数按 H3 的 `17k+5` 网格对齐，音频和视频按同一目标时长导出。上游训练长度主要约为 5–15 秒。
- **生成模式**：默认 FL2VA Turbo 4/8 步、Ref2VA 推荐 8 步。调用方显式选 Fast/Stream，默认 Fast；不支持、OOM 或内核失败不自动切换后端。
- **代表性性能测量**：上表用 5 秒和 15 秒作为短/长视频测试样例，其他可配置值无需一一列举。

**原生条件输入：** 支持文字生成、单张首／尾帧、首尾帧、多时间图像／短视频／音频锚点，以及参考图片、视频和音频；可以组合使用。以上均沿用 H3 原生条件节点和模型内部表示。详见[ComfyUI 工作流及输入格式](integrations/comfyui/README.md)。

## 模型与权重来源

1. **MiniMax H3-Base-FL2VA** 是 MiniMax 的基础音视频模型。FL2VA 指首尾帧条件路径，也可用零张或一张图输入。原生音视频生成来自这一层。 [MiniMax H3 模型规格](https://github.com/MiniMax-AI/MiniMax-H3#model-variants-and-input-specifications)
2. **LightX2V H3 Turbo** 是基于 MiniMax-H3 的蒸馏 LoRA，把原生几十步去噪压缩到 4 或 8 步，由 LightX2V 团队提供。 [LightX2V 项目主页](https://github.com/lightx2v/lightx2v)
3. **W4A8 开源主干** 沿用开源社区对 MiniMax-H3-Base-FL2VA 的权重打包。
4. **音频与视频解码** 使用官方配套的 INT8 视频 VAE 与 FP32 音频 VAE。
5. **提示词与视觉条件** 使用 Qwen3-VL 4B INT8 模型编码文本与输入图，并通过投影层与 DiT 对齐。

全部六份权重的文件名、来源仓库、SHA-256 和加载用途见[模型资产表](runtime/models/manifest.json)。

## 优化手段与执行策略

| 环节 | 当前实现方式 |
|---|---|
| 主干模型权重 | 使用社区 W4A8 检查点。将四位分组权重在需要时解压到 INT8，由前馈层分块计算，相邻行块间复用解压缓冲区。 |
| 激活与矩阵乘法 | 前馈层沿用现有的 ConvRot 分组旋转和行级 INT8 激活量化。适配后的计算内核在 INT32 中累加，并在写回时应用行列缩放；Ada 路径保持参考实现的 FP32 输出和计算顺序。 |
| Turbo 适配器 | 使用 LightX2V 少步蒸馏 LoRA 权重，逐层叠加浮点残差，保持量化主干的内存排布不变。 |
| 前馈网络与注意力 | 按行块成对处理前馈计算，在内存布局允许处重叠矩阵乘法、激活或写回操作；基础前馈路径与 LoRA 路径分别保持各自的激活计算。在融合旋转嵌入和写回前保持查询和键的归一化。 |
| 资源生命周期管理 | 将条件准备、去噪、视频解码和音频解码分阶段执行，前序权重和临时工作区在使用后及时释放。视频解码使用 INT8 ConvRot 检查点分块重叠解码；音频解码使用上游 FP32 检查点。 |
| 新输入条件准备 | 运行时使用社区 Qwen3-VL 4B INT8 视觉语言模型与文本投影层，在提取目标特征时截取前 24 层。 |

可迁移的部分是分阶段资源管理、权重和适配器驻留策略、分块融合及配对验证。接入新模型时，需要重新适配量化布局、激活和适配器的算术、潜变量接口及目标 GPU 算子，并用相同输入测时、显存和画质。当前公开实测使用 RTX 4060 Laptop（SM89）。

**可选 Ref2VA 主干：** 支持使用 joeygambino 的 `MiniMax-H3-ref2va-pruned-comfy-w4a8.safetensors` 与 LightX2V Ref2VA Turbo 8-step LoRA，替换当前任务所需的 FL2VA 主干/LoRA 组合；Qwen3-VL、视频和音频 VAE 不需要重复安装。`--variant ref2va --steps 8` 启用该组合，默认 `--variant fl2va` 不变。可通过 `uv run python scripts/fetch_weights.py --fetch --variant ref2va` 安装相应权重，或使用同目录下的 SHA-256 校验值自行取得权重。详细链接见[Ref2VA 权重表](runtime/models/ref2va.json)。

## 部署与复现

运行时接口设计保持通用性，支持通过 CLI 命令行或 ComfyUI 节点进行灵活调用。系统内置两种执行策略：
- **Fast 模式**：基于 Triton 148 寄存器融合算子加速，当前在 NVIDIA Ada 架构（SM 8.9，如 RTX 4060）完成深度调优与验收。
- **Stream 模式**：基于分头流式注意力机制，适用于更低显存占用或长视频生成场景。

环境依赖：CUDA、PyTorch、Triton、comfy-kitchen、ComfyUI 与 FFmpeg。项目推荐使用 `uv` 进行统一环境管理：

项目推荐使用 `uv` 进行统一环境管理：

```bash
# 1. 环境初始化与依赖同步（由 uv 管理）
uv sync

# 2. 自动化测试套件验证
uv run pytest -v

# 3. 检查现有环境（不启动视频生成）
uv run python deploy.py check --comfy /path/to/ComfyUI --models /path/to/weights

# 4. 默认示例：5 秒、4 步、首尾帧条件
uv run python deploy.py render --comfy /path/to/ComfyUI --models /path/to/weights \
  --condition /path/to/5s-condition.safetensors --out runs/clip-5s

# 5. 可变时长示例：15 秒一次生成，单独准备 15 秒条件
uv run python deploy.py render --comfy /path/to/ComfyUI --models /path/to/weights \
  --duration 15 --strategy stream --condition /path/to/15s-condition.safetensors --out runs/clip-15s
```

CLI：duration 默认 5 秒，width/height 默认 832×480，output-fps 默认 24（原生模型仍以 24fps 推理）。FL2VA Turbo 4/8 步，Ref2VA 推荐 8 步；strategy 默认 fast，允许显式选择 stream，不自动回退。head-group/reserve-vram 为调优选项；adapter 与 recipe-file 供开发者选择兼容的 LoRA 及采样配方。不同生成时长与空间尺寸需要重新编码对应条件。项目内有 [运行时调优手册](.agents/skills/h3-runtime-optimization/SKILL.md)，覆盖 GPU、时长、LoRA 与采样配方。

**硬件适配：** 运行环境支持通过 `--strategy` 显式指定 Fast 或 Stream 模式，不会隐式回退。如需适配至其他 GPU 架构，请参考 [运行时调优手册](.agents/skills/h3-runtime-optimization/SKILL.md)。

## ComfyUI 工作台

源码节点位于 [`custom_nodes/h3_turbo_runtime/`](custom_nodes/h3_turbo_runtime/)，可将其链接到 ComfyUI 的 `custom_nodes/`。先设置权重目录；使用独立 Python 环境或依赖目录时，可以额外配置：

```bash
export H3_RUNTIME_MODELS=/path/to/weights
# 可选：仅当运行时与 ComfyUI 使用不同 Python 或依赖目录时设置
export H3_RUNTIME_PYTHON=/path/to/python
export H3_RUNTIME_DEPS=/path/to/runtime-deps
# 可选：条件编码权重不在仓库时，指向实际目录
export H3_RUNTIME_WORKBENCH=/path/to/workbench
uv run python deploy.py configure-comfy --comfy /path/to/ComfyUI
```

在 ComfyUI 的 **H3 Turbo Runtime** 节点输入文本、可选首尾图、时长（默认5秒）与步数。可在 `guides_json` 中添加其他时间的图像引导，例如 `[{"at_seconds":5,"image":"frame5.png"}]`。图片先上传到 ComfyUI 输入目录。文本与图像条件由外部 [ComfyUI-ClipProj](https://github.com/nicolab28/ComfyUI-ClipProj) 插件和 `workbench/condition-models` 生成；安装步骤见 [ComfyUI 接口与工作流](integrations/comfyui/README.md#clipproj-conditioning-plugin-external)。

## 权重、许可与复现

使用 `uv run python scripts/fetch_weights.py --fetch` 一次性获取全部六份原作者权重，自动核对文件大小与 SHA-256（共约 23.67 GB）。本仓库提供完整下载与运行配方，不重复存储原始权重。模型版本、SHA-256、输入规格和验证依赖见[权重与复现清单](benchmarks/reproduction-manifest.json)。性能对照与测量口径见 [Benchmark](benchmarks/README.md)；具体优化和数值差异见[实现说明](benchmarks/implementation.md)与[画质证据](benchmarks/quality.md)。

本项目的代码许可、上游权重许可和模型在线服务条件需要分别遵守。[第三方许可归属](THIRD_PARTY_NOTICES.md)

## 社区贡献

主要收两类：你用这套 runtime 做出的作品，和你在自己设备上的适配与优化记录。也欢迎性能优化、工具与测试等改动。只要能提供大致可复现的内容即可；视频可选，不限制提示词、题材与格式。目录结构与提交方式见 [社区贡献说明](community/README.md)。

## 社区与致谢

感谢 [LINUX DO](https://linux.do) 社区提供开放、友善的技术交流平台。
