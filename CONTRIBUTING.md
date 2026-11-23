# 贡献指南 (Contributing Guide)

简体中文 · [English](CONTRIBUTING.en.md)

感谢你对 H3 Turbo Runtime 项目的关注与支持！我们欢迎社区开发者参与贡献，共同推动开源低显存多模态音视频推理生态的发展。

在开始贡献之前，请务必完整阅读我们的架构契约与工程红线：
👉 **[AGENTS.md](AGENTS.md) — Contributor and Agent Guidelines**

---

## 1. 欢迎的贡献方向

我们特别鼓励以下领域的贡献：

1. **新硬件架构与算子适配**：
   - 针对 Turing (SM75)、Ampere (SM80/SM86)、Hopper (SM90) 或 AMD ROCm/Triton 等架构进行算子调优与适配。
   - 请遵循 [GPU 架构适配指南](.agents/skills/h3-gpu-backend-porting/SKILL.md)。
2. **LoRA 适配器与采样配方**：
   - 接入新的社区微调权重、更少步数的 Turbo LoRA 蒸馏模型或自定义调度器。
   - 请遵循 [LoRA 配方适配指南](.agents/skills/h3-lora-recipe-adaptation/SKILL.md)。
3. **ComfyUI / CozyUI 交互与工作流**：
   - 改进节点易用性、扩展多模态条件输入控制、提供经过验证的生产级工作流模板。
4. **自动化测试与工程健壮性**：
   - 增加边缘用例覆盖、强化跨平台（Linux、Windows、macOS）接口与数据类型契约。
5. **文档与基准数据**：
   - 提供不同硬件设备的端到端测速与显存占用数据（需附带硬件配置与环境日志）。
   - 完善部署教程、修正文档勘误与国际化翻译。

> [!NOTE]
> **不接受的改动范围**：
> - 任何引入隐式降级/自动回退（如 Fast 失败自动转 Stream）的代码变更（违反核心架构契约）。
> - 直接提交大型权重文件、压缩包或未经验证的视频到 Git 仓库树（权重请使用脚本自动拉取，样片走外部资产或 PR 附件）。
> - 引入未明确开源许可或包含商业闭源限制的第三方资产。

---

## 2. 开发环境与准备工作

项目采用 `uv` 作为唯一权威的环境与依赖管理工具。

### 初始化开发环境

```bash
# 1. Fork 并克隆代码仓库
git clone https://github.com/<your-username>/H3-Turbo-Runtime.git
cd H3-Turbo-Runtime

# 2. 依赖同步与虚拟环境创建（严格使用 uv 管理，禁止裸 pip install）
uv sync

# 3. 运行自动化测试套件
uv run pytest -v
```

> [!TIP]
> **轻量级/无显卡贡献友好**：
> 项目中的工作流 Schema 校验、节点参数契约、文档本地链接测试及语法检查均**无需 GPU 硬件**。任何开发者在普通 CPU 环境下运行 `uv run pytest -v` 即可完成核心代码与文档的本地验证。

### 权重复现清单（可选）

如需在本地进行全量推理或生成验证，请通过自动化脚本获取已核验 SHA-256 的权重：
```bash
uv run python scripts/fetch_weights.py --fetch
```

---

## 3. Pull Request 提交流程

### 第一步：分支管理
- 始终基于主仓库最新的 `main` 分支拉取开发分支：
  ```bash
  git checkout -b feat/your-feature-name
  ```

### 第二步：提交信息规范 (Conventional Commits)
请使用清晰、具有描述性的提交信息，推荐遵循标准前缀：
- `feat:` 新功能或新特性
- `fix:` 缺陷修复
- `perf:` 性能与算子吞吐优化
- `docs:` 文档变更、注释更新
- `test:` 单元测试、契约测试增强
- `refactor:` 代码重构（不改变外部接口与功能）

### 第三步：本地质量门禁自检
在提交 PR 之前，请确保本地通过以下校验：
1. **全量测试通过**：
   ```bash
   uv run pytest -v
   ```
2. **零隐私信息泄露（核心红线）**：
   检查代码改动中不得包含任何个人用户名、内网私有 IP（`192.168.*`、`10.*`）或本地绝对路径。所有路径均应通过环境变量或相对路径解析。
3. **接口契约一致**：
   若修改了 ComfyUI 节点，必须同步保证 `H3TurboRuntime`、`H3TurboSettings`、`H3TurboMediaCondition` 与 `integrations/comfyui/complete-5s-workflow.json` 完全对称。

### 第四步：开启 Pull Request
- 在 GitHub 页面开启针对 `main` 分支的 PR。
- 简明扼要说明改动意图、涉及模块及验证方式。
- 若包含算子或吞吐优化，请在 PR 说明中附带微基准测速（Micro-benchmark）数据与显存峰值对比。

---

## 4. Issue 提报指南

### 报告 Bug
在提交 Bug 之前，请先通过搜索确认该问题未被重复提报。提交 Issue 时请提供：
1. **硬件与系统环境**：GPU 型号、驱动版本、CUDA 版本、操作系统（Linux 发行版 / WSL2 / Windows）。
2. **执行命令与参数**：完整的 CLI 命令或使用的 ComfyUI 工作流截图。
3. **完整错误日志**：保留未截断的完整 Python Traceback 输出。
4. **运行策略**：明确指出当前使用的是 `--strategy fast` 还是 `--strategy stream`。

### 功能与硬件适配提议
欢迎在 Issues 中提出对新硬件平台、新采样算法或新架构的适配建议，请详细描述目标场景、硬件配置参考以及预期的交互方式。
