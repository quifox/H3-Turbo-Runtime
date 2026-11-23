# 适配器与配方兼容性

| 组合 | 状态与动作 |
|---|---|
| H3 + 匹配的 Turbo 适配器，少步采样 | 已验证的默认组合 |
| 同一适配器的 4 步 / 8 步 | 步数是运行时选择；少步能力取决于蒸馏，换步数要重验画质 |
| 采样配方（sampler、scheduler、shifts） | 跟适配器作者走，不能从形状反推 |
| FL2VA 与 Ref2VA 互换 | 权重、适配器、参考哈希都不通用 |
| 不同 rank/alpha，完整 FFN 与 attention 目标 | 支持形状检查；GPU 正确性另验 |
| 部分 FFN 目标 | Stream 可能少算；Fast 硬要求全目标 |
| 部分 QKV/out 目标 | Stream 需要匹配；否则失败 |
| 常规风格/主体 LoRA | 需要作者采样配方与 base 身份 |
| 主模型无适配器 | 未验证路径 |
| 其他架构或模型 | 需证明量化、模块、形状、采样一致 |

来源：`runtime/adapter_spec.py`、`runtime/src/lora_adapter.py`、`runtime/src/fc1_ada.py`、`runtime/models/recipes.json`、`benchmarks/quality.md`。
