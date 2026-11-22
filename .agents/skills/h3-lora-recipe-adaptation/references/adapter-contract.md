# LoRA and recipe compatibility contract

| Combination | Status / action |
|---|---|
| FL2VA W4A8 + LightX2V FL2VA Turbo v1.0, 4/8 steps | Frozen main default and performance reference |
| Ref2VA W4A8 + matching Ref2VA Turbo LoRA | Supported model asset pair; do not borrow FL2VA reference hashes |
| Different rank/alpha, full FFN and attention targets | Metadata interface supports shape-checking; GPU correctness still unverified |
| Partial FFN LoRA | Stream MLP may omit individual deltas; Ada Fast requires full main FFN targets |
| Partial QKV/out LoRA | Head-tiled Stream requires matching targets, otherwise selected mode fails |
| Conventional style/subject fine-tune | Requires matching author-provided sampling recipe and base identity |
| H3 Base without LoRA | Not a currently verified runtime profile; needs dedicated no-adapter path |
| Other architecture/model | Must prove matching quantization, modules, shapes and sampler |

Source:
- runtime/adapter_spec.py checks LoRA A/B/alpha target triplets.
- runtime/src/lora_adapter.py applies adapters without a global fixed 208-target count.
- runtime/src/fc1_ada.py receives LoRA scale as an Ada kernel compile-time parameter, retaining the existing 0.0625 default specialization.
- runtime/models/recipes.json has the default Turbo 4/8 step contract.
- runtime/recipe.py checks recipe shape, not scientific validity.
- benchmarks/quality.md limits claims of numerical equivalence.

Avoid claiming new LoRA portability or quality improvement merely from passing a syntax or metadata test.
