# Backend contracts and tuning

| Setting | Classification | Rule |
|---|---|---|
| Explicit Fast/Stream | Public user API | Do not auto-route or silently fall back |
| GPU SM89 for current Fast | Current kernel qualification | Other architectures require explicit port/validation |
| W4A8/ConvRot grouping, canonical quantized layout | Weight/operator contract | Validate against checkpoint format |
| H3 56 heads, head dim 128, MLP dimensions | Model invariant | Read from model, do not improvise |
| FC1/FC2 paired 1024 rows, Triton tile 64x32x64, warps/stages | Backend implementation | Retune only as a complete kernel A/B |
| Head groups | Stream backend option | Must divide current H3 head count |
| Duration and width/height | Public user API | Match H3 latent shapes; no artificial 5s/4060 limitation |
| Output FPS | Public user API | Export FPS is independent of native H3 24 FPS generation |
| VRAM reserve and threading | Deployment choices | Default measured on 4060; not a universal optimum |
| Video VAE overlap | Decoder detail | 64 to 32 measured faster but changes RGB output |

A normal incomplete tile within a selected algorithm is not a fault recovery. However, selecting stock FFN or Stream because Fast rejects an input is prohibited. Keep hard kernel correctness checks; remove only irrelevant device-model or test-case caps.

Do not attribute the entire LightX2V INT8 vs Runtime W4A8 system speedup to a specific kernel: it mixes checkpoint format, GPU staging, arithmetic and execution scheduling.
