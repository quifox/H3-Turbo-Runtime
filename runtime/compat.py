"""GPU and dependency capability checks for H3 Turbo inference.

Supported architecture != verified output quality. The Ada SM89 fast path is
qualified; other architectures require an explicitly selected, supported
backend. Never choose a different execution engine silently. The package manager is deliberately not
asked to install or downgrade packages at runtime.
"""
from dataclasses import dataclass
from importlib.metadata import version
import warnings


@dataclass(frozen=True)
class DeviceProfile:
    capability: tuple[int, int]
    name: str
    family: str
    memory_gib: float
    qualified_fast: bool


REQUIRED_PUBLIC_API = (
    "int8_attention", "quantize_int8_rowwise", "int8_linear", "w4a8_int8_linear",
)
REQUIRED_STREAM_HELPERS = ("_int8_weight_scale_arg", "_dequant_int4_grouped_to_int8")


def profile_from_capability(capability, *, name="GPU", memory_bytes=0):
    major, minor = tuple(map(int, capability))
    if (major, minor) < (7, 5):
        raise RuntimeError(f"GPU {name} (SM {major}.{minor}) is below the required SM75 INT8 kernel capability")
    if major == 7:
        family = "turing"
    elif major == 8:
        family = "ampere" if minor < 9 else "ada"
    elif major == 9:
        family = "hopper"
    elif major >= 10:
        family = "blackwell_or_newer"
    else:
        family = "other"
    qualified = (major, minor) == (8, 9)
    return DeviceProfile((major, minor), name, family, round(memory_bytes / 2**30, 3), qualified)


def choose_strategy(profile, *, requested="fast"):
    """Return the caller's backend unchanged; the selected kernel validates its own requirements."""
    if requested not in ("fast", "stream"):
        raise ValueError("strategy must be explicitly fast or stream")
    return requested


def inspect_environment(*, tested=None, torch_module=None):
    """Read actual capabilities; report pinned versions as guidance, not limits.

    Private kitchen symbols remain necessary for the installed W4A8 kernels;
    validate them explicitly until they can be replaced with public APIs.
    """
    import comfy_kitchen
    from comfy_kitchen.backends import cuda as native
    import torch as real_torch
    torch = torch_module or real_torch
    missing_public = [x for x in REQUIRED_PUBLIC_API if not callable(getattr(comfy_kitchen, x, None))]
    missing_stream = [x for x in REQUIRED_STREAM_HELPERS if not callable(getattr(native, x, None))]
    if missing_public or missing_stream:
        raise RuntimeError(f"Installed comfy-kitchen lacks required W4A8 APIs: public={missing_public}, helpers={missing_stream}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable to PyTorch")
    info = torch.cuda.get_device_properties(0)
    profile = profile_from_capability((info.major, info.minor), name=info.name,
                                      memory_bytes=info.total_memory)
    # A device-name or wheel arch-string equality check is not a reliable
    # portability test: newer wheels can JIT kernels from embedded PTX.
    # Probe a tiny, real CUDA operation instead of blocking new architectures.
    with torch.cuda.device(0):
        probe = torch.tensor([2.0], device="cuda")
        result = (probe * 2.0).item()
    if result != 4.0:
        raise RuntimeError("Unexpected CUDA device arithmetic result")
    tested = tested or {}
    installed = {k: version(k) for k in ("torch", "triton", "comfy-kitchen")}
    for pkg, pinned in tested.items():
        # Wheel-local CUDA suffixes are not included in importlib.metadata's
        # public version on every platform. Runtime CUDA is checked separately.
        if pkg in installed and installed[pkg].split("+")[0] != pinned.split("+")[0]:
            warnings.warn(f"{pkg} {installed[pkg]} differs from tested {pinned}; checking required APIs instead", stacklevel=2)
    if not profile.qualified_fast:
        warnings.warn(f"SM {info.major}.{info.minor} has no validated Fast backend; choose a supported execution mode explicitly", stacklevel=2)
    return profile, installed
