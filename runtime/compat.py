"""GPU capability and installed-version reporting for H3 Turbo inference."""
from dataclasses import dataclass
from importlib.metadata import version


@dataclass(frozen=True)
class DeviceProfile:
    capability: tuple[int, int]
    name: str
    family: str
    memory_gib: float
    qualified_fast: bool


def profile_from_capability(capability, *, name="GPU", memory_bytes=0):
    major, minor = tuple(map(int, capability))
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


def inspect_environment(*, torch_module=None):
    """Read GPU properties and installed package versions; do not gate on them."""
    import torch as real_torch
    torch = torch_module or real_torch
    info = torch.cuda.get_device_properties(0)
    profile = profile_from_capability((info.major, info.minor), name=info.name,
                                      memory_bytes=info.total_memory)
    installed = {k: version(k) for k in ("torch", "triton", "comfy-kitchen")}
    return profile, installed
