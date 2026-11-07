"""Best-effort admission checks for native SAM allocations; no Torch import."""
import math
import os
from pathlib import Path

GIB = 1024**3


class MemoryLimitError(RuntimeError):
    """A resource limit stopped tracking without publishing partial labels."""


def available_ram():
    """Linux available RAM, also respecting an exposed cgroup memory limit."""
    try:
        values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        available = int(values["MemAvailable"].split()[0])*1024
    except (OSError, KeyError, ValueError):
        try:
            available = os.sysconf("SC_AVPHYS_PAGES")*os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError):
            return None
    for directory, limit_name, used_name in (("/sys/fs/cgroup", "memory.max", "memory.current"),
                                              ("/sys/fs/cgroup/memory", "memory.limit_in_bytes", "memory.usage_in_bytes")):
        try:
            folder = Path(directory)
            limit, used = int((folder/limit_name).read_text()), int((folder/used_name).read_text())
            available = min(available, max(0, limit-used))
        except (OSError, ValueError):
            pass
    return max(0, available)


def reserve(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number of GiB")
    return float(value)


class MemoryGuard:
    def __init__(self, *, ram_reserve_gb=2., vram_reserve_gb=1.):
        self.ram_reserve_gb = reserve(ram_reserve_gb, "ram_reserve_gb")
        self.vram_reserve_gb = reserve(vram_reserve_gb, "vram_reserve_gb")
        self.last = {}

    def check(self, stage, *, torch=None, device="cuda", required_ram=0, required_vram=0):
        available = available_ram()
        self.last = {"stage": stage, "ram_available_bytes": available}
        if self.ram_reserve_gb and available is not None and available < required_ram+self.ram_reserve_gb*GIB:
            need = f"; native frame loading alone needs about {required_ram/GIB:.2f} GiB more" if required_ram else ""
            raise MemoryLimitError(f"RAM guard stopped {stage}: {available/GIB:.2f} GiB available{need}; "
                f"keeping {self.ram_reserve_gb:g} GiB free. Use --objects-per-pass 1 or --state-device cuda, "
                "free RAM, or limit a trial with --max-frames. Saved references and completed results are retained.")
        if self.vram_reserve_gb and device != "cpu" and torch is not None:
            cuda = torch.cuda
            if hasattr(cuda, "mem_get_info"):
                free, total = cuda.mem_get_info(device)
                # Only flush cached allocations when actually under pressure.
                if free < required_vram+self.vram_reserve_gb*GIB:
                    cuda.empty_cache()
                    free, total = cuda.mem_get_info(device)
                self.last.update(vram_free_bytes=free, vram_total_bytes=total)
                if free < required_vram+self.vram_reserve_gb*GIB:
                    need = f"; model loading is budgeted at {required_vram/GIB:.2f} GiB more" if required_vram else ""
                    raise MemoryLimitError(f"VRAM guard stopped {stage}: {free/GIB:.2f} GiB free{need}; "
                        f"keeping {self.vram_reserve_gb:g} GiB free. Use --state-device cpu or "
                        "--objects-per-pass 1 and release other GPU jobs. Saved references and completed results are retained.")
        return self.last.copy()


def is_allocation_error(error, torch=None):
    types = [MemoryError]
    if torch is not None:
        for owner in (torch, torch.cuda):
            kind = getattr(owner, "OutOfMemoryError", None)
            if isinstance(kind, type):
                types.append(kind)
    message = str(error).lower()
    return isinstance(error, tuple(types)) or "cuda out of memory" in message or "defaultcpuallocator" in message
