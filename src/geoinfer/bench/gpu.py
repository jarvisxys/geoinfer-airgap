"""NVML sampler: GPU utilisation, memory, SM clock and temperature at a fixed rate.

Plan section 10: mean ``utilization.gpu`` and peak ``memory.used`` over the measured
window, sampled at 10 Hz; clock and temperature expose thermal throttling (laptop).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self


@dataclass(frozen=True)
class GpuSample:
    """One NVML reading."""

    util: float  # percent
    mem_mib: float  # memory.used, whole device
    clock_mhz: float  # SM clock
    temp_c: float


def nvml_reader(index: int = 0) -> Callable[[], GpuSample | None]:
    """A reader for GPU ``index`` (NVML numbering, ignores CUDA_VISIBLE_DEVICES)."""
    import pynvml

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(index)

    def read() -> GpuSample | None:
        try:
            u = pynvml.nvmlDeviceGetUtilizationRates(h)
            m = pynvml.nvmlDeviceGetMemoryInfo(h)
            c = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
            t = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        except pynvml.NVMLError:
            return None
        return GpuSample(float(u.gpu), m.used / 2**20, float(c), float(t))

    return read


def gpu_processes(index: int = 0) -> list[dict[str, Any]]:
    """Compute processes on GPU ``index`` (pid, MiB): logged before and after each run."""
    import pynvml

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(index)
    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
    return [{"pid": p.pid, "mib": round((p.usedGpuMemory or 0) / 2**20)} for p in procs]


class GpuSampler:
    """Background thread that calls ``read`` ``hz`` times a second while active.

    Parameters
    ----------
    read
        Returns one sample, or None when the reading failed (skipped).
    hz
        Sampling rate.
    """

    def __init__(self, read: Callable[[], GpuSample | None], hz: float = 10.0) -> None:
        self.read, self.period = read, 1.0 / hz
        self.samples: list[GpuSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            s = self.read()
            if s is not None:
                self.samples.append(s)
            self._stop.wait(self.period)

    def __enter__(self) -> Self:
        self.samples.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def summary(self) -> dict[str, float]:
        """Mean utilisation, peak memory, mean / min clock, max temperature."""
        s = self.samples
        if not s:
            return {"samples": 0}
        n = len(s)
        return {
            "samples": n,
            "util_mean": round(sum(x.util for x in s) / n, 1),
            "mem_peak_mib": max(x.mem_mib for x in s),
            "clock_mean_mhz": round(sum(x.clock_mhz for x in s) / n),
            "clock_min_mhz": min(x.clock_mhz for x in s),
            "temp_max_c": max(x.temp_c for x in s),
        }
