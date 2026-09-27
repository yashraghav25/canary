"""
CPU-load safety guard for training scripts on this machine.

HONEST LIMITATION: this system exposes no real hardware temperature sensor
via any OS-level API (checked psutil.sensors_temperatures — not implemented
on Windows; WMI MSAcpi_ThermalZoneTemperature — empty; Win32_TemperatureProbe
— empty; all standard on consumer laptops without vendor software like
HWiNFO, which requires an admin install this project won't do unasked).

So this guard monitors sustained CPU load % (via Win32_PerfFormattedData_
PerfOS_Processor, confirmed working) as a practical proxy for heat — high
sustained load is the actual thing generating heat, even without a direct
temperature reading. If load stays above a threshold across several checks
in a row, training pauses to cool down, and if it stays pegged for too
long even after pausing, the run aborts rather than continuing blind.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import subprocess
import time

try:
    import psutil
except ImportError:
    psutil = None


def read_cpu_load_percent() -> float | None:
    try:
        out = subprocess.run(
            ["powershell", "-Command",
             "(Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor | "
             "Where-Object {$_.Name -eq '_Total'}).PercentProcessorTime"],
            capture_output=True, text=True, timeout=10,
        )
        return float(out.stdout.strip())
    except Exception:
        return None


def read_ram_usage() -> tuple[float, float] | None:
    """Returns (percent_used, available_gb), or None if unreadable. Uses
    psutil (already a real dependency of this project via pandas/kaggle),
    simpler and more reliable than another PowerShell round-trip."""
    if psutil is None:
        return None
    try:
        vm = psutil.virtual_memory()
        return vm.percent, vm.available / (1024 ** 3)
    except Exception:
        return None


class CPUGuard:
    def __init__(self, warn_threshold: float = 80.0, abort_threshold_streak: int = 6, cooldown_seconds: float = 15.0,
                 ram_warn_percent: float = 90.0):
        self.warn_threshold = warn_threshold
        self.abort_threshold_streak = abort_threshold_streak
        self.cooldown_seconds = cooldown_seconds
        self.ram_warn_percent = ram_warn_percent
        self._high_load_streak = 0
        self._unreadable_streak = 0

    def check(self, context: str = "") -> bool:
        """Call periodically (e.g. once per epoch). Returns False if the
        caller should ABORT the run. Sleeps internally for a cooldown if
        load is high but not yet at the abort threshold."""
        load = read_cpu_load_percent()
        ram = read_ram_usage()
        if ram is not None:
            ram_percent, ram_available_gb = ram
            print(f"[cpu_guard] RAM: {ram_percent:.0f}% used, {ram_available_gb:.1f}GB available "
                  f"({context})")
        else:
            ram_percent = None

        if load is None:
            self._unreadable_streak += 1
            if self._unreadable_streak >= 3:
                print(f"[cpu_guard] Could not read CPU load 3 times in a row during {context} — "
                      f"stopping out of caution rather than running blind.")
                return False
            return True
        self._unreadable_streak = 0

        ram_high = ram_percent is not None and ram_percent >= self.ram_warn_percent
        if load >= self.warn_threshold or ram_high:
            self._high_load_streak += 1
            reason = []
            if load >= self.warn_threshold:
                reason.append(f"CPU load {load:.0f}% (>= {self.warn_threshold:.0f}%)")
            if ram_high:
                reason.append(f"RAM {ram_percent:.0f}% (>= {self.ram_warn_percent:.0f}%)")
            print(f"[cpu_guard] {' and '.join(reason)} during {context} "
                  f"— streak {self._high_load_streak}/{self.abort_threshold_streak}")
            if self._high_load_streak >= self.abort_threshold_streak:
                print(f"[cpu_guard] Sustained high load for {self._high_load_streak} checks — "
                      f"aborting this run rather than risking a lag/overheat lockup or an OOM crash.")
                return False
            print(f"[cpu_guard] Cooling down for {self.cooldown_seconds:.0f}s before continuing...")
            time.sleep(self.cooldown_seconds)
        else:
            self._high_load_streak = 0
        return True
