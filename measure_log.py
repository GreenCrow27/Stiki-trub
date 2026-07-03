"""CSV-лог измерений (скорость, время, стык)."""
import csv
import os
from datetime import datetime

from paths import writable_path


class MeasureCsvLog:
    def __init__(self, cfg: dict):
        c = cfg or {}
        self.enabled = bool(c.get("enabled", False))
        self.path = writable_path(str(c.get("file", "logs/measures.csv")))
        self._header = (
            "time", "dist_mm", "dt_ms", "speed_reg", "speed_m_s",
            "x_left", "x_right", "state",
        )

    def _ensure_dir(self):
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)

    def append(self, dist_mm, dt_ms, speed_reg, speed_m_s, x_left=0, x_right=0, state=""):
        if not self.enabled:
            return
        self._ensure_dir()
        new_file = not os.path.isfile(self.path)
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(self._header)
            w.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                int(dist_mm), int(dt_ms), int(speed_reg), speed_m_s,
                int(x_left), int(x_right), state,
            ])
        print("[log] {}".format(self.path))
