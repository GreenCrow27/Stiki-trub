"""Запись оригинального видео с камеры (без ROI/оверлея)."""
import os
import time
from datetime import datetime

import cv2

from paths import writable_path


def _fourcc(name: str) -> int:
    name = (name or "mp4v").strip()
    if len(name) != 4:
        name = "mp4v"
    return cv2.VideoWriter_fourcc(*name)


class RawVideoRecorder:
    """
    Пишет кадры как есть (BGR), без отрисовки.
    fps в config — метаданные файла (0 → stream_fps или 25).
    """

    def __init__(self, cfg: dict):
        c = cfg or {}
        self.enabled = bool(c.get("enabled", False))
        self.output_dir = str(c.get("output_dir", "recordings"))
        self.prefix = str(c.get("prefix", "camera"))
        self.codec = str(c.get("codec", "mp4v"))
        self.target_fps = float(c.get("fps", 0))
        self.fallback_fps = float(c.get("fallback_fps", 25))
        self.max_minutes = float(c.get("max_minutes", 0))
        self._writer = None
        self._path = ""
        self._w = 0
        self._h = 0
        self._frames = 0
        self._t0 = 0.0
        self._last_ts = 0.0
        self._fps_est = self.fallback_fps
        self._stream_fps = 0.0

    def set_stream_fps(self, fps: float):
        if fps and float(fps) > 0:
            self._stream_fps = float(fps)

    def _write_fps(self) -> float:
        if self.target_fps > 0:
            return self.target_fps
        if self._stream_fps > 0:
            return self._stream_fps
        return self.fallback_fps

    @property
    def active(self) -> bool:
        return self._writer is not None

    @property
    def path(self) -> str:
        return self._path

    def _out_path(self) -> str:
        out_dir = writable_path(self.output_dir)
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = ".avi" if self.codec.upper() in ("XVID", "MJPG", "DIVX") else ".mp4"
        return os.path.join(out_dir, "{}_{}{}".format(self.prefix, stamp, ext))

    def start(self, width: int, height: int) -> bool:
        if not self.enabled or self._writer is not None:
            return False
        self._w = int(width)
        self._h = int(height)
        self._path = self._out_path()
        fps = self._write_fps()
        self._writer = cv2.VideoWriter(
            self._path, _fourcc(self.codec), fps, (self._w, self._h)
        )
        if not self._writer.isOpened():
            self._writer = None
            print("[record] не удалось открыть файл: {}".format(self._path))
            return False
        self._frames = 0
        self._t0 = time.time()
        self._last_ts = self._t0
        print("[record] START {} {}x{} fps={:.1f}".format(self._path, self._w, self._h, fps))
        return True

    def write(self, frame) -> bool:
        if not self.enabled or frame is None:
            return False
        h, w = frame.shape[:2]
        if self._writer is None:
            if not self.start(w, h):
                return False
        if w != self._w or h != self._h:
            frame = cv2.resize(frame, (self._w, self._h))
        self._writer.write(frame)
        self._frames += 1
        now = time.time()
        dt = now - self._last_ts
        if dt > 0.001:
            self._fps_est = 0.9 * self._fps_est + 0.1 * (1.0 / dt)
        self._last_ts = now
        if self.max_minutes > 0 and (now - self._t0) >= self.max_minutes * 60.0:
            print("[record] лимит {:.0f} мин — стоп".format(self.max_minutes))
            self.stop()
            return False
        return True

    def stop(self):
        if self._writer is None:
            return
        self._writer.release()
        self._writer = None
        dur = max(0.001, time.time() - self._t0)
        print(
            "[record] STOP {}  frames={}  {:.1f}s  ~{:.1f} fps".format(
                self._path, self._frames, dur, self._frames / dur
            )
        )

    def toggle(self, width: int, height: int) -> bool:
        if self.active:
            self.stop()
            return False
        self.enabled = True
        return self.start(width, height)

    def close(self):
        self.stop()
