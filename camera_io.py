"""
Камера: RTSP поток 2 через ffmpeg pipe (низкая задержка).
Вебка: index + DirectShow.
"""
import os
import shutil
import subprocess
import threading
import time
from collections import deque
from typing import List, Optional, Tuple
from urllib.parse import quote

import cv2
import numpy as np


def build_rtsp_url(cam: dict) -> str:
    if cam.get("url"):
        return str(cam["url"])
    login = cam.get("login", "")
    password = cam.get("password", "")
    ip = cam["ip"]
    port = cam.get("port", 554)
    path = cam.get("rtsp_path")
    if not path:
        stream = int(cam.get("stream", 1))
        vendor = str(cam.get("vendor", "")).lower()
        ch = int(cam.get("channel", 1))
        if vendor == "hikvision":
            path = f"/Streaming/Channels/{ch * 100 + stream}01"
        else:
            path = f"/stream{stream}"
    if not path.startswith("/"):
        path = "/" + path
    if login:
        user = quote(str(login), safe="")
        pwd = quote(str(password), safe="")
        return f"rtsp://{user}:{pwd}@{ip}:{port}{path}"
    return f"rtsp://{ip}:{port}{path}"


def build_mjpeg_url(cam: dict) -> str:
    if cam.get("mjpeg_url"):
        return str(cam["mjpeg_url"])
    login = cam.get("login", "")
    password = cam.get("password", "")
    ip = cam["ip"]
    http_port = cam.get("http_port", 80)
    if login:
        return f"http://{login}:{password}@{ip}:{http_port}/mjpeg/stream"
    return f"http://{ip}:{http_port}/mjpeg/stream"


def _safe_label(url: str) -> str:
    if "@" not in url:
        return url
    pre, rest = url.split("@", 1)
    if "://" in pre:
        scheme, auth = pre.split("://", 1)
        user = auth.split(":")[0] if ":" in auth else auth
        return f"{scheme}://{user}:***@{rest}"
    return url


def _ffmpeg_exe(cam: dict) -> Optional[str]:
    p = cam.get("ffmpeg_path")
    if p and os.path.isfile(p):
        return p
    for fn in (_ffmpeg_from_imageio, _ffmpeg_from_path):
        exe = fn()
        if exe:
            return exe
    return None


def _ffmpeg_from_path() -> Optional[str]:
    return shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")


def _ffmpeg_from_imageio() -> Optional[str]:
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        return exe if exe and os.path.isfile(exe) else None
    except Exception:
        return None


def _subprocess_kwargs():
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {}


class _LatestFrame:
    """Последний кадр из фонового чтения."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._stop = threading.Event()
        self._thread = None

    def start(self, target):
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def put(self, frame):
        self.set(frame)

    def set(self, frame):
        with self._lock:
            self._frame = frame

    def get_copy(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def wait_first(self, timeout=10.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                if self._frame is not None:
                    return True
            time.sleep(0.03)
        return False

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)


class _FrameQueue:
    """Очередь кадров RTSP — main читает по порядку, без пропуска стыка."""

    def __init__(self, maxlen=6):
        self._lock = threading.Lock()
        self._deque = deque(maxlen=max(2, int(maxlen)))
        self._stop = threading.Event()
        self._thread = None
        self._first = threading.Event()

    def start(self, target):
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def put(self, frame):
        with self._lock:
            self._deque.append(frame)
            self._first.set()

    def get_copy(self):
        with self._lock:
            if not self._deque:
                return False, None
            return True, self._deque.popleft().copy()

    def get_latest_copy(self):
        """Сбросить очередь и вернуть самый свежий кадр (плавное превью)."""
        with self._lock:
            if not self._deque:
                return False, None
            frame = self._deque[-1]
            self._deque.clear()
            return True, frame.copy()

    def depth(self):
        with self._lock:
            return len(self._deque)

    def wait_first(self, timeout=10.0):
        return self._first.wait(timeout=timeout)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)


def _ffmpeg_cmd(exe: str, url: str, w: int, h: int, transport: str, cam: dict = None) -> List[str]:
    cam = cam or {}
    probe = str(cam.get("rtsp_probe_size", "500K"))
    analyze = str(cam.get("rtsp_analyze_duration", "500K"))
    max_delay = str(cam.get("rtsp_max_delay_us", "0"))
    cmd = [
        exe, "-nostdin", "-hide_banner", "-loglevel", "warning",
        "-rtsp_transport", transport,
        "-probesize", probe,
        "-analyzeduration", analyze,
        "-fflags", "+nobuffer+flush_packets+discardcorrupt",
        "-flags", "low_delay",
        "-err_detect", "ignore_err",
        "-max_delay", max_delay,
        "-i", url,
        "-an",
    ]
    fps_mode = cam.get("rtsp_fps_mode", "passthrough")
    if fps_mode:
        cmd.extend(["-fps_mode", str(fps_mode)])
    scale = cam.get("rtsp_scale")
    if scale is False or str(scale).lower() in ("0", "false", "off", "none"):
        pass
    else:
        cmd.extend(["-vf", f"scale={w}:{h}"])
    cmd.extend([
        "-pix_fmt", "bgr24",
        "-f", "rawvideo",
        "pipe:1",
    ])
    if transport == "tcp":
        idx = cmd.index("-rtsp_transport")
        cmd[idx + 2:idx + 2] = ["-rtsp_flags", "prefer_tcp"]
    return cmd


_BENIGN_FFMPEG_STDERR = (
    "Missing PPS",
    "Missing SPS",
    "sprop-parameter-sets",
    "deprecated",
    "non-existing PPS",
    "no frame",
)


class FfmpegRtspStream:
    """RTSP через ffmpeg pipe (низкая задержка)."""

    def __init__(self, cam: dict):
        exe = _ffmpeg_exe(cam)
        if not exe:
            raise FileNotFoundError("ffmpeg не найден")
        self.url = build_rtsp_url(cam)
        self.w = int(cam.get("width", 1280))
        self.h = int(cam.get("height", 720))
        self.label = f"ffmpeg {_safe_label(self.url)}"
        self._size = self.w * self.h * 3
        buf_mode = str(cam.get("rtsp_buffer", "queue")).lower()
        if buf_mode == "latest":
            self._buf = _LatestFrame()
        else:
            qn = int(cam.get("rtsp_queue_frames", 2))
            self._buf = _FrameQueue(maxlen=qn)
        self._proc = None
        self._stderr_tail = ""
        self.fps = float(cam.get("fps") or 25)
        self._cam = cam
        self._first_frame_sec = float(cam.get("rtsp_first_frame_sec", 20))
        self._exe = exe
        self._active_transport = "tcp"
        self._last_restart_t = 0.0

        pref = str(cam.get("rtsp_transport", "tcp")).lower()
        if pref == "udp":
            transports = ["udp", "tcp"]
        else:
            transports = ["tcp", "udp"]
        last_err = ""
        for tr in transports:
            try:
                self._start_process(exe, tr)
                if self._buf.wait_first(self._first_frame_sec):
                    self.label += f" ({tr})"
                    self._active_transport = tr
                    return
                last_err = self._format_open_error(tr)
                self.close()
            except Exception as e:
                last_err = str(e)
                self.close()
        raise RuntimeError(f"ffmpeg: {last_err}")

    def _meaningful_stderr(self) -> str:
        raw = (self._stderr_tail or "").strip()
        if not raw:
            return ""
        parts = [
            p for p in raw.split(" | ")
            if p and not any(b in p for b in _BENIGN_FFMPEG_STDERR)
        ]
        return " | ".join(parts)

    def _format_open_error(self, transport: str) -> str:
        poll = self._proc.poll() if self._proc else None
        detail = self._meaningful_stderr()
        if poll is not None:
            base = f"exit {poll} ({transport})"
            return f"{base}: {detail}" if detail else base
        if detail:
            return f"{detail} ({transport})"
        return (
            f"нет кадра за {self._first_frame_sec:g}s ({transport}) "
            f"(попробуйте rtsp_transport=tcp в config.json)"
        )

    def _start_process(self, exe: str, transport: str):
        cmd = _ffmpeg_cmd(exe, self.url, self.w, self.h, transport, self._cam)
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            **_subprocess_kwargs(),
        )
        self._buf._stop.clear()
        if isinstance(self._buf, _FrameQueue):
            with self._buf._lock:
                self._buf._deque.clear()
            self._buf._first.clear()
        else:
            self._buf._frame = None
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        self._buf.start(self._reader)

    def _drain_stderr(self):
        err = self._proc.stderr
        if not err:
            return
        lines = []
        try:
            for line in err:
                lines.append(line.decode("utf-8", errors="replace").strip())
                if len(lines) > 8:
                    lines.pop(0)
        except Exception:
            pass
        self._stderr_tail = " | ".join(lines)[-400:]

    def _reader(self):
        out = self._proc.stdout
        chunk = bytearray()
        while not self._buf._stop.is_set():
            data = out.read(65536)
            if not data:
                break
            chunk.extend(data)
            while len(chunk) >= self._size:
                frame = np.frombuffer(bytes(chunk[: self._size]), dtype=np.uint8).copy()
                del chunk[: self._size]
                frame = frame.reshape((self.h, self.w, 3))
                self._buf.put(frame)
        self._buf._stop.set()

    def _try_restart(self) -> bool:
        now = time.time()
        if now - self._last_restart_t < 4.0:
            return False
        if self._proc is not None and self._proc.poll() is None:
            return False
        self._last_restart_t = now
        print(f"[camera] ffmpeg перезапуск ({self._active_transport})...")
        try:
            if self._proc is not None:
                self.close()
            self._start_process(self._exe, self._active_transport)
            return bool(self._buf.wait_first(10.0))
        except Exception as exc:
            print(f"[camera] ffmpeg restart: {exc}")
            return False

    def isOpened(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def read_latest(self) -> Tuple[bool, Optional[object]]:
        """Превью: самый свежий кадр, очередь сбрасывается."""
        if isinstance(self._buf, _LatestFrame):
            return self.read()
        ok, frame = self._buf.get_latest_copy()
        if ok and frame is not None:
            return ok, frame
        dead = self._proc is None or self._proc.poll() is not None
        reader_done = self._buf._stop.is_set()
        if dead or reader_done:
            if self._try_restart():
                return self._buf.get_latest_copy()
        return False, None

    def read(self) -> Tuple[bool, Optional[object]]:
        ok, frame = self._buf.get_copy()
        if ok and frame is not None:
            return ok, frame
        dead = self._proc is None or self._proc.poll() is not None
        reader_done = self._buf._stop.is_set()
        if dead or reader_done:
            if self._try_restart():
                return self._buf.get_copy()
        return False, None

    def read_batch(self, max_n: int = 1) -> List:
        """Последовательно до max_n кадров из очереди (для WAIT_L/R без пропуска стыка)."""
        max_n = max(1, int(max_n))
        out = []
        for _ in range(max_n):
            ok, frame = self.read()
            if not ok or frame is None:
                break
            out.append(frame)
            if not isinstance(self._buf, _FrameQueue):
                break
        return out

    def close(self):
        self._buf.stop()
        if self._proc is None:
            return
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None


class OpenCvStream:
    """Как раньше: cv2.VideoCapture."""

    def __init__(self, cam: dict):
        self._grab_drop = max(0, min(3, int(cam.get("rtsp_grab_drop", 1))))
        self._webcam_grayscale = bool(cam.get("webcam_grayscale", True))
        self._cap, self.label, self._kind = self._open(cam)
        cfg_fps = float(cam.get("fps") or 0)
        cap_fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0)
        if cfg_fps > 0:
            self.fps = cfg_fps
        elif cap_fps > 1:
            self.fps = cap_fps
        else:
            self.fps = 25.0

    def _open(self, cam: dict):
        if cam.get("index") is not None:
            cap = cv2.VideoCapture(int(cam["index"]), cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cam.get("width", 640)))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cam.get("height", 480)))
            return cap, f"opencv webcam:{cam['index']}", "webcam"

        if str(cam.get("source", "rtsp")).lower() == "mjpeg":
            url = build_mjpeg_url(cam)
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap, f"opencv {_safe_label(url)}", "mjpeg"

        url = build_rtsp_url(cam)
        transport = str(cam.get("rtsp_transport", "udp")).lower()
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            f"rtsp_transport;{transport}|"
            "fflags;nobuffer|"
            "flags;low_delay|"
            "max_delay;0"
        )
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap, f"opencv {_safe_label(url)}", "rtsp"

    def isOpened(self) -> bool:
        return self._cap.isOpened()

    def read(self) -> Tuple[bool, Optional[object]]:
        if self._kind == "rtsp" and self._grab_drop > 0:
            for _ in range(self._grab_drop):
                self._cap.grab()
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return ok, frame
        if self._kind == "webcam" and self._webcam_grayscale:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return ok, frame

    def close(self):
        self._cap.release()


class VideoFileStream:
    """Воспроизведение видеофайла (mp4/avi и т.д.) — тот же интерфейс, что у камеры."""

    is_video_file = True

    def __init__(self, path: str, *, loop: bool = True, realtime: bool = False, fps: float = 0):
        self.path = os.path.normpath(os.path.abspath(path))
        if not os.path.isfile(self.path):
            raise FileNotFoundError("Видео не найдено: {}".format(self.path))
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise RuntimeError("Не удалось открыть видео: {}".format(self.path))
        self.loop = bool(loop)
        self.realtime = bool(realtime)
        meta_fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0)
        if fps and float(fps) > 0:
            self.fps = float(fps)
        elif meta_fps > 1:
            self.fps = meta_fps
        else:
            self.fps = 25.0
        self.frame_count = max(0, int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.label = "video:{}".format(os.path.basename(self.path))
        self._ended = False
        self._last_read_t = None

    def isOpened(self) -> bool:
        return self._cap.isOpened() and not self._ended

    @property
    def position(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))

    def seek(self, frame_idx: int) -> bool:
        if self.frame_count > 0:
            frame_idx = max(0, min(int(frame_idx), self.frame_count - 1))
        else:
            frame_idx = max(0, int(frame_idx))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self._ended = False
        return True

    def rewind(self):
        self.seek(0)

    def read(self) -> Tuple[bool, Optional[object]]:
        if self.realtime and self._last_read_t is not None and self.fps > 0:
            want = 1.0 / self.fps
            dt = time.perf_counter() - self._last_read_t
            if dt < want:
                time.sleep(want - dt)
        ok, frame = self._cap.read()
        if ok and frame is not None:
            self._last_read_t = time.perf_counter()
            return True, frame
        if self.loop:
            self.seek(0)
            ok, frame = self._cap.read()
            if ok and frame is not None:
                self._last_read_t = time.perf_counter()
                return True, frame
        self._ended = not self.loop
        return False, None

    def close(self):
        self._cap.release()


def _want_video(cam: dict, run_mode=None) -> bool:
    """video — файл; camera/rtsp/webcam — живая камера."""
    rm = run_mode or {}
    media = str(rm.get("media", "")).strip().lower()
    if media in ("camera", "rtsp", "webcam", "live"):
        return False
    if media in ("video", "file"):
        return True
    if cam.get("use_video_file") is False:
        return False
    vf = cam.get("video_file") or cam.get("file")
    if not vf:
        return False
    vf = str(vf).strip()
    return bool(vf) and os.path.isfile(vf)


def open_video_file(path: str, *, loop: bool = True, realtime: bool = False, fps: float = 0) -> VideoFileStream:
    return VideoFileStream(path, loop=loop, realtime=realtime, fps=fps)


def open_media_source(cam: dict, run_mode=None):
    """
    Источник кадров по run_mode.media:
      camera / rtsp / webcam — RTSP или вебка (open_camera)
      video / file         — video_file из config
    Если media не задан — video_file при наличии файла, иначе камера.
    """
    c = dict(cam)
    cfg_fps = float(c.get("fps") or 0)
    if _want_video(c, run_mode):
        vf = str(c.get("video_file") or c.get("file")).strip()
        return open_video_file(
            vf,
            loop=bool(c.get("video_loop", True)),
            realtime=bool(c.get("video_realtime", False)),
            fps=cfg_fps,
        )
    return open_camera(c)


def open_camera(cam: dict):
    c = dict(cam)
    if c.get("use_webcam") or c.get("index") is not None:
        if c.get("index") is None:
            c["index"] = int(c.get("webcam_index", 0))
        return OpenCvStream(c)
    backend = str(c.get("rtsp_backend", "ffmpeg")).strip().lower()
    if backend in ("opencv", "cv2"):
        c = dict(c)
        c["source"] = "rtsp"
        return OpenCvStream(c)
    return FfmpegRtspStream(c)


def blur_rois_gray(frame, rects, blur_k: int):
    k = blur_k if blur_k % 2 else blur_k + 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return [cv2.GaussianBlur(gray[y:y + h, x:x + w], (k, k), 0) for x, y, w, h in rects]
