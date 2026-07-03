"""
Поиск двух лазерных полосок в search_roi.

Режимы (search_mode):
  grayscale — яркость: кадр как есть (2D) или gray из BGR, без цветовой обработки
  rgb       — цвет: red / red_enhanced / max_channel (rgb_method)
"""
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class LaserInfo:
    found: bool
    x0: float = 0.0
    x1: float = 0.0
    raw_x0: float = 0.0
    raw_x1: float = 0.0
    peak_l: float = 0.0
    peak_r: float = 0.0
    peaks_live: bool = False
    y: int = 0
    h: int = 0
    bottom_y: int = 0
    gap_px: float = 0.0
    debug: str = ""


def _legacy_search_mode(cfg: dict) -> str:
    if "search_mode" in cfg:
        m = str(cfg["search_mode"]).lower()
        return "grayscale" if m in ("gray", "grayscale", "bw", "mono") else "rgb"
    src = str(cfg.get("brightness_source", "rgb")).lower()
    if src in ("gray", "grayscale", "max_channel"):
        return "grayscale"
    return "rgb"


def _legacy_rgb_method(cfg: dict) -> str:
    if "rgb_method" in cfg:
        return str(cfg["rgb_method"]).lower()
    src = str(cfg.get("brightness_source", "red_enhanced")).lower()
    if src in ("red", "red_channel"):
        return "red"
    if src == "max_channel":
        return "max_channel"
    return "red_enhanced"


class LaserDetector:
    def __init__(self, cfg: dict, frame_w: int, frame_h: int):
        self.fw, self.fh = frame_w, frame_h
        self.search_mode = _legacy_search_mode(cfg)
        self.rgb_method = _legacy_rgb_method(cfg)
        self.bright_thresh = int(cfg.get("bright_thresh", 0))
        self.bright_percentile = float(cfg.get("bright_percentile", 98.0))
        self.min_col_fill = float(cfg.get("min_col_fill", 0.12))
        self.min_gap_ratio = float(cfg.get("min_gap_ratio", 0.12))
        self.max_gap_ratio = float(cfg.get("max_gap_ratio", 0.95))
        self.blur = int(cfg.get("blur", 3))
        self.morph_k = int(cfg.get("morph_kernel", 3))
        self.smooth_alpha = float(cfg.get("smooth_alpha", 0.25))
        self.lost_hold_frames = int(cfg.get("lost_hold_frames", 15))
        self.max_jump_px = int(cfg.get("max_jump_px", 40))
        self._seam_lost = 0
        self._seam_last: Optional[LaserInfo] = None
        self.bottom_band_ratio = float(cfg.get("bottom_band_ratio", 0.42))
        self.peak_prominence = float(cfg.get("peak_prominence", 0.35))

        self._init_search_roi(cfg, frame_w, frame_h)

        self._x0 = None
        self._x1 = None
        self._y = None
        self._h = None
        self._bottom = None
        self._lost = 0
        self._last_peak_l = None
        self._last_peak_r = None
        self.seam_track_window_px = int(cfg.get("seam_track_window_px", 120))
        self.last_mask = None
        self.last_proj = None
        self.last_thr = 0
        self.last_diag = ""
        self.last_col_max = 0.0

    def _init_search_roi(self, cfg: dict, frame_w: int, frame_h: int):
        sr = cfg.get("search_roi")
        if sr and int(sr.get("w", 0)) > 0 and int(sr.get("h", 0)) > 0:
            self.sx = int(sr.get("x", 0))
            self.sy = int(sr.get("y", 0))
            self.sw = int(sr.get("w", frame_w))
            self.sh = int(sr.get("h", frame_h))
        else:
            band_h = max(80, int(frame_h * self.bottom_band_ratio))
            self.sx, self.sw = 0, frame_w
            self.sy = frame_h - band_h
            self.sh = band_h
        self.clamp_search_roi(frame_w, frame_h)

    def mode_label(self) -> str:
        if self.search_mode == "grayscale":
            return "MODE=grayscale (brightness)"
        return "MODE=rgb ({})".format(self.rgb_method)

    def clamp_search_roi(self, fw: int, fh: int):
        self.sw = max(40, min(self.sw, fw))
        self.sh = max(40, min(self.sh, fh))
        self.sx = max(0, min(self.sx, fw - self.sw))
        self.sy = max(0, min(self.sy, fh - self.sh))

    def move_search_roi(self, dx: int, dy: int):
        self.sx += dx
        self.sy += dy
        self.clamp_search_roi(self.fw, self.fh)

    def resize_search_roi(self, dw: int, dh: int):
        self.sw = max(40, self.sw + dw)
        self.sh = max(40, self.sh + dh)
        self.clamp_search_roi(self.fw, self.fh)

    def reset_smooth(self):
        self._x0 = self._x1 = self._y = self._h = self._bottom = None
        self._lost = 0
        self._seam_lost = 0
        self._seam_last = None

    def cycle_search_mode(self) -> str:
        self.search_mode = "rgb" if self.search_mode == "grayscale" else "grayscale"
        self.reset_smooth()
        return self.search_mode

    def cycle_rgb_method(self) -> str:
        order = ("red_enhanced", "red", "max_channel")
        try:
            i = order.index(self.rgb_method)
        except ValueError:
            i = 0
        self.rgb_method = order[(i + 1) % len(order)]
        self.reset_smooth()
        return self.rgb_method

    def nudge_bright(self, delta: int):
        if self.bright_thresh <= 0:
            self.bright_thresh = max(20, self.last_thr if self.last_thr > 0 else 120)
        self.bright_thresh = int(np.clip(self.bright_thresh + delta, 5, 255))

    def nudge_percentile(self, delta: float):
        self.bright_thresh = 0
        self.bright_percentile = float(np.clip(self.bright_percentile + delta, 88.0, 99.9))

    def nudge_peak_prominence(self, delta: float):
        self.peak_prominence = float(np.clip(self.peak_prominence + delta, 0.12, 0.85))

    def nudge_min_col_fill(self, delta: float):
        self.min_col_fill = float(np.clip(self.min_col_fill + delta, 0.02, 0.45))

    def nudge_gap_ratio(self, which: str, delta: float):
        if which == "min":
            self.min_gap_ratio = float(np.clip(self.min_gap_ratio + delta, 0.03, 0.5))
        else:
            self.max_gap_ratio = float(np.clip(self.max_gap_ratio + delta, 0.5, 0.99))

    def nudge_blur(self, delta: int):
        self.blur = int(np.clip(self.blur + delta, 0, 15))

    def nudge_morph(self, delta: int):
        self.morph_k = int(np.clip(self.morph_k + delta, 1, 15))

    def threshold_label(self) -> str:
        thr_part = (
            "THR={} (manual)".format(self.bright_thresh)
            if self.bright_thresh > 0
            else "THR pct={:.1f}% -> {}".format(self.bright_percentile, self.last_thr)
        )
        return "{}  {}".format(self.mode_label(), thr_part)

    def _to_brightness(self, crop):
        """grayscale: кадр не перекрашиваем; rgb: канал/красный."""
        if self.search_mode == "grayscale":
            if crop.ndim == 2:
                return crop
            return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        if crop.ndim == 2:
            return crop
        method = self.rgb_method
        if method == "red":
            return crop[:, :, 2]
        if method == "max_channel":
            return np.max(crop, axis=2).astype(np.uint8)
        b, g, r = cv2.split(crop)
        return cv2.subtract(r, cv2.max(g, b))

    def _smooth_proj(self, proj: np.ndarray) -> np.ndarray:
        k = 5
        if len(proj) < k:
            return proj.astype(np.float32)
        kernel = np.ones(k, dtype=np.float32) / float(k)
        return np.convolve(proj.astype(np.float32), kernel, mode="same")

    def _list_peaks(self, sm: np.ndarray, peak_thr: float):
        peaks = []
        for i in range(1, len(sm) - 1):
            if sm[i] >= sm[i - 1] and sm[i] >= sm[i + 1] and sm[i] >= peak_thr:
                peaks.append((i, float(sm[i])))
        return peaks

    def _pick_peak_pair(self, peaks, roi_w: int) -> Optional[Tuple[int, int, str]]:
        min_gap = max(8, int(roi_w * self.min_gap_ratio))
        max_gap = int(roi_w * self.max_gap_ratio)
        best = None
        peaks.sort(key=lambda p: -p[1])
        for i in range(len(peaks)):
            for j in range(i + 1, len(peaks)):
                xa, sa = peaks[i]
                xb, sb = peaks[j]
                left, right = (xa, xb) if xa < xb else (xb, xa)
                gap = right - left
                if gap < min_gap or gap > max_gap:
                    continue
                score = sa + sb
                if best is None or score > best[0]:
                    best = (score, left, right)
        if best is None:
            return None
        return best[1], best[2], "peaks={} gap={}px".format(len(peaks), best[2] - best[1])

    def _fallback_two_columns(self, sm: np.ndarray, roi_w: int) -> Optional[Tuple[int, int, str]]:
        min_gap = max(8, int(roi_w * self.min_gap_ratio))
        order = np.argsort(sm)[::-1]
        for i in order[: min(12, len(order))]:
            for j in order[: min(12, len(order))]:
                if i == j:
                    continue
                left, right = (int(i), int(j)) if i < j else (int(j), int(i))
                if right - left < min_gap:
                    continue
                if sm[left] < self.min_col_fill * 0.5 or sm[right] < self.min_col_fill * 0.5:
                    continue
                return left, right, "fallback cols gap={}px".format(right - left)
        return None

    def _peak_in_window(self, sm: np.ndarray, ox: float, hint_x: float) -> Optional[float]:
        """Максимум проекции в окне вокруг ожидаемой позиции пика (для стыка)."""
        if hint_x is None or len(sm) < 2:
            return None
        cx = int(round(hint_x - ox))
        half = max(15, self.seam_track_window_px // 2)
        x0 = max(0, cx - half)
        x1 = min(len(sm), cx + half + 1)
        if x1 - x0 < 3:
            return None
        idx = x0 + int(np.argmax(sm[x0:x1]))
        return float(ox + idx)

    def _peaks_from_halves(self, sm: np.ndarray, ox: float, w_m: int) -> Tuple[float, float]:
        mid = max(1, w_m // 2)
        li = int(np.argmax(sm[:mid]))
        ri = int(np.argmax(sm[mid:])) + mid
        return float(ox + li), float(ox + ri)

    def _prepare_projection(self, frame_bgr, fw: int, fh: int):
        """Яркость → col_fill. Возвращает (col_fill, mask, ox, oy, ox2, oy2) или None."""
        self.clamp_search_roi(fw, fh)
        ox, oy = self.sx, self.sy
        ox2 = min(fw, self.sx + self.sw)
        oy2 = min(fh, self.sy + self.sh)
        crop = frame_bgr[oy:oy2, ox:ox2]
        if crop.size == 0:
            return None

        bright = self._to_brightness(crop)
        if self.blur > 0:
            k = self.blur | 1
            bright = cv2.GaussianBlur(bright, (k, k), 0)

        if self.bright_thresh > 0:
            thr = self.bright_thresh
        else:
            thr = int(np.percentile(bright, self.bright_percentile))
        self.last_thr = int(thr)
        _, mask = cv2.threshold(bright, thr, 255, cv2.THRESH_BINARY)

        if self.morph_k > 1:
            k = self.morph_k | 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        h_m, w_m = mask.shape[:2]
        col_fill = mask.sum(axis=0).astype(np.float32) / max(h_m * 255.0, 1.0)
        self.last_col_max = float(col_fill.max())
        full_mask = np.zeros((fh, fw), dtype=np.uint8)
        full_mask[oy:oy2, ox:ox2] = mask
        self.last_mask = full_mask
        return col_fill, mask, ox, oy, ox2, oy2, w_m

    def read_seam_peaks(
        self,
        frame_bgr,
        fw: int,
        fh: int,
        *,
        hint_l: Optional[float] = None,
        hint_r: Optional[float] = None,
        active: str = "both",
    ) -> LaserInfo:
        """
        Пики гистограммы каждый кадр (без hold / без проверки gap).
        active: 'left' | 'right' | 'both' — какой пик обновлять (окно вокруг hint).
        """
        prep = self._prepare_projection(frame_bgr, fw, fh)
        if prep is None:
            return self._hold_or_lost_seam("empty")
        col_fill, mask, ox, oy, ox2, oy2, w_m = prep

        min_sig = self.min_col_fill * 0.2
        if col_fill.max() < min_sig:
            return self._hold_or_lost_seam("dim proj")

        sm = self._smooth_proj(col_fill)
        self.last_proj = sm
        half_l, half_r = self._peaks_from_halves(sm, ox, w_m)

        pl = self._peak_in_window(sm, ox, hint_l) if hint_l is not None else half_l
        pr = self._peak_in_window(sm, ox, hint_r) if hint_r is not None else half_r
        if pl is None:
            pl = half_l
        if pr is None:
            pr = half_r

        if active == "left" and self._last_peak_r is not None:
            pr = float(self._last_peak_r)
        elif active == "right" and self._last_peak_l is not None:
            pl = float(self._last_peak_l)

        self._last_peak_l = pl
        self._last_peak_r = pr

        bx0 = int(round(pl - ox))
        bx1 = int(round(pr - ox))
        top, bottom = self._combined_stripe_extent(
            mask,
            int(np.clip(bx0, 0, w_m - 1)),
            int(np.clip(bx1, 0, w_m - 1)),
        )
        ry = oy + top
        rh = max(11, bottom - top + 1)
        bot = oy + bottom

        info = LaserInfo(
            found=True,
            x0=pl,
            x1=pr,
            raw_x0=pl,
            raw_x1=pr,
            peak_l=pl,
            peak_r=pr,
            peaks_live=True,
            y=ry,
            h=rh,
            bottom_y=int(bot),
            gap_px=pr - pl,
            debug="seam proj L={:.0f} R={:.0f}".format(pl, pr),
        )
        self._seam_lost = 0
        self._seam_last = info
        return info

    def _hold_or_lost_seam(self, reason: str) -> LaserInfo:
        self._seam_lost += 1
        if self._seam_last is not None and self._seam_lost <= self.lost_hold_frames:
            li = self._seam_last
            return LaserInfo(
                found=True,
                x0=li.x0,
                x1=li.x1,
                raw_x0=li.raw_x0,
                raw_x1=li.raw_x1,
                peak_l=li.peak_l,
                peak_r=li.peak_r,
                peaks_live=False,
                y=li.y,
                h=li.h,
                bottom_y=li.bottom_y,
                gap_px=li.gap_px,
                debug="seam hold({}/{}) {}".format(
                    self._seam_lost, self.lost_hold_frames, reason
                ),
            )
        return LaserInfo(False, debug=reason)

    def _find_two_peaks(self, proj: np.ndarray, roi_w: int) -> Optional[Tuple[int, int]]:
        if roi_w < 20:
            self.last_diag = "roi too narrow"
            return None
        sm = self._smooth_proj(proj)
        self.last_proj = sm
        peak_thr = float(sm.max()) * self.peak_prominence
        if peak_thr < 0.01:
            self.last_diag = "signal=0"
            return None

        peaks = self._list_peaks(sm, peak_thr)
        picked = self._pick_peak_pair(peaks, roi_w)
        if picked is not None:
            self.last_diag = picked[2]
            return picked[0], picked[1]

        if len(peaks) == 1:
            self.last_diag = "1 peak only prom={:.2f}".format(self.peak_prominence)
        elif len(peaks) == 0:
            self.last_diag = "0 peaks prom={:.2f} max={:.2f}".format(
                self.peak_prominence, sm.max()
            )
        else:
            min_gap = max(8, int(roi_w * self.min_gap_ratio))
            max_gap = int(roi_w * self.max_gap_ratio)
            self.last_diag = "{} peaks bad gap need {}-{}px".format(
                len(peaks), min_gap, max_gap
            )

        fb = self._fallback_two_columns(sm, roi_w)
        if fb is not None:
            self.last_diag = fb[2]
            return fb[0], fb[1]
        return None

    def _stripe_extent_at(self, mask: np.ndarray, bx: int) -> Tuple[int, int]:
        h_m, w_m = mask.shape[:2]
        xc = int(np.clip(bx, 0, w_m - 1))
        x_left = max(0, xc - 1)
        x_right = min(w_m - 1, xc + 1)
        col = mask[:, x_left:x_right + 1]
        rows = np.where(col.max(axis=1) > 127)[0]
        if rows.size == 0:
            return 0, h_m - 1
        return int(rows[0]), int(rows[-1])

    def _combined_stripe_extent(self, mask: np.ndarray, bx0: int, bx1: int) -> Tuple[int, int]:
        t0, b0 = self._stripe_extent_at(mask, bx0)
        t1, b1 = self._stripe_extent_at(mask, bx1)
        top = min(t0, t1)
        bottom = max(b0, b1)
        if bottom - top < 11:
            bottom = min(mask.shape[0] - 1, top + 11)
        return top, bottom

    def _smooth_pos(self, x0, x1, y, h, bottom_y):
        if self._x0 is not None:
            if abs(x0 - self._x0) > self.max_jump_px or abs(x1 - self._x1) > self.max_jump_px:
                a = min(self.smooth_alpha, 0.1)
            else:
                a = self.smooth_alpha
        else:
            a = 1.0
        if self._x0 is None:
            self._x0, self._x1 = x0, x1
            self._y, self._h, self._bottom = y, h, bottom_y
        else:
            self._x0 = self._x0 * (1 - a) + x0 * a
            self._x1 = self._x1 * (1 - a) + x1 * a
            self._y = int(round(self._y * (1 - a) + y * a))
            self._h = int(round(self._h * (1 - a) + h * a))
            self._bottom = int(round(self._bottom * (1 - a) + bottom_y * a))
        return self._x0, self._x1, self._y, self._h, self._bottom

    def detect(self, frame_bgr, fw: int, fh: int) -> LaserInfo:
        prep = self._prepare_projection(frame_bgr, fw, fh)
        if prep is None:
            self.last_mask = None
            return self._hold_or_lost("empty")
        col_fill, mask, ox, oy, ox2, oy2, w_m = prep

        if col_fill.max() < self.min_col_fill:
            self.last_diag = "dim fill={:.2f} need>={:.2f}".format(
                col_fill.max(), self.min_col_fill
            )
            return self._hold_or_lost("dim")

        peaks = self._find_two_peaks(col_fill, w_m)
        if peaks is None:
            return self._hold_or_lost(self.last_diag or "no 2 peaks")

        bx0, bx1 = peaks
        top, bottom = self._combined_stripe_extent(mask, bx0, bx1)
        rx0 = float(ox + bx0)
        rx1 = float(ox + bx1)
        self._last_peak_l = rx0
        self._last_peak_r = rx1
        ry = oy + top
        rh = bottom - top + 1
        frame_bottom = oy + bottom
        sx0, sx1, sy, sh, sbot = self._smooth_pos(rx0, rx1, ry, rh, frame_bottom)
        self._lost = 0
        return LaserInfo(
            found=True,
            x0=sx0,
            x1=sx1,
            raw_x0=rx0,
            raw_x1=rx1,
            peak_l=rx0,
            peak_r=rx1,
            peaks_live=True,
            y=sy,
            h=sh,
            bottom_y=int(sbot),
            gap_px=sx1 - sx0,
            debug="peaks L={:.0f} R={:.0f} gap={} {}".format(
                rx0, rx1, int(sx1 - sx0), self.search_mode
            ),
        )

    def _hold_or_lost(self, reason: str) -> LaserInfo:
        self._lost += 1
        if self._x0 is not None and self._lost <= self.lost_hold_frames:
            pl = self._last_peak_l if self._last_peak_l is not None else float(self._x0)
            pr = self._last_peak_r if self._last_peak_r is not None else float(self._x1)
            return LaserInfo(
                found=True,
                x0=self._x0,
                x1=self._x1,
                raw_x0=pl,
                raw_x1=pr,
                peak_l=pl,
                peak_r=pr,
                peaks_live=False,
                y=self._y,
                h=self._h,
                bottom_y=int(self._bottom) if self._bottom is not None else int(self._y + self._h - 1),
                gap_px=self._x1 - self._x0,
                debug="hold({}) {}".format(self._lost, reason),
            )
        return LaserInfo(False, debug=reason)

    def to_config(self, r_cfg: dict) -> dict:
        out = dict(r_cfg)
        out.update({
            "search_mode": self.search_mode,
            "rgb_method": self.rgb_method,
            "bright_thresh": self.bright_thresh,
            "bright_percentile": self.bright_percentile,
            "min_col_fill": self.min_col_fill,
            "min_gap_ratio": self.min_gap_ratio,
            "max_gap_ratio": self.max_gap_ratio,
            "blur": self.blur,
            "morph_kernel": self.morph_k,
            "smooth_alpha": self.smooth_alpha,
            "lost_hold_frames": self.lost_hold_frames,
            "max_jump_px": self.max_jump_px,
            "bottom_band_ratio": self.bottom_band_ratio,
            "peak_prominence": self.peak_prominence,
            "seam_track_window_px": self.seam_track_window_px,
            "search_roi": {
                "x": self.sx, "y": self.sy, "w": self.sw, "h": self.sh,
            },
        })
        return out


def rois_from_lasers(
    laser: LaserInfo,
    roi_w: int,
    roi_h: int,
    seam_off_l_x: int,
    seam_off_r_x: int,
    seam_off_y: int,
    fw: int,
    fh: int,
):
    cx_l = int(round(laser.x0)) + int(seam_off_l_x)
    cx_r = int(round(laser.x1)) + int(seam_off_r_x)
    stripe_bottom = int(laser.bottom_y if laser.bottom_y > 0 else laser.y + laser.h - 1)
    stripe_bottom += int(seam_off_y)
    rh = int(roi_h)
    ry = stripe_bottom - rh + 1
    ry = max(0, min(ry, fh - rh))
    lx = max(0, min(cx_l - roi_w // 2, fw - roi_w))
    rx = max(0, min(cx_r - roi_w // 2, fw - roi_w))
    return (
        {"x": lx, "y": ry, "w": roi_w, "h": rh},
        {"x": rx, "y": ry, "w": roi_w, "h": rh},
    )
