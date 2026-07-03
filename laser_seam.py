"""
Стык: изменение профиля гистограммы (col_fill) под лазером.
auto_tune — подстройка порогов по шуму и успешным стыкам в процессе работы.
"""
from collections import deque
from typing import Deque, Dict, Optional, Tuple

import numpy as np

from vision_utils import SeamResult


class _SeamAutoTune:
    """Онлайн: шум в WATCH + обучение на реальных DROP."""

    def __init__(self, cfg: dict, defaults: dict):
        c = cfg.get("auto_tune", {}) or {}
        self.enabled = bool(c.get("enabled", True))
        self.noise_margin = float(c.get("noise_margin", 2.8))
        self.rise_margin = float(c.get("rise_margin", 2.0))
        self.hit_ratio = float(c.get("hit_ratio", 0.32))
        self.learn_cap_ratio = float(c.get("learn_cap_ratio", 0.42))
        self.chg_abs_cap = float(c.get("chg_abs_cap", 0.048))
        self.learn_alpha = float(c.get("learn_alpha", 0.22))
        self.buf_size = int(c.get("noise_buffer", 90))
        self.min_samples = int(c.get("min_noise_samples", 10))
        self.chg_floor_ratio = float(c.get("chg_floor_ratio", 1.20))
        self.noise_max_drop = float(c.get("noise_max_drop", 0.03))
        self.learn_min_drop_ratio = float(c.get("learn_min_drop_ratio", 0.15))
        self.inflate_chg_from_learned = bool(c.get("inflate_chg_from_learned", False))
        self._defaults = defaults
        self._noise_l: Deque[float] = deque(maxlen=self.buf_size)
        self._noise_r: Deque[float] = deque(maxlen=self.buf_size)
        self._spike_l: Deque[float] = deque(maxlen=self.buf_size)
        self._spike_r: Deque[float] = deque(maxlen=self.buf_size)
        self._learned_chg: Optional[float] = c.get("learned_hit_chg")
        self._learned_spike: Optional[float] = c.get("learned_hit_spike")
        self._learned_drop: Optional[float] = c.get("learned_hit_drop")
        self._cycles = int(c.get("cycles", 0))
        self.last_noise_l = 0.0
        self.last_noise_r = 0.0
        self.last_chg_thr = defaults["profile_change_min"]
        self.last_rise_thr = defaults["rise_min"]
        self.last_spike_thr = defaults["change_spike_min"]

    def reset_cycle(self):
        self._noise_l.clear()
        self._noise_r.clear()
        self._spike_l.clear()
        self._spike_r.clear()

    def note_quiet(self, side: str, chg: float, spike: float):
        if not self.enabled:
            return
        buf = self._noise_l if side == "left" else self._noise_r
        sp = self._spike_l if side == "left" else self._spike_r
        buf.append(float(chg))
        sp.append(float(spike))

    def note_cycle_complete(self, left_m: dict, right_m: dict):
        """Обучение только после полного цикла L→R с достаточным сигналом."""
        if not self.enabled:
            return
        ld = float(left_m["drop_ratio"])
        rd = float(right_m["drop_ratio"])
        if ld < self.learn_min_drop_ratio or rd < self.learn_min_drop_ratio:
            return
        ls = float(left_m["change_spike"])
        rs = float(right_m["change_spike"])
        min_sp = self._defaults["change_spike_min"] * 0.75
        if ls < min_sp or rs < min_sp:
            return
        drop = max(ld, rd)
        spike = max(ls, rs)
        chg = 0.5 * (float(left_m["prof_change"]) + float(right_m["prof_change"]))
        a = self.learn_alpha
        if self._learned_chg is None:
            self._learned_chg = chg
            self._learned_spike = spike
            self._learned_drop = drop
        else:
            self._learned_chg = (1.0 - a) * self._learned_chg + a * chg
            self._learned_spike = (1.0 - a) * self._learned_spike + a * spike
            self._learned_drop = (1.0 - a) * self._learned_drop + a * drop
        self._cycles += 1

    def note_hit(self, m: dict):
        """Устарело — не вызывать по одному DROP."""
        self.note_cycle_complete(m, m)

    def _p90(self, buf: Deque[float]) -> float:
        if len(buf) < self.min_samples:
            return 0.0
        return float(np.percentile(np.array(buf, dtype=np.float32), 90))

    def _p90_values(self, values) -> float:
        if len(values) < self.min_samples:
            return 0.0
        return float(np.percentile(np.array(values, dtype=np.float32), 90))

    def _compute_thresholds(
        self, noise: float, noise_sp: float, apply_learned: bool,
    ) -> Tuple[float, float, float]:
        d = self._defaults
        chg_thr = d["profile_change_min"]
        chg_floor = d["profile_change_min"] * self.chg_floor_ratio
        if noise > 0:
            chg_thr = max(chg_thr, noise * self.noise_margin)
        chg_thr = max(chg_thr, chg_floor)
        rise_thr = d["rise_min"]
        if noise > 0:
            rise_thr = max(rise_thr, noise * self.rise_margin)
        spike_thr = d["change_spike_min"]
        if noise_sp > 0:
            spike_thr = max(spike_thr, noise_sp * 2.5)
        if (
            apply_learned
            and self.inflate_chg_from_learned
            and self._learned_chg is not None
            and self._learned_chg > 0
        ):
            from_hit = self._learned_chg * self.hit_ratio
            chg_thr = max(chg_thr, from_hit)
            cap = min(
                self._learned_chg * self.learn_cap_ratio,
                self.chg_abs_cap,
            )
            chg_thr = min(chg_thr, cap)
        chg_thr = float(np.clip(chg_thr, 0.010, self.chg_abs_cap))
        rise_thr = float(np.clip(rise_thr, 0.005, 0.05))
        spike_thr = float(np.clip(spike_thr, 0.004, 0.06))
        return chg_thr, rise_thr, spike_thr

    def unified_thresholds(self, apply_learned: bool = True) -> Tuple[float, float, float]:
        """Одинаковые пороги L/R: шум с обеих сторон."""
        d = self._defaults
        if not self.enabled:
            self.last_chg_thr = d["profile_change_min"]
            self.last_rise_thr = d["rise_min"]
            self.last_spike_thr = d["change_spike_min"]
            return self.last_chg_thr, self.last_rise_thr, self.last_spike_thr
        noise_vals = list(self._noise_l) + list(self._noise_r)
        spike_vals = list(self._spike_l) + list(self._spike_r)
        noise = self._p90_values(noise_vals)
        noise_sp = self._p90_values(spike_vals)
        self.last_noise_l = noise
        self.last_noise_r = noise
        chg_thr, rise_thr, spike_thr = self._compute_thresholds(
            noise, noise_sp, apply_learned,
        )
        self.last_chg_thr = chg_thr
        self.last_rise_thr = rise_thr
        self.last_spike_thr = spike_thr
        return chg_thr, rise_thr, spike_thr

    def thresholds(self, side: str, apply_learned: bool = True) -> Tuple[float, float, float]:
        return self.unified_thresholds(apply_learned)

    def to_dict(self) -> dict:
        out = {
            "enabled": self.enabled,
            "noise_margin": self.noise_margin,
            "rise_margin": self.rise_margin,
            "hit_ratio": self.hit_ratio,
            "learn_cap_ratio": self.learn_cap_ratio,
            "chg_abs_cap": self.chg_abs_cap,
            "learn_alpha": self.learn_alpha,
            "noise_buffer": self.buf_size,
            "min_noise_samples": self.min_samples,
            "learn_min_drop_ratio": self.learn_min_drop_ratio,
            "inflate_chg_from_learned": self.inflate_chg_from_learned,
            "cycles": self._cycles,
        }
        if self._learned_chg is not None:
            out["learned_hit_chg"] = round(self._learned_chg, 4)
            out["learned_hit_spike"] = round(self._learned_spike, 4)
            out["learned_hit_drop"] = round(self._learned_drop, 4)
        return out


class LaserSeamTracker:
    def __init__(self, cfg: dict):
        c = cfg or {}
        self.profile_half_width_px = int(c.get("profile_half_width_px", 40))
        self.peak_drop_ratio = float(c.get("peak_drop_ratio", 0.08))
        self.profile_change_min = float(c.get("profile_change_min", 0.020))
        self.change_spike_min = float(c.get("change_spike_min", 0.010))
        self.rise_min = float(c.get("rise_min", 0.010))
        self.frame_drop_ratio = float(c.get("frame_drop_ratio", 0.06))
        self.warmup_frames = int(c.get("warmup_frames", 15))
        self.settle_frames = int(c.get("settle_frames", 10))
        self.confirm_frames = int(c.get("jump_confirm_frames", c.get("confirm_frames", 3)))
        self.strong_chg_ratio = float(c.get("strong_chg_ratio", 1.22))
        self.event_rise_min_ratio = float(c.get("event_rise_min_ratio", 1.0))
        self.event_spike_ratio = float(c.get("event_spike_ratio", 0.85))
        self.event_spike_abs_min = float(c.get("event_spike_abs_min", 0.013))
        self.strong_drop_ratio = float(c.get("strong_drop_ratio", 0.15))
        self.drop_spike_abs_min = float(c.get("drop_spike_abs_min", 0.012))
        self.drop_spike_ratio = float(c.get("drop_spike_ratio", 0.55))
        self.drop_spike_free_ratio = float(c.get("drop_spike_free_ratio", 0.18))
        self.fast_confirm_drop_ratio = float(
            c.get("fast_confirm_drop_ratio", self.strong_drop_ratio)
        )
        self.fast_confirm_spike_ratio = float(c.get("fast_confirm_spike_ratio", 2.5))
        self.fast_confirm_frames = int(c.get("fast_confirm_frames", 1))
        # Шов трубы (горизонтальный): spike+chg без drop — не стык для замера
        self.spike_only_min_drop = float(c.get("spike_only_min_drop", 0.04))
        self.spike_only_mult = float(c.get("spike_only_mult", 4.2))
        self.gradual_seam_chg_mult = float(c.get("gradual_seam_chg_mult", 1.35))
        self.gradual_seam_chg_high_mult = float(c.get("gradual_seam_chg_high_mult", 2.0))
        self.gradual_seam_drop_relax = float(c.get("gradual_seam_drop_relax", 0.75))
        self._cfg_base = {
            "profile_change_min": self.profile_change_min,
            "rise_min": self.rise_min,
            "change_spike_min": self.change_spike_min,
            "strong_chg_ratio": self.strong_chg_ratio,
        }
        self._auto = _SeamAutoTune(c, self._cfg_base)
        self._thr = (self.profile_change_min, self.rise_min, self.change_spike_min)
        self._thr_l = self._thr
        self._thr_r = self._thr
        self._baseline_l: Optional[float] = None
        self._baseline_r: Optional[float] = None
        self._prof_l: Optional[np.ndarray] = None
        self._prof_r: Optional[np.ndarray] = None
        self._win_l: Optional[Tuple[int, int]] = None
        self._win_r: Optional[Tuple[int, int]] = None
        self._peak_base_l = 0.0
        self._peak_base_r = 0.0
        self._peak_prev_l = 0.0
        self._peak_prev_r = 0.0
        self._chg_prev_l = 0.0
        self._chg_prev_r = 0.0
        self._frames_l = 0
        self._frames_r = 0
        self._locked_l = False
        self._locked_r = False
        self._confirm_l = 0
        self._confirm_r = 0
        self._fired_l = False
        self._fired_r = False
        self._ewma_chg_l = 0.0
        self._ewma_chg_r = 0.0
        self._pending_left_hit: Optional[dict] = None
        self.last_drop_l = 0.0
        self.last_drop_r = 0.0
        self.last_change_l = 0.0
        self.last_change_r = 0.0
        self.last_spike_l = 0.0
        self.last_spike_r = 0.0

    @property
    def auto_tune_enabled(self) -> bool:
        return self._auto.enabled

    def effective_chg_min(self, side: str = "left") -> float:
        return self._thr[0]

    def _sync_thresholds(self, apply_learned: bool = True):
        self._thr = self._auto.unified_thresholds(apply_learned)
        self._thr_l = self._thr
        self._thr_r = self._thr

    def reset(self):
        self._baseline_l = self._baseline_r = None
        self._prof_l = self._prof_r = None
        self._win_l = self._win_r = None
        self._peak_base_l = self._peak_base_r = 0.0
        self._peak_prev_l = self._peak_prev_r = 0.0
        self._chg_prev_l = self._chg_prev_r = 0.0
        self._frames_l = self._frames_r = 0
        self._locked_l = self._locked_r = False
        self._confirm_l = self._confirm_r = 0
        self._fired_l = self._fired_r = False
        self._ewma_chg_l = self._ewma_chg_r = 0.0
        self._pending_left_hit = None
        self.last_drop_l = self.last_drop_r = 0.0
        self.last_change_l = self.last_change_r = 0.0
        self.last_spike_l = self.last_spike_r = 0.0

    def to_config(self, cfg: dict) -> dict:
        out = dict(cfg or {})
        out.update({
            "profile_half_width_px": self.profile_half_width_px,
            "peak_drop_ratio": self.peak_drop_ratio,
            "profile_change_min": self.profile_change_min,
            "change_spike_min": self.change_spike_min,
            "rise_min": self.rise_min,
            "frame_drop_ratio": self.frame_drop_ratio,
            "warmup_frames": self.warmup_frames,
            "settle_frames": self.settle_frames,
            "confirm_frames": self.confirm_frames,
            "auto_tune": self._auto.to_dict(),
        })
        for old in (
            "jump_threshold_px", "jump_confirm_frames", "jump_mode",
            "use_raw_position", "peak_drop_abs", "spike_chg_ratio",
            "soft_change_ratio", "soft_confirm_extra",
        ):
            out.pop(old, None)
        return out

    def nudge_drop_ratio(self, delta: float):
        self.peak_drop_ratio = float(np.clip(self.peak_drop_ratio + delta, 0.02, 0.5))

    def nudge_profile_change(self, delta: float):
        self.profile_change_min = float(np.clip(self.profile_change_min + delta, 0.008, 0.25))
        self._cfg_base["profile_change_min"] = self.profile_change_min

    def nudge_change_spike(self, delta: float):
        self.change_spike_min = float(np.clip(self.change_spike_min + delta, 0.004, 0.08))
        self._cfg_base["change_spike_min"] = self.change_spike_min

    def nudge_frame_drop(self, delta: float):
        self.frame_drop_ratio = float(np.clip(self.frame_drop_ratio + delta, 0.02, 0.35))

    def nudge_confirm(self, delta: int):
        self.confirm_frames = max(1, min(10, self.confirm_frames + delta))

    def nudge_warmup(self, delta: int):
        self.warmup_frames = max(0, min(60, self.warmup_frames + delta))

    @property
    def baseline_l(self) -> Optional[float]:
        return self._baseline_l

    @property
    def baseline_r(self) -> Optional[float]:
        return self._baseline_r

    @property
    def watch_ready_l(self) -> bool:
        return self._locked_l

    @property
    def watch_ready_r(self) -> bool:
        return self._locked_r

    @staticmethod
    def _window(sm: np.ndarray, ox: float, center_x: float, half_w: int):
        cx = int(round(center_x - ox))
        x0 = max(0, cx - half_w)
        x1 = min(len(sm), cx + half_w + 1)
        if x1 - x0 < 3:
            return None, None
        return (x0, x1), sm[x0:x1].astype(np.float32)

    def _profile_in_window(self, sm: np.ndarray, win: Tuple[int, int]) -> Optional[np.ndarray]:
        x0, x1 = win
        if x1 > len(sm):
            return None
        return sm[x0:x1].astype(np.float32)

    def _begin_cycle(self):
        self._auto.reset_cycle()
        self._pending_left_hit = None
        self._sync_thresholds(apply_learned=True)

    def complete_measure(self, right_m: dict):
        if self._pending_left_hit is not None:
            self._auto.note_cycle_complete(self._pending_left_hit, right_m)
        self._pending_left_hit = None

    def _capture_side(self, sm: np.ndarray, ox: float, center_x: float, side: str) -> bool:
        hw = self.profile_half_width_px
        win, prof = self._window(sm, ox, center_x, hw)
        if prof is None:
            return False
        if side == "left":
            self._win_l = win
            self._prof_l = prof.copy()
            self._peak_base_l = float(prof.max())
            self._peak_prev_l = self._peak_base_l
            self._chg_prev_l = 0.0
            self._confirm_l = 0
            self._frames_l = 0
            self._locked_l = False
            self._ewma_chg_l = 0.0
        else:
            self._win_r = win
            self._prof_r = prof.copy()
            self._peak_base_r = float(prof.max())
            self._peak_prev_r = self._peak_base_r
            self._chg_prev_r = 0.0
            self._confirm_r = 0
            self._frames_r = 0
            self._locked_r = False
            self._fired_r = False
            self._ewma_chg_r = 0.0
        return True

    def _lock_watch_baseline(self, side: str, det) -> bool:
        sm = getattr(det, "last_proj", None)
        if sm is None:
            return False
        win = self._win_l if side == "left" else self._win_r
        if win is None:
            return False
        prof = self._profile_in_window(sm, win)
        if prof is None:
            return False
        peak = float(prof.max())
        if side == "left":
            self._prof_l = prof.copy()
            self._peak_base_l = peak
            self._peak_prev_l = peak
            self._chg_prev_l = 0.0
            self._confirm_l = 0
            self._locked_l = True
            self._ewma_chg_l = 0.0
        else:
            self._prof_r = prof.copy()
            self._peak_base_r = peak
            self._peak_prev_r = peak
            self._chg_prev_r = 0.0
            self._confirm_r = 0
            self._locked_r = True
            self._ewma_chg_r = 0.0
        self._sync_thresholds(apply_learned=True)
        return True

    def _update_ewma(self, side: str, chg: float) -> float:
        alpha = 0.07
        if side == "left":
            self._ewma_chg_l = alpha * chg + (1.0 - alpha) * self._ewma_chg_l
            return self._ewma_chg_l
        self._ewma_chg_r = alpha * chg + (1.0 - alpha) * self._ewma_chg_r
        return self._ewma_chg_r

    def _side_thresholds(self, side: str) -> Tuple[float, float, float]:
        return self._thr

    def _gradual_seam_drop_ok(self, m: dict, chg_thr: float) -> bool:
        """
        Медленный проход стыка: chg высокий, spike≈0 (нарастание плавное, кадр-к-кадру).
        Шов трубы: drop≈0 при высоком chg — не проходит.
        """
        drop = m["drop_ratio"]
        chg = m["prof_change"]
        min_drop = self.peak_drop_ratio * self.gradual_seam_drop_relax
        if drop < min_drop or chg < chg_thr * self.gradual_seam_chg_mult:
            return False
        if drop >= self.peak_drop_ratio:
            return True
        return chg >= chg_thr * self.gradual_seam_chg_high_mult

    def _drop_spike_ok(self, m: dict, spike_thr: float, chg_thr: float = 0.0) -> bool:
        """Падение пика без всплеска профиля — шов/шум, не стык замера."""
        if chg_thr <= 0:
            chg_thr = self._thr[0]
        if self._gradual_seam_drop_ok(m, chg_thr):
            return True
        drop = m["drop_ratio"]
        spike = m["change_spike"]
        min_spike = self._min_event_spike(spike_thr)
        need = max(self.drop_spike_abs_min, min_spike * self.drop_spike_ratio)
        if (
            self._auto._learned_spike is not None
            and self._auto._cycles > 0
        ):
            need = max(need, float(self._auto._learned_spike) * 0.55)
        if drop >= self.drop_spike_free_ratio:
            # Большой drop — порог spike ниже, но не ноль (шов: chg↑ drop~0 spike~0)
            if spike >= self.drop_spike_abs_min:
                return True
            if drop >= self.drop_spike_free_ratio + 0.12:
                return True
            return False
        if spike >= need:
            return True
        if (
            drop >= self.peak_drop_ratio
            and m["frame_drop_ratio"] >= self.frame_drop_ratio * 0.65
            and spike >= need * 0.55
        ):
            return True
        return False

    def _min_event_spike(self, spike_thr: float) -> float:
        return max(spike_thr * self.event_spike_ratio, self.event_spike_abs_min)

    def _has_dynamic(self, m: dict, spike_thr: float) -> bool:
        if m["frame_drop_ratio"] >= self.frame_drop_ratio * 0.50:
            return True
        return self._spike_is_seam(m, spike_thr)

    def _spike_is_seam(self, m: dict, spike_thr: float) -> bool:
        """Всплеск профиля: при малом drop — только очень сильный (не шов трубы)."""
        min_spike = self._min_event_spike(spike_thr)
        spike = m["change_spike"]
        if spike < min_spike:
            return False
        drop = m["drop_ratio"]
        if drop >= self.spike_only_min_drop:
            return True
        return spike >= min_spike * self.spike_only_mult

    def _drop_is_seam(self, m: dict, spike_thr: float) -> bool:
        chg_thr = self._thr[0]
        drop = m["drop_ratio"]
        gradual = self._gradual_seam_drop_ok(m, chg_thr)
        if gradual:
            return True
        if drop < self.peak_drop_ratio:
            return False
        if not self._drop_spike_ok(m, spike_thr, chg_thr):
            return False
        if drop >= self.strong_drop_ratio:
            return True
        min_spike = self._min_event_spike(spike_thr)
        return (
            m["change_spike"] >= min_spike * 0.55
            or m["frame_drop_ratio"] >= self.frame_drop_ratio * 0.40
        )

    def _seam_event(self, m: dict, ewma_chg: float, side: str) -> bool:
        chg_thr, rise_thr, spike_thr = self._side_thresholds(side)
        chg = m["prof_change"]
        if chg < chg_thr:
            return False
        if self._drop_is_seam(m, spike_thr):
            return True
        rise = chg - ewma_chg
        if rise < rise_thr * self.event_rise_min_ratio:
            return False
        if m["drop_ratio"] >= self.peak_drop_ratio:
            if not self._drop_spike_ok(m, spike_thr, chg_thr):
                return False
        return self._has_dynamic(m, spike_thr)

    def arm(self, laser, det) -> bool:
        sm = getattr(det, "last_proj", None)
        if sm is None or not laser.found:
            return False
        pl = float(laser.peak_l)
        pr = float(laser.peak_r)
        if pl <= 0 and pr <= 0:
            return False
        ox = float(det.sx)
        self._baseline_l = pl
        self._baseline_r = pr
        if not self._capture_side(sm, ox, pl, "left"):
            return False
        if not self._capture_side(sm, ox, pr, "right"):
            return False
        self._fired_l = self._fired_r = False
        self.last_drop_l = self.last_drop_r = 0.0
        self.last_change_l = self.last_change_r = 0.0
        self.last_spike_l = self.last_spike_r = 0.0
        self._begin_cycle()
        return True

    def rearm_right(self, laser, det) -> bool:
        sm = getattr(det, "last_proj", None)
        if sm is None or not laser.found:
            return False
        pr = float(laser.peak_r)
        if pr <= 0:
            pr = self._baseline_r
        if pr is None or pr <= 0:
            return False
        ok = self._capture_side(sm, float(det.sx), pr, "right")
        if ok:
            self._sync_thresholds(apply_learned=True)
        return ok

    def _watch_start_frame(self, side: str) -> int:
        return self.warmup_frames + self.settle_frames

    def _measure_side(self, det, side: str):
        sm = getattr(det, "last_proj", None)
        if sm is None:
            return None
        win = self._win_l if side == "left" else self._win_r
        base_prof = self._prof_l if side == "left" else self._prof_r
        peak_base = self._peak_base_l if side == "left" else self._peak_base_r
        locked = self._locked_l if side == "left" else self._locked_r
        if win is None or base_prof is None or peak_base < 1e-6 or not locked:
            return None
        curr = self._profile_in_window(sm, win)
        if curr is None:
            return None
        if len(curr) != len(base_prof):
            n = min(len(curr), len(base_prof))
            curr = curr[:n]
            base = base_prof[:n]
        else:
            base = base_prof
        peak_curr = float(curr.max())
        peak_drop = peak_base - peak_curr
        drop_ratio = max(0.0, peak_drop / max(peak_base, 1e-6))
        prof_change = float(np.mean(np.abs(curr - base)))
        prev_peak = self._peak_prev_l if side == "left" else self._peak_prev_r
        frame_drop = max(0.0, prev_peak - peak_curr)
        frame_drop_ratio = frame_drop / max(peak_base, 1e-6)
        prev_chg = self._chg_prev_l if side == "left" else self._chg_prev_r
        if prev_chg > 0.002:
            change_spike = max(0.0, prof_change - prev_chg)
        else:
            change_spike = 0.0
        center_x = self._baseline_l if side == "left" else self._baseline_r
        return {
            "peak_curr": peak_curr,
            "peak_drop": peak_drop,
            "drop_ratio": drop_ratio,
            "prof_change": prof_change,
            "change_spike": change_spike,
            "frame_drop": frame_drop,
            "frame_drop_ratio": frame_drop_ratio,
            "x": center_x,
        }

    def _refresh_thresholds(self, side: str):
        self._sync_thresholds(apply_learned=True)

    def _update_side(self, det, side: str) -> Tuple[bool, SeamResult]:
        tag = "L" if side == "left" else "R"
        fired = self._fired_l if side == "left" else self._fired_r
        if fired:
            bx = self._baseline_l if side == "left" else self._baseline_r
            return True, SeamResult(
                hit=True, visible=True, x_peak=bx or 0.0,
                phase="DROP", debug="latched",
            )

        if side == "left":
            frame_n = self._frames_l
            self._frames_l += 1
        else:
            frame_n = self._frames_r
            self._frames_r += 1

        locked = self._locked_l if side == "left" else self._locked_r
        if frame_n < self.warmup_frames:
            sm = getattr(det, "last_proj", None)
            if sm is not None:
                win = self._win_l if side == "left" else self._win_r
                if win is not None:
                    prof = self._profile_in_window(sm, win)
                    if prof is not None:
                        base = self._prof_l if side == "left" else self._prof_r
                        if base is not None and len(prof) == len(base):
                            chg = float(np.mean(np.abs(prof - base)))
                            if side == "left":
                                self.last_change_l = chg
                                self._chg_prev_l = chg
                            else:
                                self.last_change_r = chg
                                self._chg_prev_r = chg
                            self._auto.note_quiet(side, chg, 0.0)
            if frame_n == self.warmup_frames - 1:
                self._lock_watch_baseline(side, det)
            return False, SeamResult(
                hit=False, visible=True,
                x_peak=(self._baseline_l if side == "left" else self._baseline_r) or 0.0,
                phase="WARMUP",
                debug="peak{} warmup {}/{}".format(tag, frame_n + 1, self.warmup_frames),
            )

        settle_end = self._watch_start_frame(side)
        if frame_n < settle_end:
            if not locked:
                self._lock_watch_baseline(side, det)
                locked = self._locked_l if side == "left" else self._locked_r
            m = self._measure_side(det, side)
            if m is not None:
                if side == "left":
                    self.last_change_l = m["prof_change"]
                    self.last_spike_l = m["change_spike"]
                else:
                    self.last_change_r = m["prof_change"]
                    self.last_spike_r = m["change_spike"]
                self._update_ewma(side, m["prof_change"])
                chg_thr, _, _ = self._side_thresholds(side)
                if m["prof_change"] < chg_thr * 0.92 and m["drop_ratio"] < self._auto.noise_max_drop:
                    self._auto.note_quiet(side, m["prof_change"], m["change_spike"])
                if side == "left":
                    self._chg_prev_l = m["prof_change"]
                else:
                    self._chg_prev_r = m["prof_change"]
            if frame_n == settle_end - 1:
                self._lock_watch_baseline(side, det)
            sn = frame_n - self.warmup_frames + 1
            return False, SeamResult(
                hit=False, visible=True,
                x_peak=(self._baseline_l if side == "left" else self._baseline_r) or 0.0,
                phase="SETTLE",
                debug="peak{} settle {}/{}".format(tag, sn, self.settle_frames),
            )

        if not locked:
            self._lock_watch_baseline(side, det)
            locked = self._locked_l if side == "left" else self._locked_r

        m = self._measure_side(det, side)
        if m is None:
            return False, SeamResult(phase="---", debug="no base {}".format(tag))

        chg_thr, rise_thr, _ = self._side_thresholds(side)

        if side == "left":
            self.last_drop_l = m["drop_ratio"]
            self.last_change_l = m["prof_change"]
            self.last_spike_l = m["change_spike"]
        else:
            self.last_drop_r = m["drop_ratio"]
            self.last_change_r = m["prof_change"]
            self.last_spike_r = m["change_spike"]

        ewma = self._update_ewma(side, m["prof_change"])
        rise = m["prof_change"] - ewma
        event = self._seam_event(m, ewma, side)

        if not event and m["prof_change"] < chg_thr * 0.92 and m["drop_ratio"] < self._auto.noise_max_drop:
            self._auto.note_quiet(side, m["prof_change"], m["change_spike"])
            total_quiet = len(self._auto._noise_l) + len(self._auto._noise_r)
            if total_quiet > 0 and total_quiet % 15 == 0:
                self._refresh_thresholds(side)
            chg_thr, rise_thr, _ = self._side_thresholds(side)

        need_cf = self.confirm_frames
        spike_thr = self._side_thresholds(side)[2]
        min_spike = self._min_event_spike(spike_thr)
        gradual_evt = self._gradual_seam_drop_ok(m, chg_thr)
        if gradual_evt:
            need_cf = max(2, self.confirm_frames - 1)
        if (
            m["drop_ratio"] >= self.fast_confirm_drop_ratio
            and m["change_spike"] >= self.drop_spike_abs_min
        ):
            need_cf = max(1, self.fast_confirm_frames)
        elif (
            m["change_spike"] >= min_spike * self.fast_confirm_spike_ratio
            and m["drop_ratio"] >= self.spike_only_min_drop
        ):
            need_cf = max(1, self.fast_confirm_frames)
        elif (
            m["drop_ratio"] >= self.peak_drop_ratio * 0.70
            and m["change_spike"] >= min_spike * 0.85
        ):
            need_cf = max(1, self.confirm_frames - 1)

        if side == "left":
            if event:
                self._confirm_l += 1
            else:
                self._confirm_l = 0
            if event and m["drop_ratio"] >= self.fast_confirm_drop_ratio:
                if m["change_spike"] >= self.drop_spike_abs_min:
                    self._confirm_l = max(self._confirm_l, need_cf)
            self._chg_prev_l = m["prof_change"]
            if m["drop_ratio"] < self.peak_drop_ratio:
                self._peak_prev_l = m["peak_curr"]
            confirm = self._confirm_l
        else:
            if event:
                self._confirm_r += 1
            else:
                self._confirm_r = 0
            if event and m["drop_ratio"] >= self.fast_confirm_drop_ratio:
                if m["change_spike"] >= self.drop_spike_abs_min:
                    self._confirm_r = max(self._confirm_r, need_cf)
            self._chg_prev_r = m["prof_change"]
            if m["drop_ratio"] < self.peak_drop_ratio:
                self._peak_prev_r = m["peak_curr"]
            confirm = self._confirm_r

        if event and confirm >= need_cf:
            if side == "left":
                self._fired_l = True
                self._pending_left_hit = dict(m)
            else:
                self._fired_r = True
                self.complete_measure(m)
            return True, SeamResult(
                hit=True, visible=True, x_peak=m["x"],
                phase="DROP",
                debug=(
                    "peak{} drop {:.0%} chg {:.3f} spike {:.3f}".format(
                        tag, m["drop_ratio"], m["prof_change"], m["change_spike"],
                    )
                ),
            )

        return False, SeamResult(
            hit=False, visible=True, x_peak=m["x"],
            phase="WATCH",
            debug=(
                "peak{} drop {:.0%} chg {:.3f}/{:.3f} rise {:.3f} spike {:.3f}".format(
                    tag, m["drop_ratio"], m["prof_change"],
                    chg_thr, rise, m["change_spike"],
                )
            ),
        )

    def check_left(self, laser, det) -> Tuple[bool, SeamResult]:
        if not laser.found:
            return False, SeamResult(phase="---", debug="no laser L")
        return self._update_side(det, "left")

    def check_right(self, laser, det) -> Tuple[bool, SeamResult]:
        if not laser.found:
            return False, SeamResult(phase="---", debug="no laser R")
        return self._update_side(det, "right")

    def hud_line(self) -> str:
        at = self._auto
        if at.enabled:
            learned = ""
            if at._learned_chg is not None:
                learned = " hit~{:.3f}".format(at._learned_chg)
            return (
                "auto chg>={:.3f} noiseL={:.3f} cf={}{}  "
                "L {:.3f} sp {:.3f} | R {:.3f} sp {:.3f}".format(
                    at.last_chg_thr,
                    at.last_noise_l,
                    self.confirm_frames,
                    learned,
                    self.last_change_l,
                    self.last_spike_l,
                    self.last_change_r,
                    self.last_spike_r,
                )
            )
        return (
            "chg>={:.3f} warmup={}+{} cf={}  "
            "L chg {:.3f} sp {:.3f} | R chg {:.3f} sp {:.3f}".format(
                self.profile_change_min,
                self.warmup_frames,
                self.settle_frames,
                self.confirm_frames,
                self.last_change_l,
                self.last_spike_l,
                self.last_change_r,
                self.last_spike_r,
            )
        )
