"""
Детекция стыка в вертикальной ROI (bg_diff).
|кадр − эталон без стыка| + центральная линия >= min_center_line_ratio.
"""
import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class SeamResult:
    hit: bool = False
    visible: bool = False
    x_peak: float = 0.0
    x_left: int = 0
    x_right: int = 0
    prominence: float = 0.0
    v_span: int = 0
    peak_w: int = 0
    phase: str = "---"
    debug: str = ""


class _SeamState:
    def __init__(self):
        self.track = "OUT"
        self.confirm = 0
        self.fired = False
        self.smooth_x = None
        self.prev_ix = None
        self.prev_in_center = False
        self.prev_x0 = None
        self.prev_x1 = None

    def reset(self):
        self.track = "OUT"
        self.confirm = 0
        self.fired = False
        self.smooth_x = None
        self.prev_ix = None
        self.prev_in_center = False
        self.prev_x0 = None
        self.prev_x1 = None


class _PulseState:
    """Состояние center_pulse / общая логика HIT по центральной линии."""

    def __init__(self):
        self.confirm = 0
        self.fired = False
        self.prev_line_ok = False
        self.track = "OUT"
        self.line_armed = False
        self.hit_hold = 0
        self.hit_hold = 0

    def reset(self):
        self.confirm = 0
        self.fired = False
        self.prev_line_ok = False
        self.track = "OUT"
        self.line_armed = False
        self.hit_hold = 0


class _BgDiffState(_PulseState):
    """Эталон ROI без стыка + сравнение с текущим кадром."""

    def __init__(self):
        super().__init__()
        self.ref = None
        self.samples = []
        self.calm = 0
        self.bg_updates = 0
        self.ref_freeze = 0

    def reset(self):
        super().reset()
        self.ref = None
        self.samples = []
        self.calm = 0
        self.bg_updates = 0
        self.ref_freeze = 0

    def set_ref(self, gray: np.ndarray):
        self.ref = gray.copy()
        self.samples = [gray.copy()]
        self.calm = 0
        self.ref_freeze = int(0)


def capture_bg_ref(state, gray: np.ndarray):
    """Запомнить фон (стыка в ROI сейчас нет)."""
    if state is None:
        state = _BgDiffState()
    state.set_ref(gray)
    return state


def reset_track_state(state):
    """Сброс HIT/confirm без удаления эталона bg_diff."""
    if state is None:
        return None
    state.confirm = 0
    state.fired = False
    state.prev_line_ok = False
    state.track = "OUT"
    state.line_armed = False
    state.hit_hold = 0
    return state


def _effective_confirm_frames(m: dict, p: dict) -> int:
    """Меньше кадров подтверждения, если стык виден почти на всю высоту ROI."""
    cf = max(1, int(p.get("confirm_frames", 2)))
    ratio = float(m.get("ratio", 0.0))
    fast_r = float(p.get("fast_confirm_ratio", 0.5))
    if ratio >= fast_r:
        return max(1, min(cf, int(p.get("fast_confirm_frames", 1))))
    return cf


def reset_bg_state(state):
    """Полный сброс эталона фона — вызывать по START с ПЛК (до bg INIT)."""
    if state is None:
        return None
    if isinstance(state, _BgDiffState):
        state.reset()
        return state
    if hasattr(state, "reset"):
        state.reset()
        return state
    return None


def arm_detection_on_start(seam_l, seam_r, params):
    """Подготовка к измерению: фон bg_diff снимается заново после START."""
    if params.get("detect_mode") == "bg_diff":
        return reset_bg_state(seam_l), reset_bg_state(seam_r)
    return reset_track_state(seam_l), reset_track_state(seam_r)


def _line_ok_hysteresis(state, m: dict, p: dict) -> bool:
    """Гистерезис по доле линии; wide/empty блокируют, short/narrow — нет."""
    min_r = float(p.get("min_center_line_ratio", 0.5))
    hyst = float(p.get("line_hysteresis", 0.10))
    off_r = max(0.12, min_r - hyst)
    ratio = float(m.get("ratio", 0.0))
    sdbg = str(m.get("sdbg", ""))
    blocked = sdbg.startswith("wide=") or sdbg.startswith("empty")

    if state.line_armed:
        if ratio < off_r or blocked:
            state.line_armed = False
    elif ratio >= min_r and not blocked:
        state.line_armed = True
    return state.line_armed


def _edge_margin(w: int, ratio: float = 0.05) -> int:
    return max(1, int(w * ratio))


def _in_band(ix: int, w: int, ratio: float = 0.05) -> bool:
    m = _edge_margin(w, ratio)
    return m <= ix <= w - 1 - m


def _center_bounds(w: int, ratio: float = 0.35):
    cx = (w - 1) / 2.0
    half = max(1, int(w * ratio / 2))
    return max(0, int(cx - half)), min(w - 1, int(cx + half))


def _smooth_x(st: _SeamState, ix: int, alpha: float = 0.4) -> int:
    if st.smooth_x is None:
        st.smooth_x = float(ix)
    else:
        st.smooth_x = alpha * ix + (1.0 - alpha) * st.smooth_x
    return int(round(st.smooth_x))


def _overlap_len(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def _region_in_center(x0: int, x1: int, w: int, center_ratio: float) -> bool:
    lo_c, hi_c = _center_bounds(w, center_ratio)
    return _overlap_len(x0, x1, lo_c, hi_c) > 0


def _region_entered_center(prev_in: bool, now_in: bool) -> bool:
    return not prev_in and now_in


def _region_crossed_center(prev_x0, prev_x1, x0: int, x1: int, w: int, center_ratio: float) -> bool:
    if prev_x0 is None or prev_x1 is None:
        return False
    lo_c, hi_c = _center_bounds(w, center_ratio)
    was = _overlap_len(prev_x0, prev_x1, lo_c, hi_c)
    now = _overlap_len(x0, x1, lo_c, hi_c)
    return was == 0 and now > 0


def _centroid_crossed_line(prev_ix, ix: int, w: int) -> bool:
    if prev_ix is None:
        return False
    cx = (w - 1) // 2
    return (prev_ix < cx <= ix) or (prev_ix > cx >= ix)


def _update_track(st: _SeamState, sx: int, w: int, center_ratio: float) -> str:
    lo_c, hi_c = _center_bounds(w, center_ratio)
    tr = st.track
    if tr == "OUT":
        if sx < lo_c:
            st.track = "ENTER"
        elif sx <= hi_c:
            st.track = "CENTER"
        else:
            st.track = "EXIT"
    elif tr == "ENTER":
        if lo_c <= sx <= hi_c:
            st.track = "CENTER"
        elif sx > hi_c:
            st.track = "EXIT"
    elif tr == "CENTER" and sx > hi_c:
        st.track = "EXIT"
    elif tr == "EXIT":
        st.track = "DONE"
    return st.track


def _display_phase(track: str) -> str:
    return "---" if track in ("OUT", "DONE") else track


def _cap_median_k(w: int, h: int, k: int) -> int:
    k = max(5, int(k) | 1)
    cap = max(5, min(w, h) // 3)
    if cap % 2 == 0:
        cap -= 1
    return min(k, cap)


def _peak_ratio(proj: np.ndarray, ix: int, x0: int, x1: int, w: int) -> float:
    peak = float(proj[ix]) if 0 <= ix < len(proj) else 0.0
    if peak <= 0:
        return 0.0
    p2 = proj.copy()
    p2[max(0, x0): min(len(p2), x1 + 1)] = 0
    guard = max(3, int(w * 0.08))
    p2[max(0, ix - guard): min(len(p2), ix + guard + 1)] = 0
    second = float(np.max(p2))
    if second < 1e-3:
        rest = proj[(proj > 0)]
        second = float(np.median(rest)) if rest.size else 1.0
    return min(peak / (second + 1e-6), 30.0)


def _band_around_peak(proj: np.ndarray, ix: int, half_ratio: float, max_band_px: int) -> tuple:
    """Локальная полоса вокруг пика проекции (не по всей ROI)."""
    peak_val = float(proj[ix])
    if peak_val <= 1e-6:
        return ix, ix, ix
    thr = peak_val * half_ratio
    x0, x1 = ix, ix
    while x0 > 0 and float(proj[x0 - 1]) >= thr:
        x0 -= 1
    while x1 < len(proj) - 1 and float(proj[x1 + 1]) >= thr:
        x1 += 1
    width = x1 - x0 + 1
    if width > max_band_px:
        half = max_band_px // 2
        x0 = max(0, ix - half)
        x1 = min(len(proj) - 1, ix + half)
    weights = proj[x0: x1 + 1]
    if float(np.sum(weights)) > 1e-6:
        ix = x0 + int(round(float(np.average(np.arange(x0, x1 + 1), weights=weights))))
    return x0, x1, ix


def _smooth_proj(proj: np.ndarray, k: int = 5) -> np.ndarray:
    k = max(3, int(k) | 1)
    if proj.size < k:
        return proj
    ker = np.ones(k, dtype=np.float32) / k
    return np.convolve(proj, ker, mode="same")


def _build_seam_mask(gray: np.ndarray, p: dict) -> tuple:
    h, w = gray.shape[:2]
    thr = float(p.get("grad_threshold", 35))

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    mag = np.abs(gx)
    grad_mask = (mag >= thr).astype(np.uint8) * 255
    mask = grad_mask

    if p.get("use_texture_diff", True):
        mk = _cap_median_k(w, h, p.get("diff_median_k", 9))
        bg = cv2.medianBlur(gray, mk)
        diff = cv2.absdiff(gray, bg)
        dfix = float(p.get("diff_threshold", 0))
        if dfix > 0:
            dthr = dfix
        else:
            pct = float(p.get("diff_percentile", 88))
            dthr = float(np.percentile(diff, pct))
        tex = (diff >= dthr).astype(np.uint8) * 255
        if p.get("texture_near_grad", True):
            kw = max(5, int(p.get("texture_grad_span_px", 11)) | 1)
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
            near = cv2.dilate(grad_mask, k, iterations=1)
            tex = cv2.bitwise_and(tex, near)
        mask = cv2.bitwise_or(grad_mask, tex)

    mh = max(3, int(p.get("vertical_morph_h", 5)) | 1)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, mh))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask, mag


def _region_from_mask(mask: np.ndarray, mag: np.ndarray, p: dict) -> tuple:
    h, w = mask.shape[:2]
    if w < 3 or h < 3:
        return 0, 0, 0, np.zeros(w), 0, 0, "tiny"

    col_sum = np.sum(mask > 0, axis=0).astype(np.float32)
    mag_proj = np.sum(mag, axis=0).astype(np.float32)
    proj = np.maximum(col_sum, mag_proj * 0.5)

    edge_ratio = float(p.get("edge_margin_ratio", 0.05))
    margin = _edge_margin(w, edge_ratio)
    if margin > 0 and proj.size > margin * 2 + 1:
        proj[:margin] = 0
        proj[-margin:] = 0

    if np.max(proj) <= 0:
        return 0, w - 1, 0, proj, 0, 0, "empty"

    sk = int(p.get("proj_smooth_k", 3))
    if sk > 1:
        proj = _smooth_proj(proj, sk)

    ix = int(np.argmax(proj))
    half_ratio = float(p.get("band_half_ratio", 0.55))
    max_band = int(p.get("max_band_half_px", 0))
    if max_band <= 0:
        max_band = int(p.get("max_peak_width_px", 24))
    x0, x1, ix = _band_around_peak(proj, ix, half_ratio, max_band)

    peak_w = x1 - x0 + 1
    col_half = max(1, int(p.get("v_span_column_half", 2)))
    c0, c1 = max(0, ix - col_half), min(w, ix + col_half + 1)
    band = mask[:, c0:c1]
    row_on = np.any(band > 0, axis=1)
    v_span = int(np.where(row_on)[0][-1] - np.where(row_on)[0][0] + 1) if np.any(row_on) else 0

    return x0, x1, ix, proj, v_span, peak_w, "ok"


def _seam_geometry_region(x0: int, x1: int, v_span: int, h: int, w: int, p: dict) -> tuple:
    width = x1 - x0 + 1
    min_w = int(p.get("min_peak_width_px", 2))
    max_w = int(p.get("max_peak_width_px", 38))
    min_span = int(float(p.get("min_vertical_span_ratio", 0.25)) * h)
    max_span = int(float(p.get("max_vertical_span_ratio", 0.88)) * h)

    if width < min_w:
        return False, f"narrow={width}"
    if width > max_w:
        return False, f"wide={width}"
    if v_span < min_span:
        return False, f"short={v_span}"
    if v_span > max_span:
        return False, f"tall={v_span}"
    return True, f"ok w={width} v={v_span}"


def _measure_seam(gray: np.ndarray, p: dict) -> SeamResult:
    r = SeamResult()
    if gray is None or gray.size == 0:
        return r
    h, w = gray.shape[:2]
    if h < 8 or w < 3:
        return r

    mask, mag = _build_seam_mask(gray, p)
    x0, x1, ix, proj, v_span, peak_w, mdbg = _region_from_mask(mask, mag, p)

    if mdbg == "empty":
        r.debug = f"empty thr={p.get('grad_threshold', 35):.0f}"
        return r

    ratio = _peak_ratio(proj, ix, x0, x1, w)
    r_min = float(p.get("prominence_on", 2.0))
    min_peak = float(p.get("min_projection_abs", 12.0))
    min_r = float(p.get("min_projection_ratio", 0.05))
    proj_ok = ix < len(proj) and float(proj[ix]) >= max(min_peak, min_r * h)

    in_band = _in_band(ix, w, p.get("edge_margin_ratio", 0.05))
    signal_ok = ratio >= r_min and proj_ok and in_band

    geom_ok, gdbg = _seam_geometry_region(x0, x1, v_span, h, w, p)
    lo_c, hi_c = _center_bounds(w, p.get("center_zone_ratio", 0.35))
    center_px = _overlap_len(x0, x1, lo_c, hi_c)
    min_center_px = max(1, int(p.get("min_center_overlap_px", 2)))
    center_ok = center_px >= min_center_px

    ok = signal_ok and geom_ok and center_ok

    r.x_peak = float(ix)
    r.x_left, r.x_right = x0, x1
    r.prominence = ratio
    r.v_span = v_span
    r.peak_w = peak_w
    r.visible = ok
    r.phase = "IN" if ok else "---"
    sig = "sig" if signal_ok else "sig-"
    geo = "geo" if geom_ok else gdbg
    cen = f"cen={center_px}" if center_ok else f"cen={center_px}"
    r.debug = f"R={ratio:.2f}/{r_min:.2f} {sig} {geo} {cen} x={ix}[{x0}-{x1}]"
    return r


def _longest_run_1d(profile: np.ndarray) -> int:
    best = cur = 0
    for v in profile:
        if v:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def _grad_seam_mask(gray: np.ndarray, p: dict) -> np.ndarray:
    """Маска стыка: только вертикальный градиент >= thr (без заливки текстурой)."""
    thr = float(p.get("grad_threshold", 40))
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    mask = (np.abs(gx) >= thr).astype(np.uint8)
    mh = max(3, int(p.get("vertical_morph_h", 5)) | 1)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, mh))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)


def _center_line_from_mask(mask: np.ndarray, gray: np.ndarray, p: dict, thr_val: float, thr_name: str) -> dict:
    """Непрерывное пересечение центральной линии ROI по маске (diff или grad)."""
    h, w = gray.shape[:2]
    cx = w // 2
    min_ratio = float(
        p.get("min_center_line_ratio", p.get("min_pulse_v_span_ratio", 0.5))
    )
    line_half = max(0, int(p.get("center_line_half", 2)))
    max_gap = max(0, int(p.get("center_line_max_gap", 5)))
    min_w = int(p.get("min_pulse_width_px", 2))
    max_w = int(p.get("max_pulse_width_px", 20))

    c0 = max(0, cx - line_half)
    c1 = min(w, cx + line_half + 1)
    band = mask[:, c0:c1]
    band_px = max(1, band.size)
    hot_px = int(np.sum(band > 0))
    hot_pct = 100.0 * hot_px / band_px

    profile = (np.max(band, axis=1) > 0).astype(np.uint8)
    if max_gap > 0:
        kh = max(3, max_gap * 2 + 1)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kh))
        profile = cv2.morphologyEx(
            profile.reshape(h, 1), cv2.MORPH_CLOSE, k,
        ).reshape(h)

    run_len = _longest_run_1d(profile)
    ratio = run_len / float(h) if h > 0 else 0.0
    line_ok = ratio >= min_ratio

    ys, xs = np.where(band > 0)
    if ys.size == 0:
        return {
            "line_ok": False,
            "run_len": 0,
            "ratio": 0.0,
            "min_ratio": min_ratio,
            "width": 0,
            "x_peak": cx,
            "x0": cx,
            "x1": cx,
            "hot_px": 0,
            "hot_pct": hot_pct,
            "sdbg": f"empty {thr_name}={thr_val:.0f} hot={hot_pct:.0f}%",
            "thr": thr_val,
            "thr_name": thr_name,
        }

    width = int(xs.max() - xs.min() + 1)
    x_peak = int(round(float(xs.mean()))) + c0
    x0 = int(xs.min()) + c0
    x1 = int(xs.max()) + c0

    if width < min_w:
        line_ok = False
        sdbg = f"narrow={width}"
    elif width > max_w:
        line_ok = False
        sdbg = f"wide={width}"
    elif not line_ok:
        need = int(min_ratio * h)
        sdbg = f"short={run_len}/{need}"
    else:
        sdbg = "ok"

    return {
        "line_ok": line_ok,
        "run_len": run_len,
        "ratio": ratio,
        "min_ratio": min_ratio,
        "width": width,
        "x_peak": x_peak,
        "x0": x0,
        "x1": x1,
        "hot_px": hot_px,
        "hot_pct": hot_pct,
        "sdbg": sdbg,
        "thr": thr_val,
        "thr_name": thr_name,
    }


def _center_line_measure_grad(gray: np.ndarray, p: dict) -> dict:
    thr = float(p.get("grad_threshold", 40))
    mask = _grad_seam_mask(gray, p)
    return _center_line_from_mask(mask, gray, p, thr, "thr")


def _build_diff_mask(gray: np.ndarray, ref: np.ndarray, p: dict) -> tuple:
    diff = cv2.absdiff(gray, ref)
    kb = int(p.get("diff_blur_k", 3))
    if kb > 1:
        kb = kb | 1
        diff = cv2.GaussianBlur(diff, (kb, kb), 0)

    fixed = int(p.get("diff_threshold", 12))
    adapt_pct = float(p.get("diff_adaptive_pct", 0))
    if adapt_pct > 0:
        pctl = float(np.percentile(diff, adapt_pct))
        cap = float(p.get("diff_adaptive_cap", max(fixed * 3, 28)))
        dthr = max(fixed, min(pctl, cap))
    else:
        dthr = float(fixed)

    mask = (diff >= dthr).astype(np.uint8)
    ko = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    if int(p.get("diff_morph_open", 1)):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ko, iterations=1)
    mh = max(3, int(p.get("vertical_morph_h", 5)) | 1)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, mh))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    mean_diff = float(np.mean(diff))
    return mask, diff, float(dthr), mean_diff


def _bg_sync_shape(state: _BgDiffState, gray: np.ndarray) -> None:
    """ROI сменила размер (динамические зоны) — сбросить буфер bg_diff."""
    shape = gray.shape
    if state.ref is not None and state.ref.shape != shape:
        state.reset()
        return
    if state.samples and state.samples[0].shape != shape:
        state.samples = []
        state.ref = None
        state.calm = 0
        state.ref_freeze = 0


def _bg_bootstrap_ref(state: _BgDiffState, gray: np.ndarray, p: dict) -> bool:
    """Накопить первый эталон из нескольких кадров без стыка."""
    _bg_sync_shape(state, gray)
    state.samples.append(gray.copy())
    need = int(p.get("bg_init_frames", 8))
    if len(state.samples) < need:
        return False
    state.ref = np.median(np.stack(state.samples), axis=0).astype(np.uint8)
    state.samples = [state.ref.copy()]
    return True


def _bg_no_seam(m: dict, p: dict) -> bool:
    """В ROI нет стыка — можно медленно обновлять эталон (не при смене света)."""
    if m.get("line_ok"):
        return False
    ratio = float(m.get("ratio", 0.0))
    hot = float(m.get("hot_pct", 100.0))
    mean_d = float(m.get("mean_diff", 99.0))
    max_r = float(p.get("bg_max_seam_ratio", 0.08))
    max_h = float(p.get("bg_max_hot_pct", 4.0))
    max_mean = float(p.get("bg_max_mean_diff", 5.0))
    return ratio < max_r and hot < max_h and mean_d < max_mean


def _bg_auto_update_ref(state: _BgDiffState, gray: np.ndarray, p: dict, m: dict) -> bool:
    """
    Обновить эталон, пока стыка нет: плавно (EMA) или медианой буфера.
    Возвращает True, если ref изменился в этом кадре.
    """
    if not p.get("bg_auto_update", True) or state.ref is None:
        return False

    if state.ref_freeze > 0:
        state.ref_freeze -= 1
        state.calm = 0
        return False

    if not _bg_no_seam(m, p):
        state.calm = 0
        return False

    state.calm += 1
    calm_need = int(p.get("bg_calm_frames", 10))
    if state.calm < calm_need:
        return False

    mode = str(p.get("bg_update_mode", "ema")).lower()
    if mode == "median":
        cap = int(p.get("bg_median_n", 10))
        _bg_sync_shape(state, gray)
        state.samples.append(gray.copy())
        if len(state.samples) > cap:
            state.samples.pop(0)
        if len(state.samples) >= 3:
            state.ref = np.median(np.stack(state.samples), axis=0).astype(np.uint8)
            state.bg_updates += 1
            return True
    return False

    alpha = float(p.get("bg_ema_alpha", 0.02))
    if state.calm < calm_need * 2:
        alpha *= 0.5
    ref = state.ref.astype(np.float32)
    cur = gray.astype(np.float32)
    state.ref = np.clip((1.0 - alpha) * ref + alpha * cur, 0, 255).astype(np.uint8)
    state.bg_updates += 1
    return True


def _finalize_center_line(state, m: dict, cf: int, h: int, mode_tag: str, p: dict) -> SeamResult:
    r = SeamResult()
    line_ok = _line_ok_hysteresis(state, m, p)
    m["line_ok"] = line_ok

    if line_ok:
        state.confirm = min(state.confirm + 1, 10)
        state.track = "CENTER"
    else:
        state.confirm = max(state.confirm - 1, 0)
        if state.confirm == 0 and not state.fired:
            state.track = "OUT"

    cf_eff = _effective_confirm_frames(m, p)
    rising = (not state.prev_line_ok) and line_ok
    need_rise = bool(p.get("hit_need_rising", False))
    if line_ok and state.confirm >= cf_eff and not state.fired:
        if (rising if need_rise else True):
            r.hit = True
            state.fired = True
            state.hit_hold = int(p.get("hit_hold_frames", 8))

    hold = int(getattr(state, "hit_hold", 0))
    if state.fired and not r.hit and hold > 0:
        r.hit = True
        if not line_ok:
            state.hit_hold = hold - 1

    state.prev_line_ok = line_ok

    if isinstance(state, _BgDiffState):
        freeze_ratio = float(p.get("bg_freeze_trigger_ratio", 0.12))
        if line_ok or float(m.get("ratio", 0)) >= freeze_ratio:
            state.ref_freeze = int(p.get("bg_freeze_after_seam_frames", 45))

    r.x_peak = float(m["x_peak"])
    r.x_left = m["x0"]
    r.x_right = m["x1"]
    r.prominence = m["ratio"]
    r.v_span = m["run_len"]
    r.peak_w = m["width"]
    r.visible = line_ok
    r.phase = "CENTER" if r.hit else (state.track if line_ok else "---")

    pct = int(round(m["ratio"] * 100))
    need_pct = int(round(m["min_ratio"] * 100))
    tn = m.get("thr_name", "thr")
    r.debug = (
        f"{mode_tag} cline {m['run_len']}/{h} ({pct}%/{need_pct}%) {m['sdbg']} "
        f"{tn}={m['thr']:.0f} hot={m.get('hot_pct', 0):.0f}%"
    )
    if "mean_diff" in m:
        r.debug += f" md={m['mean_diff']:.1f}"
    r.debug += f" w={m['width']} | cf={state.confirm}/{cf_eff} hold={getattr(state, 'hit_hold', 0)} rise={int(rising)}"
    return r


def _detect_roi_bg_diff(gray, p, state=None):
    if state is None:
        state = _BgDiffState()
    h, w = gray.shape[:2]
    if h < 8 or w < 3:
        return SeamResult(), state

    cf = int(p.get("confirm_frames", 2))

    _bg_sync_shape(state, gray)

    if state.ref is None:
        if not _bg_bootstrap_ref(state, gray, p):
            r = SeamResult()
            need = int(p.get("bg_init_frames", 8))
            r.debug = f"bg INIT {len(state.samples)}/{need} (нет стыка в ROI)"
            return r, state

    mask, _diff, dthr, mean_diff = _build_diff_mask(gray, state.ref, p)
    m = _center_line_from_mask(mask, gray, p, dthr, "diff")
    m["mean_diff"] = mean_diff
    bg_upd = _bg_auto_update_ref(state, gray, p, m)
    r = _finalize_center_line(state, m, cf, h, "bg", p)
    if bg_upd:
        r.debug += f" bg↑#{state.bg_updates}"
    elif _bg_no_seam(m, p):
        r.debug += f" bg~{state.calm}"
    return r, state


def detect_roi(gray, _method_unused, p, state=None):
    return _detect_roi_bg_diff(gray, p, state)


# --- HUD ---
_FONT = cv2.FONT_HERSHEY_DUPLEX
_FS_SM, _FS_MD = 0.42, 0.52
_THIN, _MID = 1, 2

C_PANEL = (42, 44, 52)
C_TEXT = (235, 238, 242)
C_DIM = (155, 162, 175)
C_LEFT = (130, 210, 150)
C_LEFT_DIM = (80, 120, 90)
C_RIGHT = (210, 175, 130)
C_RIGHT_DIM = (110, 95, 75)
C_CENTER = (140, 200, 255)
C_AXIS = (100, 105, 115)
C_SEAM = (200, 230, 255)
C_EDGE_L = (160, 210, 255)
C_EDGE_R = (200, 160, 255)
C_HIT = (120, 230, 255)
C_WARN = (130, 200, 255)
C_OK = (140, 220, 160)


def _blend_rect(frame, x1, y1, x2, y2, color=C_PANEL, alpha=0.82):
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return
    patch = frame[y1:y2, x1:x2].copy()
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(frame[y1:y2, x1:x2], alpha, patch, 1.0 - alpha, 0, frame[y1:y2, x1:x2])


def _text_size(text, scale=_FS_MD):
    return cv2.getTextSize(text, _FONT, scale, _THIN)[0]


def _draw_text(frame, text, x, y, color=C_TEXT, scale=_FS_MD):
    cv2.putText(frame, text, (x, y), _FONT, scale, color, _THIN, cv2.LINE_AA)


def _draw_badge(frame, text, x, y, fg, scale=_FS_SM):
    tw, th = _text_size(text, scale)
    pad_x, pad_y = 6, 4
    x1, y1 = x, y - th - pad_y
    x2, y2 = x + tw + pad_x * 2, y + pad_y
    _blend_rect(frame, x1, y1, x2, y2, C_PANEL, 0.9)
    _draw_text(frame, text, x + pad_x, y, fg, scale)


def _phase_color(phase, accent):
    if phase == "CENTER":
        return C_OK
    if phase == "HIT":
        return C_HIT
    if phase == "ENTER":
        return C_WARN
    if phase == "EXIT":
        return C_EDGE_R
    if phase == "---":
        return C_DIM
    return accent


def draw_roi_viz(frame, rect, res, side, active=False, show_detail=True, center_ratio=0.35):
    x, y, rw, rh = rect
    accent = C_LEFT if side == "L" else C_RIGHT
    accent_dim = C_LEFT_DIM if side == "L" else C_RIGHT_DIM
    border = accent if active else accent_dim

    cv2.rectangle(frame, (x, y), (x + rw, y + rh), border, _MID)
    cx = x + rw // 2
    cv2.line(frame, (cx, y + 2), (cx, y + rh - 2), C_AXIS, 1, cv2.LINE_AA)
    lo, hi = _center_bounds(rw, center_ratio)
    cv2.line(frame, (x + lo, y), (x + lo, y + rh), C_CENTER, 1, cv2.LINE_AA)
    cv2.line(frame, (x + hi, y), (x + hi, y + rh), C_CENTER, 1, cv2.LINE_AA)

    tag = f"{side}  {'CENTER*' if res.hit else res.phase}"
    if res.visible:
        pct = int(round(res.prominence * 100)) if res.prominence <= 1.0 else int(res.prominence)
        tag += f"  fill={pct}% v={res.v_span} w={res.peak_w}"

    fw = frame.shape[1]
    tag_y = y + min(24, max(18, rh // 3))
    tw, _ = _text_size(tag, _FS_SM)
    tag_x = max(8, x - tw - 20) if side == "L" else min(fw - tw - 20, x + rw + 10)
    _draw_badge(frame, tag, tag_x, tag_y, _phase_color(res.phase, accent))

    if not show_detail or not res.visible:
        return

    px = x + int(round(res.x_peak))
    px = max(x, min(x + rw - 1, px))
    lx = x + max(0, min(rw - 1, res.x_left))
    rx = x + max(0, min(rw - 1, res.x_right))
    cv2.line(frame, (lx, y + 2), (lx, y + rh - 2), C_EDGE_L, 1, cv2.LINE_AA)
    cv2.line(frame, (rx, y + 2), (rx, y + rh - 2), C_EDGE_R, 1, cv2.LINE_AA)
    cv2.line(frame, (px, y + 1), (px, y + rh - 1), C_SEAM, 2, cv2.LINE_AA)


def draw_overlay(frame, l_res, r_res, l_rect, r_rect, center_ratio,
                 status, last_speed, err_code, thr, prom_on, max_w,
                 modbus_bits, tuning, act_side, speed_scale=1000):
    h, w = frame.shape[:2]
    draw_roi_viz(frame, l_rect, l_res, "L", act_side == "left", True, center_ratio)
    draw_roi_viz(frame, r_rect, r_res, "R", act_side == "right", True, center_ratio)

    row_h = 22
    pad = 12
    lines_top = [
        ("LEFT", f"{'HIT' if l_res.hit else l_res.phase}", l_res.visible, C_LEFT),
        ("RIGHT", f"{'HIT' if r_res.hit else r_res.phase}", r_res.visible, C_RIGHT),
        ("STATE", status, True, C_TEXT),
    ]
    if last_speed > 0:
        v = int(last_speed)
        if speed_scale == 1000:
            spd = f"{v // speed_scale}.{v % speed_scale:03d} mm/s"
        else:
            spd = f"{v / speed_scale:.3f} mm/s"
        lines_top.append(("SPEED", spd, True, C_OK))
    if err_code and int(err_code) > 0:
        lines_top.append(("ERR", str(int(err_code)), True, C_WARN))

    panel_h = pad * 2 + row_h * len(lines_top)
    _blend_rect(frame, 8, 8, min(w - 8, 320), 8 + panel_h, C_PANEL, 0.88)
    y = 8 + pad + 14
    for label, val, on, col in lines_top:
        _draw_text(frame, label, 18, y, C_DIM, _FS_SM)
        _draw_text(frame, val, 88, y, col if on else C_DIM, _FS_MD)
        y += row_h

    mb = f"L={modbus_bits & 1} R={(modbus_bits >> 1) & 1}  J={(modbus_bits >> 2) & 1}"
    foot_h = 52 if tuning else 30
    fy0 = h - foot_h - 8
    _blend_rect(frame, 8, fy0, w - 8, h - 8, C_PANEL, 0.88)
    _draw_text(frame, f"T={thr:.0f} cline={prom_on:.0%} maxW={max_w}", 18, fy0 + 18, C_DIM, _FS_SM)
    _draw_text(frame, f"Modbus {mb}", 18, fy0 + 36, C_TEXT, _FS_SM)
    if tuning:
        _draw_text(
            frame,
            "TUNE: t=panel P=save r/o/e/y modes — see console/HUD",
            18, fy0 + 52, C_WARN, _FS_SM,
        )
