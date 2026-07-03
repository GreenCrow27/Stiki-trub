"""
Скорость трубы: левая ROI → старт, plправая ROI → стоп, Modbus SPEED.
"""
import cv2
import time

from config_utils import load_config, save_config
from camera_io import open_camera, open_media_source, blur_rois_gray, VideoFileStream
from roi_manager import ROIManager
from laser_detector import LaserDetector, rois_from_lasers
from laser_seam import LaserSeamTracker
from laser_ui import draw_laser_overlay
from vision_utils import (
    SeamResult,
    detect_roi,
    draw_overlay,
    capture_bg_ref,
    reset_track_state,
    arm_detection_on_start,
)
from modbus_io import (
    ModbusIO,
    ERR_NONE,
    ERR_GENERAL,
    ERR_NO_DIST,
    ERR_TIMEOUT,
    ERR_NO_LASER,
    ERR_INVALID_DIST,
    ERR_MODBUS,
    ERR_CAMERA,
    ERR_LASER_UNSTABLE,
    ERR_BAD_MEASURE,
    ERR_BAD_START,
    ERR_ROI,
    check_distance_mm,
    error_status_text,
)
from tuning_controls import KeyboardTuning
from video_recorder import RawVideoRecorder
from measure_log import MeasureCsvLog

OFF, SEARCH, WAIT_L, WAIT_R = "OFF", "SEARCH", "WAIT_LEFT", "WAIT_RIGHT"


class _LaserGrace:
    """
    Краткий пропуск лазера: hold_frames кадров держим последний OK,
    затем error_frames без лазера → ошибка (только если allow_error).
    """

    def __init__(self, hold_frames: int, error_frames: int):
        self.hold_frames = max(1, int(hold_frames))
        self.error_frames = max(1, int(error_frames))
        self._last_good = None
        self._grace_left = 0
        self._miss_after_grace = 0

    def reset(self):
        self._last_good = None
        self._grace_left = 0
        self._miss_after_grace = 0

    def update(self, laser, *, allow_error: bool = True):
        raw_ok = laser is not None and laser.found
        if raw_ok:
            self._last_good = laser
            self._grace_left = self.hold_frames
            self._miss_after_grace = 0
            return laser, True, False

        if self._last_good is not None and self._grace_left > 0:
            self._grace_left -= 1
            held = self._last_good
            return held, True, False

        self._miss_after_grace += 1
        should_err = allow_error and self._miss_after_grace >= self.error_frames
        return None, False, should_err

    @property
    def holding(self):
        return self._grace_left > 0 and self._last_good is not None

    @property
    def grace_left(self):
        return self._grace_left

    @property
    def miss_frames(self):
        return self._miss_after_grace


def _parse_key(delay_ms=30):
    """waitKeyEx: отдельно символ и Shift (для Shift++ / Shift+-)."""
    k = cv2.waitKeyEx(delay_ms)
    if k < 0:
        return None, False
    ch = k & 0xFF
    ext = (k >> 16) & 0xFF
    shifted = ext not in (0, 255, 254)
    return ch, shifted


def _resolve_dt_fps(stream, cam, meas_cfg):
    """FPS для режима dt_mode=fps (0 → stream → camera → 25)."""
    fps = float(meas_cfg.get("dt_fps") or 0)
    if fps <= 0:
        fps = float(getattr(stream, "fps", 0) or 0)
    if fps <= 0:
        fps = float(cam.get("fps") or 25)
    return fps


def _measure_dt_ms(mode, left_ts_ms, right_ts_ms, left_frame, right_frame, fps):
    """
    timer  — perf_counter между LEFT и RIGHT (мс).
    fps    — (кадр_RIGHT − кадр_LEFT) × 1000 / fps.
    hybrid — max(timer, fps) при RTSP/очереди кадров.
    """
    dt_fps = 0
    n = int(right_frame) - int(left_frame)
    if n > 0 and fps > 0:
        dt_fps = max(1, int(round(n * 1000.0 / float(fps))))
    dt_timer = 0
    if left_ts_ms > 0 and right_ts_ms > left_ts_ms:
        dt_timer = max(1, int(right_ts_ms - left_ts_ms))
    if mode == "hybrid":
        if dt_fps > 0 and dt_timer > 0:
            return max(dt_fps, dt_timer)
        return dt_fps or dt_timer
    if mode == "fps":
        return dt_fps if dt_fps > 0 else dt_timer
    return dt_timer if dt_timer > 0 else dt_fps


def _begin_measure(
    modbus, cmd, use_cfg_dist, cfg_dist_mm, manual=False,
    min_dist_mm=10, max_dist_mm=5000,
):
    """
    Старт цикла L→R. Возвращает (ok, dist_mm для скорости).
    Modbus: HR dist_mm; бит DIST_OK — подтверждение (если HR258=0).
    """
    if use_cfg_dist:
        dist = int(cfg_dist_mm)
    else:
        dist, dist_read_ok = modbus.read_dist_mm_status()
        dist = int(dist)
        if not manual:
            has_dist_ok = modbus.dist_ok(cmd)
            if not has_dist_ok and dist <= 0:
                hr = modbus.reg_plc["dist_mm"]
                cmd_raw = int(cmd) & 0xFFFF
                hint = (
                    f"CMD={cmd_raw} без DIST_OK (бит4, +16 → {cmd_raw + 16})"
                    if not has_dist_ok
                    else ""
                )
                read_hint = (
                    f", чтение HR{hr} не удалось"
                    if not dist_read_ok
                    else f", HR{hr}={dist}"
                )
                print(
                    f"[Modbus] ERR 2: нет дистанции — {hint}{read_hint}"
                )
                modbus.write_error(ERR_NO_DIST)
                return False, 0
            if not has_dist_ok and dist > 0:
                print(
                    f"[Modbus] HR{modbus.reg_plc['dist_mm']}={dist} мм "
                    f"(DIST_OK не установлен — принято по HR)"
                )
    err = check_distance_mm(dist, min_dist_mm, max_dist_mm)
    if err:
        if err == ERR_NO_DIST:
            print(
                f"[Modbus] ERR 2: HR{modbus.reg_plc['dist_mm']}={dist} "
                f"(≤ 0 или нет данных)"
            )
        elif err == ERR_INVALID_DIST:
            print(
                f"[Modbus] ERR 5: HR{modbus.reg_plc['dist_mm']}={dist} "
                f"вне [{min_dist_mm}..{max_dist_mm}]"
            )
        modbus.write_error(err)
        return False, 0
    modbus.reset_on_start()
    modbus.measure_active(True)
    return True, dist


def _roi_valid(fw, fh, left, right, min_gap_px=15):
    for r in (left, right):
        if int(r.get("w", 0)) <= 0 or int(r.get("h", 0)) <= 0:
            return False
        if int(r["x"]) < 0 or int(r["y"]) < 0:
            return False
        if int(r["x"]) + int(r["w"]) > fw or int(r["y"]) + int(r["h"]) > fh:
            return False
    gap = int(right["x"]) - (int(left["x"]) + int(left["w"]))
    return gap >= int(min_gap_px)


def _laser_roi_placeholder(w, h):
    return {"x": 0, "y": 0, "w": w, "h": h}


def _clear_laser_rois(roi, fw, fh, roi_w, roi_h):
    ph = _laser_roi_placeholder(roi_w, roi_h)
    roi.left.update(ph)
    roi.right.update(ph.copy())
    roi.clamp(fw, fh)


def _laser_gap_ok(laser, sw, min_ratio, max_ratio):
    if not laser.found or laser.gap_px <= 0:
        return False
    gap_ratio = laser.gap_px / max(1.0, float(sw))
    return min_ratio <= gap_ratio <= max_ratio


def _update_laser_stable(laser, det, prev_x0, prev_x1, max_jitter, min_gap, max_gap):
    """Стабильный кадр лазера: 2 полоски, зазор OK, мало дрожания."""
    if not _laser_gap_ok(laser, det.sw, min_gap, max_gap):
        return False, None, None
    x0_i = int(round(laser.x0))
    x1_i = int(round(laser.x1))
    if prev_x0 is not None and (
        abs(x0_i - prev_x0) > max_jitter or abs(x1_i - prev_x1) > max_jitter
    ):
        return False, None, None
    return True, x0_i, x1_i


def _start_from_lasers(
    roi, laser, laser_det, fw, fh,
    roi_w, roi_h, off_l_x, off_r_x, off_y,
    modbus, cmd, use_cfg_dist, cfg_dist_mm,
    seam_l, seam_r, params, calib_autosave, cfg, l_cfg,
    seam_tracker=None,
    min_dist_mm=10, max_dist_mm=5000,
):
    """ROI от лазеров → цикл измерения WAIT_L."""
    left_roi, right_roi = rois_from_lasers(
        laser, roi_w, roi_h, off_l_x, off_r_x, off_y, fw, fh,
    )
    roi.left.update(left_roi)
    roi.right.update(right_roi)
    roi.clamp(fw, fh)
    if not _roi_valid(fw, fh, roi.left, roi.right):
        modbus.write_error(ERR_ROI)
        return False, OFF, "OFF", seam_l, seam_r, 0
    lx, ly, lw, lh = roi.left["x"], roi.left["y"], roi.left["w"], roi.left["h"]
    rx, ry, rw, rh = roi.right["x"], roi.right["y"], roi.right["w"], roi.right["h"]

    ok_m, dist_mm = _begin_measure(
        modbus, cmd, use_cfg_dist, cfg_dist_mm,
        min_dist_mm=min_dist_mm, max_dist_mm=max_dist_mm,
    )
    if not ok_m:
        return False, OFF, "OFF", seam_l, seam_r, dist_mm

    if seam_tracker is not None:
        seam_tracker.arm(laser, laser_det)
    else:
        seam_l, seam_r = arm_detection_on_start(seam_l, seam_r, params)
    if calib_autosave:
        cfg["roi_left"] = roi.left.copy()
        cfg["roi_right"] = roi.right.copy()
        cfg["laser"] = laser_det.to_config(l_cfg)
        if seam_tracker is not None:
            cfg["laser_seam"] = seam_tracker.to_config(cfg.get("laser_seam", {}))
        save_config(cfg)
    print(f"[laser] START↑ ROI L=({lx},{ly},{lw},{lh}) R=({rx},{ry},{rw},{rh}) — гистограммы")
    return True, WAIT_L, "WAIT LEFT", seam_l, seam_r, dist_mm


def _after_seam_complete(
    laser_enabled, search_active, roi, fw, fh, roi_w, roi_h, laser_det,
    seam_tracker=None,
):
    """После стыка: сброс ROI/трекера, снова SEARCH и ждём фронт START."""
    if seam_tracker is not None:
        seam_tracker.reset()
    if laser_enabled:
        _clear_laser_rois(roi, fw, fh, roi_w, roi_h)
        if search_active:
            if laser_det is not None:
                laser_det.reset_smooth()
            return SEARCH, "SEARCH — ждём START", 0, 0
        return OFF, "OFF", 0, 0
    return OFF, "OFF", 0, 0


def _abort_measure_to_idle(
    laser_enabled, search_active, roi, fw, fh, roi_w, roi_h, laser_det,
    modbus, seam_tracker=None,
):
    modbus.measure_active(False)
    return _after_seam_complete(
        laser_enabled, search_active, roi, fw, fh, roi_w, roi_h, laser_det,
        seam_tracker=seam_tracker,
    )


def _search_active(manual_search, modbus, cmd, cmd_ok):
    """SEARCH (Modbus бит1 / клавиша L) — только при ERR=0."""
    if modbus.error_code != 0:
        return False
    return bool(manual_search or (cmd_ok and modbus.has_search(cmd)))


def _cmd_start_on(modbus, cmd, cmd_ok):
    """START удерживается ПЛК (бит2), не фронт."""
    return bool(cmd_ok and modbus.has_start(cmd))


def _after_measure_idle(
    laser_enabled, search_active, roi, fw, fh, roi_w, roi_h, laser_det,
    seam_tracker=None,
):
    """Таймаут / сброс ошибки — то же что после стыка."""
    return _after_seam_complete(
        laser_enabled, search_active, roi, fw, fh, roi_w, roi_h, laser_det,
        seam_tracker=seam_tracker,
    )


def _safe_modbus_error(modbus, code):
    try:
        modbus.write_error(code)
    except Exception as exc:
        print(f"[Modbus] write_error({code}) не отправлен: {exc}")


def main():
    cfg = load_config()
    det = cfg["detection"]
    mb = cfg["modbus"]
    l_cfg = cfg.get("laser", {})
    l_main = cfg.get("laser_main", {})
    l_seam_cfg = cfg.get("laser_seam", {})
    cam = cfg["camera"]

    params = {
        "detect_mode": "bg_diff",
        "min_center_line_ratio": det.get("min_center_line_ratio", 0.5),
        "center_line_half": det.get("center_line_half", 2),
        "diff_threshold": det.get("diff_threshold", 12),
        "diff_adaptive_pct": det.get("diff_adaptive_pct", 0),
        "diff_adaptive_cap": det.get("diff_adaptive_cap", 28),
        "diff_blur_k": det.get("diff_blur_k", 3),
        "diff_morph_open": det.get("diff_morph_open", 1),
        "line_hysteresis": det.get("line_hysteresis", 0.1),
        "hit_need_rising": det.get("hit_need_rising", False),
        "bg_init_frames": det.get("bg_init_frames", 8),
        "bg_calm_frames": det.get("bg_calm_frames", 12),
        "bg_freeze_after_seam_frames": det.get("bg_freeze_after_seam_frames", 45),
        "bg_freeze_trigger_ratio": det.get("bg_freeze_trigger_ratio", 0.12),
        "bg_median_n": det.get("bg_median_n", 10),
        "bg_auto_update": det.get("bg_auto_update", True),
        "bg_update_mode": det.get("bg_update_mode", "ema"),
        "bg_ema_alpha": det.get("bg_ema_alpha", 0.015),
        "bg_max_seam_ratio": det.get("bg_max_seam_ratio", 0.06),
        "bg_max_hot_pct": det.get("bg_max_hot_pct", 5.0),
        "bg_max_mean_diff": det.get("bg_max_mean_diff", 8.0),
        "diff_median_k": det.get("diff_median_k", 9),
        "vertical_morph_h": det.get("vertical_morph_h", 5),
        "band_half_ratio": det.get("band_half_ratio", 0.55),
        "max_band_half_px": det.get("max_band_half_px", 24),
        "diff_percentile": det.get("diff_percentile", 88),
        "texture_near_grad": det.get("texture_near_grad", True),
        "texture_grad_span_px": det.get("texture_grad_span_px", 11),
        "v_span_column_half": det.get("v_span_column_half", 2),
        "confirm_frames": det.get("confirm_frames", 2),
        "fast_confirm_ratio": det.get("fast_confirm_ratio", 0.5),
        "fast_confirm_frames": det.get("fast_confirm_frames", 1),
        "hit_hold_frames": det.get("hit_hold_frames", 8),
        "center_line_max_gap": det.get("center_line_max_gap", 10),
        "min_pulse_width_px": det.get("min_pulse_width_px", 2),
        "max_pulse_width_px": det.get("max_pulse_width_px", 28),
        "center_zone_ratio": det.get("center_zone_ratio", 0.35),
        "min_center_overlap_px": det.get("min_center_overlap_px", 2),
        "min_vertical_span_ratio": det.get("min_vertical_span_ratio", 0.25),
        "max_vertical_span_ratio": det.get("max_vertical_span_ratio", 0.88),
    }
    center_ratio = params["center_zone_ratio"]
    min_measure_ms = cfg.get("measurement", {}).get("min_measure_ms", 800)
    blur = det.get("blur", 5)
    cooldown = det.get("cooldown_ms", 200)
    cooldown_after_left = det.get("cooldown_after_left_ms", 0)
    timeout_ms = det.get("measure_timeout_ms", 30000)
    cfg_dist_mm = cfg.get("measurement", {}).get("distance_mm", 1200)
    use_cfg_dist = cfg.get("measurement", {}).get("use_config_distance", True)
    cycle_dist_mm = int(cfg_dist_mm)
    meas = cfg.get("measurement", {})
    min_dist_mm = int(meas.get("min_distance_mm", 10))
    max_dist_mm = int(meas.get("max_distance_mm", 5000))
    max_speed_reg = int(meas.get("max_speed_reg", 65535))
    dt_mode = str(meas.get("dt_mode", "timer")).strip().lower()
    if dt_mode not in ("timer", "fps", "hybrid"):
        dt_mode = "timer"
    cmd_fail_frames = int(mb.get("cmd_fail_frames", 40))
    frame_fail_frames = int(cam.get("frame_fail_frames", 25))
    unstable_error_frames = int(l_main.get("unstable_error_frames", 90))
    dbg = cfg.get("debug", {})
    run_mode = cfg.get("run_mode", {})
    modbus_plc = bool(run_mode.get("modbus_plc", False))
    keyboard_plc = bool(run_mode.get("keyboard_plc", not modbus_plc))
    auto_start = dbg.get("auto_start", False)
    auto_repeat = dbg.get("auto_repeat", False)
    modbus_cmd_trace = dbg.get("modbus_cmd_trace", False)
    last_cmd_trace = -1
    _warned_start_in_error = False
    _warned_search_in_error = False

    log_on = cfg.get("tuning_log", {}).get("print_to_console", True)
    log_ms = cfg.get("tuning_log", {}).get("interval_ms", 250)
    last_log = 0.0

    modbus = ModbusIO(
        mb.get("ip", "127.0.0.1"),
        mb.get("port", 502),
        timeout=mb.get("timeout_sec", 0.3),
        modbus_cfg=mb,
    )
    seam_l, seam_r = None, None

    stream = open_media_source(cam, run_mode)
    if not stream.isOpened():
        raise RuntimeError(f"Камера не открылась: {stream.label}")
    using_video = isinstance(stream, VideoFileStream)
    print(f"Источник: {stream.label}")
    if not using_video:
        print(
            f"[run_mode] КАМЕРА RTSP {cam.get('ip')}:{cam.get('port', 554)} "
            f"{cam.get('width')}x{cam.get('height')} backend={cam.get('rtsp_backend', 'ffmpeg')}"
        )
    dt_fps = _resolve_dt_fps(stream, cam, meas)
    if dt_mode == "fps":
        print(f"[measure] dt_mode=fps  fps={dt_fps:g}  (кадры × 1000 / fps)")
    elif dt_mode == "hybrid":
        print(f"[measure] dt_mode=hybrid  fps={dt_fps:g}  max(timer, кадры)")
    else:
        print("[measure] dt_mode=timer  (perf_counter между LEFT и RIGHT)")
    if modbus_plc and using_video:
        rt = "реальное время" if cam.get("video_realtime", False) else "макс. FPS"
        print(f"[run_mode] VIDEO ({rt}) + Modbus ПЛК — SEARCH/START только с HR cmd (клавиши L/S выкл)")
    elif modbus_plc:
        print("[run_mode] Modbus ПЛК + RTSP — SEARCH/START только с HR cmd")

    # Laser stripes -> auto ROI calibration on START (optional)
    laser_enabled = bool(l_main.get("enabled", False))
    calib_min_gap_ratio = float(l_main.get("min_gap_ratio", 0.08))
    calib_max_gap_ratio = float(l_main.get("max_gap_ratio", 0.92))
    calib_max_jitter_px = int(l_main.get("max_jitter_px", 10))
    calib_autosave_rois = bool(l_main.get("autosave_rois", True))
    auto_repeat_delay_ms = int(l_main.get("auto_repeat_delay_ms", 5000))
    l_roi_w = int(l_cfg.get("roi_w", cfg["roi_left"]["w"]))
    l_roi_h = int(l_cfg.get("roi_h", cfg["roi_left"]["h"]))
    seam_off_y = int(l_cfg.get("seam_offset_y", l_cfg.get("roi_y_offset", 0)))
    seam_off_l_x = int(l_cfg.get("seam_offset_l_x", 0))
    seam_off_r_x = int(l_cfg.get("seam_offset_r_x", 0))

    if laser_enabled:
        _ph = _laser_roi_placeholder(l_roi_w, l_roi_h)
        roi = ROIManager(_ph, _ph.copy())
    else:
        roi = ROIManager(cfg["roi_left"], cfg["roi_right"])

    laser_det = None
    seam_tracker = None
    if laser_enabled:
        laser_det = LaserDetector(l_cfg, int(cam["width"]), int(cam["height"]))
        seam_tracker = LaserSeamTracker(l_seam_cfg)
        lost_hold_frames = int(l_cfg.get("lost_hold_frames", 15))
        lost_error_frames = int(l_main.get("lost_error_frames", 15))
        laser_grace = _LaserGrace(lost_hold_frames, lost_error_frames)
        print(
            f"[laser_main] mode={laser_det.search_mode} "
            f"стык=hist "
            f"{'auto ' if seam_tracker.auto_tune_enabled else ''}"
            f"chg>={seam_tracker.effective_chg_min():.3f} "
            f"warmup={seam_tracker.warmup_frames}+{seam_tracker.settle_frames}fr settle "
            f"hold={lost_hold_frames}fr err={lost_error_frames}fr"
        )
    else:
        laser_grace = _LaserGrace(15, 15)

    tune = KeyboardTuning(
        cfg,
        laser_det=laser_det,
        laser_enabled=laser_enabled,
        seam_tracker=seam_tracker,
        params=params,
        blur=blur,
        cooldown_ms=cooldown,
        l_main=l_main,
        roi_manager=roi,
    )

    state = OFF
    status = "OFF"
    left_t = 0.0
    left_ts_ms = 0.0
    left_frame = 0
    frame_idx = 0
    frame_ts_ms = 0.0
    last_speed = 0.0
    last_evt = 0.0
    left_armed = True
    poll_ms = mb.get("poll_interval_ms", 50)
    last_poll = 0.0
    cmd = 0
    laser_ok_count = 0
    laser_lost_count = 0
    laser_prev_x0 = None
    laser_prev_x1 = None
    manual_search = False
    right_hit_latched = False
    show_laser_mask = False
    prev_state = OFF
    cmd_fail_streak = 0
    frame_fail_streak = 0
    search_unstable_frames = 0
    auto_arm_pending = False

    rec_cfg = cfg.get("recording", {})
    rec_vis_cfg = cfg.get("recording_overlay", {})
    rec_raw_want = bool(rec_cfg.get("enabled", False) or rec_cfg.get("on_startup", False))
    rec_vis_want = bool(
        rec_vis_cfg.get("enabled", False) or rec_vis_cfg.get("on_startup", False)
    )
    recorder_raw = RawVideoRecorder(rec_cfg)
    if rec_cfg.get("on_startup", rec_cfg.get("enabled", False)):
        recorder_raw.enabled = True
    recorder_vis = RawVideoRecorder(rec_vis_cfg)
    if rec_vis_cfg.get("on_startup", rec_vis_cfg.get("enabled", False)):
        recorder_vis.enabled = True
    stream_fps = float(getattr(stream, "fps", 0) or cam.get("fps") or 25)
    recorder_raw.set_stream_fps(stream_fps)
    recorder_vis.set_stream_fps(stream_fps)
    measure_log = MeasureCsvLog(cfg.get("measure_log", {}))


    modbus.clear_plc_outputs()
    _cmd0, _ok0 = modbus.read_cmd_status()
    if not _ok0:
        print(
            f"[Modbus] CMD HR wire={modbus.reg_wire['cmd']} не читается — "
            "slave HR 195–197 / 257–258, unit_id=7, cmd wire=257"
        )
    modbus.arm_cmd_baseline(_cmd0)

    print("=== Pipe vision ===")
    if modbus_plc:
        print(
            f"ПЛК→камера HR{modbus.reg['cmd']} wire={modbus.reg_wire['cmd']}: "
            f"бит1 SEARCH  бит2 START↑  бит0 RESET  бит4 DIST_OK"
        )
        if laser_enabled:
            print(
                "[laser] START удерживается (бит2=1) → после стыка авто START "
                f"через {auto_repeat_delay_ms}ms (без нового фронта)"
            )
        print(
            f"STATUS HR{modbus.reg['status_out']}: "
            "б0 LEFT  б1 RIGHT  б2 JOINT  б3 DONE  б5 ACTIVE  б6 READY  б7 ERR  б8 ALIVE"
        )
        print(
            f"SPEED HR{modbus.reg['speed']}  ERROR HR{modbus.reg['error_code']}"
        )
        print(f"Дистанция: HR{modbus.reg['dist_mm']} мм + DIST_OK (бит4 в CMD)")
    if laser_enabled:
        if keyboard_plc:
            print("Лазер: SEARCH=1 (бит1 / L) → поиск лазера, настройка (t)")
            print("       START↑ 0→1 (бит2 / S) → фиксация гистограмм, проезд")
        else:
            print("Лазер: SEARCH=бит1, START↑=бит2 (только Modbus)")
        print("       SEARCH/START только при ERR=0 → сначала RESET (бит0)")
        print("       SEARCH=0 → поиск выкл, START игнор")
        print("       После стыка → снова ждём START↑ при SEARCH=1")
        print("       Нет лазера при START → ERR 4")
    else:
        print("Состояние: OFF — анализ bg только после START с ПЛК")
    print("Детекция стыка: bg_diff  +/- cline  [/] diff  B=фон")
    print(
        f"Старт: diff={params['diff_threshold']} cline>={params['min_center_line_ratio']:.0%} "
        f"init={params['bg_init_frames']}fr cf={params['confirm_frames']}"
    )
    print("Tuning: t=панель  P=save  см. NASTROYKI.txt")
    print("Ошибки Modbus: см. ERRORS.txt")
    tune.print_help()
    print(
        f"Modbus CMD: HR{modbus.reg['cmd']} — "
        f"RESET=бит0 SEARCH=бит1 START=бит2 DIST_OK=бит4"
    )
    if use_cfg_dist:
        print(f"Дистанция: из config ({cfg_dist_mm} mm)")
    else:
        print(
            f"Дистанция: Modbus HR{modbus.reg['dist_mm']} + бит DIST_OK в CMD "
            f"(config {cfg_dist_mm} mm — только запас)"
        )
    if laser_enabled and not mb.get("errors_enabled", True):
        print("Внимание: modbus.errors_enabled=false — ERR 4 (нет лазера) не уйдёт на ПЛК")
    if dt_mode == "fps":
        print(f"Скорость: dt_mode=fps, fps={dt_fps:g} (measurement.dt_fps / camera.fps)")
    elif dt_mode == "hybrid":
        print(f"Скорость: dt_mode=hybrid, fps={dt_fps:g} (max таймер и кадры)")
    else:
        print("Скорость: dt_mode=timer (perf_counter LEFT→RIGHT)")

    if recorder_raw.enabled or recorder_vis.enabled:
        parts = []
        if recorder_raw.enabled:
            parts.append("raw")
        if recorder_vis.enabled:
            parts.append("overlay")
        print(
            "Запись видео: {}  R=вкл/выкл  папка recordings/".format("+".join(parts))
        )
    cv2.namedWindow("Pipe Vision", cv2.WINDOW_NORMAL)
    tune.attach_window("Pipe Vision")
    modbus.signal_ready()
    if modbus.async_enabled:
        modbus.start_async_poll(poll_ms)

    pending_frames = None

    try:
        while True:
            l_roi_w = tune.roi_w
            l_roi_h = tune.roi_h
            seam_off_l_x = tune.off_l_x
            seam_off_r_x = tune.off_r_x
            seam_off_y = tune.off_y
            center_ratio = params["center_zone_ratio"]
            calib_need_frames = int(l_main.get("need_found_frames", 5))
            lost_error_frames = int(l_main.get("lost_error_frames", 15))
            calib_min_gap_ratio = float(
                laser_det.min_gap_ratio if laser_det else l_main.get("min_gap_ratio", 0.08)
            )
            calib_max_gap_ratio = float(
                laser_det.max_gap_ratio if laser_det else l_main.get("max_gap_ratio", 0.92)
            )

            if pending_frames is None:
                preview_latest = bool(cam.get("rtsp_preview_latest", True))
                if state in (WAIT_L, WAIT_R) and hasattr(stream, "read_batch"):
                    drain_n = int(cam.get("rtsp_measure_batch", 4))
                    frame_batch = stream.read_batch(drain_n)
                elif preview_latest and hasattr(stream, "read_latest"):
                    ok, one = stream.read_latest()
                    frame_batch = [one] if ok and one is not None else []
                elif hasattr(stream, "read_batch"):
                    frame_batch = stream.read_batch(1)
                else:
                    ok, one = stream.read()
                    frame_batch = [one] if ok and one is not None else []
                if not frame_batch:
                    frame_fail_streak += 1
                    if frame_fail_streak >= frame_fail_frames and modbus.error_code == 0:
                        _safe_modbus_error(modbus, ERR_CAMERA)
                        print(f"[camera] нет кадра {frame_fail_streak}fr → ERR 7")
                    time.sleep(0.03)
                    _parse_key(1)
                    continue
                frame_fail_streak = 0
                pending_frames = list(frame_batch)

            if not pending_frames:
                pending_frames = None
                continue

            frame = pending_frames.pop(0)
            frame_idx += 1
            frame_ts_ms = time.perf_counter() * 1000.0

            raw_frame = frame.copy()
            fh, fw = frame.shape[:2]
            roi.clamp(fw, fh)
            (lx, ly, lw, lh), (rx, ry, rw, rh) = roi.get_rects()

            l_roi, r_roi = blur_rois_gray(
                frame,
                [(lx, ly, lw, lh), (rx, ry, rw, rh)],
                blur,
            )
            idle_res = SeamResult(phase="OFF", debug="OFF")
            laser_cur = None
            eff_laser = None
            laser_present = False
            laser_err = False
            if laser_enabled and laser_det is not None and state in (SEARCH, WAIT_L, WAIT_R):
                if state == SEARCH:
                    laser_cur = laser_det.detect(frame, fw, fh)
                else:
                    active = "left" if state == WAIT_L else "right"
                    laser_cur = laser_det.read_seam_peaks(
                        frame, fw, fh,
                        hint_l=seam_tracker.baseline_l,
                        hint_r=seam_tracker.baseline_r,
                        active=active,
                    )

            if laser_enabled and state in (SEARCH, WAIT_L, WAIT_R):
                allow_laser_err = state == SEARCH
                eff_laser, laser_present, laser_err = laser_grace.update(
                    laser_cur, allow_error=allow_laser_err
                )
            else:
                laser_grace.reset()
                eff_laser = laser_cur if (laser_cur and laser_cur.found) else None
                laser_present = eff_laser is not None
                laser_err = False

            if laser_enabled and state == WAIT_L and eff_laser is not None:
                _, l_res = seam_tracker.check_left(eff_laser, laser_det)
                r_res = idle_res
            elif laser_enabled and state == WAIT_R and eff_laser is not None:
                l_res = idle_res
                _, r_res = seam_tracker.check_right(eff_laser, laser_det)
            elif state == WAIT_L:
                l_res, seam_l = detect_roi(l_roi, 1, params, seam_l)
                r_res = idle_res
            elif state == WAIT_R:
                l_res = idle_res
                r_res, seam_r = detect_roi(r_roi, 1, params, seam_r)
            else:
                l_res = r_res = idle_res

            now = time.time() * 1000.0
            cmd_ok = True
            search_on = _search_active(manual_search, modbus, cmd, cmd_ok)

            laser_preview = eff_laser if eff_laser is not None else laser_cur
            modbus.sync_active(
                state in (WAIT_L, WAIT_R) and laser_present
            )
            if (
                laser_enabled
                and laser_det is not None
                and state == SEARCH
                and eff_laser is not None
                and modbus.error_code == 0
            ):
                stable, x0_i, x1_i = _update_laser_stable(
                    eff_laser, laser_det,
                    laser_prev_x0, laser_prev_x1,
                    calib_max_jitter_px, calib_min_gap_ratio, calib_max_gap_ratio,
                )
                det_hold = bool(
                    laser_cur is not None and laser_cur.found and not laser_cur.peaks_live
                )
                if stable:
                    laser_ok_count += 1
                    search_unstable_frames = 0
                    laser_prev_x0, laser_prev_x1 = x0_i, x1_i
                    status = f"SEARCH OK {laser_ok_count}/{calib_need_frames}"
                elif laser_grace.holding or det_hold:
                    status = (
                        f"SEARCH hold {laser_grace.grace_left}/{laser_grace.hold_frames}"
                        if laser_grace.holding
                        else "SEARCH hold det"
                    )
                else:
                    laser_ok_count = 0
                    laser_prev_x0, laser_prev_x1 = None, None
                    search_unstable_frames += 1
                    status = "SEARCH NO LASER"
                    if search_unstable_frames >= unstable_error_frames:
                        modbus.write_error(ERR_LASER_UNSTABLE)
                        status = error_status_text(ERR_LASER_UNSTABLE)
                        state = OFF
                        search_unstable_frames = 0
                        laser_grace.reset()
                if laser_err:
                    modbus.write_error(ERR_NO_LASER)
                    status = error_status_text(ERR_NO_LASER)
                    state = OFF
                    laser_ok_count = 0
                    laser_prev_x0 = laser_prev_x1 = None
                    laser_grace.reset()
            elif (
                laser_enabled
                and state in (WAIT_L, WAIT_R)
                and not laser_present
                and modbus.error_code == 0
            ):
                status = (
                    f"{'WAIT LEFT' if state == WAIT_L else 'WAIT RIGHT'}"
                    f" NO LASER {laser_grace.miss_frames}/{laser_grace.error_frames}"
                )

            if log_on and now - last_log >= log_ms:
                last_log = now
                print(
                    f"[L] {l_res.phase} {l_res.debug} hit={int(l_res.hit)} vis={int(l_res.visible)} | "
                    f"[R] {r_res.phase} {r_res.debug} hit={int(r_res.hit)} vis={int(r_res.visible)} | "
                    f"{state}"
                )

            if now - last_poll >= poll_ms:
                last_poll = now
                cmd, cmd_ok = modbus.read_cmd_status()
                if not cmd_ok:
                    cmd = 0
                    cmd_fail_streak += 1
                    if cmd_fail_streak >= cmd_fail_frames and modbus.error_code == 0:
                        modbus.write_error(ERR_MODBUS)
                        print(f"[Modbus] CMD не читается {cmd_fail_streak}fr → ERR 6")
                else:
                    cmd_fail_streak = 0
                if modbus_cmd_trace and cmd_ok and cmd != last_cmd_trace:
                    last_cmd_trace = cmd
                    err_part = f" ERR={modbus.error_code}" if modbus.error_code else ""
                    print(
                        f"[Modbus] HR{modbus.reg['cmd']} CMD {modbus.format_cmd(cmd)}{err_part}"
                    )
                search_on = _search_active(manual_search, modbus, cmd, cmd_ok)

                did_reset = modbus.reset_errors(cmd)
                if did_reset:
                    auto_arm_pending = False
                    left_armed = True
                    seam_l = seam_r = None
                    modbus.reset_cycle()
                    _warned_start_in_error = False
                    _warned_search_in_error = False
                    if laser_enabled and search_on:
                        state = SEARCH
                        status = "SEARCH LASER"
                        laser_ok_count = 0
                        laser_lost_count = 0
                        laser_prev_x0 = laser_prev_x1 = None
                        laser_det.reset_smooth()
                    else:
                        state = OFF
                        status = "OFF"
                elif modbus.error_code != 0:
                    if cmd_ok and modbus.has_search(cmd) and not _warned_search_in_error:
                        _warned_search_in_error = True
                        print(
                            f"[Modbus] SEARCH при ERR {modbus.error_code} — "
                            "сначала RESET (CMD=1)"
                        )
                    if state not in (WAIT_L, WAIT_R):
                        if state == SEARCH:
                            laser_ok_count = 0
                            laser_lost_count = 0
                            laser_prev_x0 = laser_prev_x1 = None
                        state = OFF
                        status = error_status_text(modbus.error_code)
                    modbus.measure_active(False)
                    if modbus.has_start(cmd) and not _warned_start_in_error:
                        _warned_start_in_error = True
                        print(
                            f"[Modbus] START при ERR {modbus.error_code} — "
                            "сначала RESET (CMD=1)"
                        )

                if laser_enabled and state in (OFF, SEARCH, WAIT_L, WAIT_R):
                    if search_on and modbus.error_code == 0:
                        if state == OFF:
                            state = SEARCH
                            laser_ok_count = 0
                            laser_lost_count = 0
                            laser_prev_x0 = laser_prev_x1 = None
                            laser_det.reset_smooth()
                            status = "SEARCH LASER"
                    elif not search_on:
                        auto_arm_pending = False
                        if state in (WAIT_L, WAIT_R):
                            state, status, laser_ok_count, laser_lost_count = (
                                _abort_measure_to_idle(
                                    laser_enabled, False, roi, fw, fh,
                                    l_roi_w, l_roi_h, laser_det, modbus,
                                    seam_tracker=seam_tracker,
                                )
                            )
                            laser_prev_x0 = laser_prev_x1 = None
                            seam_l = seam_r = None
                            print("[laser] SEARCH=0 — измерение прервано")
                        elif state == SEARCH:
                            state = OFF
                            status = "OFF"
                            laser_ok_count = 0
                            laser_lost_count = 0
                            laser_prev_x0 = laser_prev_x1 = None

                modbus.sync_ready()

                if not modbus.async_enabled:
                    modbus.tick_status(now)

                if auto_arm_pending and cmd_ok and not modbus.has_start(cmd):
                    auto_arm_pending = False

                start_edge = (
                    modbus.error_code == 0
                    and modbus.start_measure_edge(cmd)
                )
                if start_edge and state in (WAIT_L, WAIT_R):
                    modbus.write_error(ERR_BAD_START)
                    status = error_status_text(ERR_BAD_START)
                    print("[Modbus] START во время проезда → ERR 10")
                elif (
                    start_edge
                    and laser_enabled
                    and state == OFF
                ):
                    modbus.write_error(ERR_BAD_START)
                    status = error_status_text(ERR_BAD_START)
                    print("[Modbus] START без SEARCH → ERR 10")
                elif (
                    modbus.error_code == 0
                    and state == SEARCH
                    and search_on
                    and start_edge
                ):
                    if laser_enabled and laser_det is not None:
                        if laser_present:
                            ok_m, new_st, new_status, seam_l, seam_r, cycle_dist_mm = (
                                _start_from_lasers(
                                    roi, laser_preview, laser_det, fw, fh,
                                    l_roi_w, l_roi_h,
                                    seam_off_l_x, seam_off_r_x, seam_off_y,
                                    modbus, cmd, use_cfg_dist, cfg_dist_mm,
                                    seam_l, seam_r, params,
                                    calib_autosave_rois, cfg, l_cfg,
                                    seam_tracker=seam_tracker,
                                    min_dist_mm=min_dist_mm,
                                    max_dist_mm=max_dist_mm,
                                )
                            )
                            state = new_st
                            status = new_status
                            if ok_m:
                                left_armed = True
                                last_evt = now
                                right_hit_latched = False
                                laser_ok_count = 0
                                laser_lost_count = 0
                                laser_prev_x0 = laser_prev_x1 = None
                                print("[laser] START↑ — гистограммы зафиксированы")
                            else:
                                _clear_laser_rois(
                                    roi, fw, fh, l_roi_w, l_roi_h
                                )
                        else:
                            modbus.write_error(ERR_NO_LASER)
                            status = error_status_text(ERR_NO_LASER)
                            laser_grace.reset()
                            print("[laser] START↑ без лазера → ERR 4")
                    elif not laser_enabled:
                        ok_m, cycle_dist_mm = _begin_measure(
                            modbus, cmd, use_cfg_dist, cfg_dist_mm,
                            min_dist_mm=min_dist_mm, max_dist_mm=max_dist_mm,
                        )
                        if ok_m:
                            seam_l, seam_r = arm_detection_on_start(
                                seam_l, seam_r, params
                            )
                            state = WAIT_L
                            status = "WAIT LEFT"
                            left_armed = True
                            last_evt = now
                            print(
                                f"[Measure] START↑ dist={cycle_dist_mm}mm "
                                f"({'config' if use_cfg_dist else 'Modbus'})"
                            )
                elif start_edge and not laser_enabled and state == OFF:
                    ok_m, cycle_dist_mm = _begin_measure(
                        modbus, cmd, use_cfg_dist, cfg_dist_mm,
                        min_dist_mm=min_dist_mm, max_dist_mm=max_dist_mm,
                    )
                    if ok_m:
                        seam_l, seam_r = arm_detection_on_start(seam_l, seam_r, params)
                        state = WAIT_L
                        status = "WAIT LEFT"
                        left_armed = True
                        last_evt = now
                        print(
                            f"[Measure] START↑ dist={cycle_dist_mm}mm "
                            f"({'config' if use_cfg_dist else 'Modbus'})"
                        )

                if (
                    auto_arm_pending
                    and _cmd_start_on(modbus, cmd, cmd_ok)
                    and modbus.error_code == 0
                    and state == SEARCH
                    and search_on
                    and laser_enabled
                    and laser_det is not None
                    and laser_present
                    and laser_ok_count >= calib_need_frames
                    and (now - last_evt) >= auto_repeat_delay_ms
                ):
                    stable_n = laser_ok_count
                    ok_m, new_st, new_status, seam_l, seam_r, cycle_dist_mm = (
                        _start_from_lasers(
                            roi, laser_preview, laser_det, fw, fh,
                            l_roi_w, l_roi_h,
                            seam_off_l_x, seam_off_r_x, seam_off_y,
                            modbus, cmd, use_cfg_dist, cfg_dist_mm,
                            seam_l, seam_r, params,
                            calib_autosave_rois, cfg, l_cfg,
                            seam_tracker=seam_tracker,
                            min_dist_mm=min_dist_mm,
                            max_dist_mm=max_dist_mm,
                        )
                    )
                    auto_arm_pending = False
                    state = new_st
                    status = new_status
                    if ok_m:
                        left_armed = True
                        last_evt = now
                        right_hit_latched = False
                        laser_ok_count = 0
                        laser_lost_count = 0
                        laser_prev_x0 = laser_prev_x1 = None
                        print(
                            "[laser] auto-repeat START (START=1) после стыка "
                            f"({stable_n}/{calib_need_frames} стаб.)"
                        )
                    else:
                        _clear_laser_rois(roi, fw, fh, l_roi_w, l_roi_h)

                modbus.note_cmd(cmd)
            else:
                search_on = _search_active(manual_search, modbus, cmd, cmd_ok)

            if state == WAIT_R and left_t > 0 and (now - left_t) > timeout_ms:
                modbus.write_error(ERR_TIMEOUT)
                modbus.measure_active(False)
                seam_l = seam_r = None
                state, status, laser_ok_count, laser_lost_count = _after_measure_idle(
                    laser_enabled, search_on, roi, fw, fh, l_roi_w, l_roi_h, laser_det,
                    seam_tracker=seam_tracker,
                )
                if modbus.errors_enabled and modbus.error_code:
                    status = error_status_text(modbus.error_code)
                laser_prev_x0 = laser_prev_x1 = None

            if state in (WAIT_L, WAIT_R) and modbus.error_code == 0:
                measure_tick = laser_enabled or (now - last_evt) >= cooldown
                if state == WAIT_L and measure_tick:
                    if (
                        l_res.hit
                        and l_res.phase == "DROP"
                        and left_armed
                        and seam_tracker is not None
                        and seam_tracker.watch_ready_l
                    ):
                        left_t = now
                        left_ts_ms = frame_ts_ms
                        left_frame = frame_idx
                        state = WAIT_R
                        status = "TIMER ON"
                        left_armed = False
                        last_evt = now
                        right_hit_latched = False
                        if laser_enabled and laser_cur is not None:
                            seam_tracker.rearm_right(laser_cur, laser_det)
                        modbus.pulse_left(now)
                        if not laser_enabled:
                            seam_l = reset_track_state(seam_l)
                        tag = "DROP" if laser_enabled else "CENTER"
                        print(f">>> LEFT {tag} x={l_res.x_peak:.0f} {l_res.debug}")

                elif state == WAIT_R and (
                    laser_enabled or (now - left_t) >= cooldown_after_left
                ):
                    if (
                        r_res.hit
                        and r_res.phase == "DROP"
                        and seam_tracker is not None
                        and seam_tracker.watch_ready_r
                    ):
                        right_hit_latched = True
                    elif r_res.hit and not laser_enabled:
                        right_hit_latched = True
                    if right_hit_latched:
                        dt_ms = _measure_dt_ms(
                            dt_mode, left_ts_ms, frame_ts_ms,
                            left_frame, frame_idx, dt_fps,
                        )
                        n_fr = max(0, frame_idx - left_frame)
                        if dt_ms < min_measure_ms:
                            status = f"TIMER ON ({int(dt_ms)}/{min_measure_ms}ms)"
                        elif dt_ms > 0:
                            dist = int(cycle_dist_mm)
                            last_speed = modbus.calc_speed_raw(
                                dist, int(dt_ms), modbus.speed_scale
                            )
                            if last_speed > max_speed_reg:
                                modbus.write_error(ERR_BAD_MEASURE)
                                modbus.measure_active(False)
                                seam_l = seam_r = None
                                state, status, laser_ok_count, laser_lost_count = (
                                    _after_measure_idle(
                                        laser_enabled, search_on, roi, fw, fh,
                                        l_roi_w, l_roi_h, laser_det,
                                        seam_tracker=seam_tracker,
                                    )
                                )
                                laser_prev_x0 = laser_prev_x1 = None
                                print(
                                    f"[Measure] speed reg={last_speed} > {max_speed_reg} → ERR 9"
                                )
                            else:
                                modbus.pulse_right(now)
                                modbus.write_speed(dist, int(dt_ms))
                                modbus.joint_found()
                                v_mm_s = modbus.format_speed_mm_s(
                                    last_speed, modbus.speed_scale
                                )
                                v_mm_s_float = last_speed / modbus.speed_scale
                                v_m_min = v_mm_s_float * 60.0 / 1000.0
                                measure_log.append(
                                    dist, dt_ms, last_speed, v_mm_s,
                                    int(l_res.x_peak) if l_res.x_peak else 0,
                                    int(r_res.x_peak) if r_res.x_peak else 0,
                                    "OK",
                                )
                                dt_tag = (
                                    f"fps {dt_fps:g} n={n_fr}"
                                    if dt_mode == "fps"
                                    else "hybrid" if dt_mode == "hybrid" else "timer"
                                )
                                print(
                                    f">>> RIGHT {'DROP' if laser_enabled else 'CENTER'} "
                                    f"x={r_res.x_peak:.0f} {r_res.debug} "
                                    f"dist={dist}mm dt={int(dt_ms)}ms ({dt_tag}) "
                                    f"v={v_mm_s} mm/s ({v_m_min:.2f} m/min reg={last_speed})"
                                )
                                last_evt = now
                                left_armed = True
                                right_hit_latched = False
                                seam_l = seam_r = None
                                if auto_repeat and not laser_enabled:
                                    ok_m, cycle_dist_mm = _begin_measure(
                                        modbus, cmd, use_cfg_dist, cfg_dist_mm,
                                        min_dist_mm=min_dist_mm,
                                        max_dist_mm=max_dist_mm,
                                    )
                                    if ok_m:
                                        seam_l, seam_r = arm_detection_on_start(
                                            seam_l, seam_r, params
                                        )
                                        state = WAIT_L
                                        status = "WAIT LEFT"
                                        print("[Measure] auto_repeat (фон bg — заново)")
                                    else:
                                        state = OFF
                                        status = "OFF"
                                elif laser_enabled:
                                    state, status, laser_ok_count, laser_lost_count = (
                                        _after_seam_complete(
                                            laser_enabled, search_on, roi, fw, fh,
                                            l_roi_w, l_roi_h, laser_det,
                                            seam_tracker=seam_tracker,
                                        )
                                    )
                                    laser_prev_x0 = laser_prev_x1 = None
                                    if search_on and _cmd_start_on(modbus, cmd, cmd_ok):
                                        auto_arm_pending = True
                                        print(
                                            "[laser] стык OK → SEARCH, "
                                            "START=1 — авто START когда лазер стабилен"
                                        )
                                    else:
                                        print(
                                            "[laser] стык OK → SEARCH, ждём START↑ (бит2)"
                                        )
                                else:
                                    state = OFF
                                    status = "OFF"

            if state == WAIT_L and prev_state != WAIT_L:
                right_hit_latched = False

            laser_ui_mode = (
                laser_enabled and laser_det is not None
                and state in (SEARCH, WAIT_L, WAIT_R)
            )
            tune.draw_drag_preview(frame)
            tune_hud = tune.hud_lines()
            if laser_ui_mode and laser_cur is not None:
                hud_extra = [status]
                if state == SEARCH:
                    if auto_arm_pending:
                        remain_s = max(
                            0.0, (auto_repeat_delay_ms - (now - last_evt)) / 1000.0
                        )
                        if remain_s > 0.05:
                            hud_extra.append(
                                f"SEARCH START=1 через {remain_s:.1f}s "
                                f"({laser_ok_count}/{calib_need_frames})"
                            )
                        else:
                            hud_extra.append(
                                f"SEARCH START=1 авто {laser_ok_count}/{calib_need_frames}"
                            )
                    else:
                        hud_extra.append("SEARCH=1  START↑=arm  L=search  v=mask")
                elif seam_tracker is not None:
                    hud_extra.append(seam_tracker.hud_line())
                    if state == WAIT_L:
                        hud_extra.append(f"L {l_res.debug}")
                    elif state == WAIT_R:
                        hud_extra.append(f"R {r_res.debug}")
                hud_extra.extend(tune_hud)
                draw_laser_overlay(
                    frame, laser_det, laser_cur, fw, fh,
                    l_roi_w, l_roi_h, seam_off_l_x, seam_off_r_x, seam_off_y,
                    active_calib=(state == SEARCH),
                    show_mask=show_laser_mask and state == SEARCH,
                    draw_seam_rois=state == SEARCH and laser_present,
                    hud_extra=hud_extra,
                    seam_tracker=seam_tracker,
                )
                if state in (WAIT_L, WAIT_R):
                    for r, col, lab in (
                        (roi.left, (0, 255, 0), "L"),
                        (roi.right, (0, 180, 255), "R"),
                    ):
                        cv2.rectangle(
                            frame,
                            (r["x"], r["y"]),
                            (r["x"] + r["w"] - 1, r["y"] + r["h"] - 1),
                            col, 2, cv2.LINE_AA,
                        )
                        cv2.putText(
                            frame, lab, (r["x"], max(12, r["y"] - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA,
                        )
            elif state in (WAIT_L, WAIT_R):
                draw_overlay(
                    frame, l_res, r_res,
                    (lx, ly, lw, lh), (rx, ry, rw, rh),
                    center_ratio, status, last_speed,
                    modbus.error_code if modbus.error_code else 0,
                    params["diff_threshold"],
                    params["min_center_line_ratio"],
                    0,
                    modbus.status, tune.active, roi.active,
                    modbus.speed_scale,
                )
                for i, line in enumerate(tune_hud):
                    cv2.putText(
                        frame, line, (10, fh - 80 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA,
                    )
            elif laser_enabled and state == OFF:
                cv2.putText(
                    frame, status, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA,
                )
                cv2.putText(
                    frame, "L=SEARCH (Modbus bit1)", (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1, cv2.LINE_AA,
                )
            elif not laser_enabled:
                draw_overlay(
                    frame, l_res, r_res,
                    (lx, ly, lw, lh), (rx, ry, rw, rh),
                    center_ratio, status, last_speed,
                    modbus.error_code if modbus.error_code else 0,
                    params["diff_threshold"],
                    params["min_center_line_ratio"],
                    0,
                    modbus.status, tune.active, roi.active,
                    modbus.speed_scale,
                )
                for i, line in enumerate(tune_hud):
                    cv2.putText(
                        frame, line, (10, fh - 80 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA,
                    )

            recorder_raw.write(raw_frame)
            recorder_vis.write(frame)

            cv2.imshow("Pipe Vision", frame)
            key, shifted = _parse_key(1)
            if key is None:
                continue
            if tune.handle_key(key, shifted):
                continue
            if key == 27:
                break
            if key in (ord("r"), ord("R")) and not tune.active:
                if recorder_raw.active or recorder_vis.active:
                    if recorder_raw.active:
                        recorder_raw.stop()
                    if recorder_vis.active:
                        recorder_vis.stop()
                    print("[record] OFF")
                else:
                    if rec_raw_want:
                        recorder_raw.enabled = True
                        recorder_raw.start(fw, fh)
                    if rec_vis_want:
                        recorder_vis.enabled = True
                        recorder_vis.start(fw, fh)
                    print(
                        "[record] ON raw={} overlay={}".format(
                            rec_raw_want, rec_vis_want
                        )
                    )
            if (
                keyboard_plc
                and laser_enabled
                and key in (ord("l"), ord("L"))
                and not tune.blocks_search_toggle(key)
            ):
                manual_search = not manual_search
                print(f"[laser] SEARCH {'ON' if manual_search else 'OFF'} (клавиша L)")
                if manual_search and state == OFF and modbus.error_code == 0:
                    state = SEARCH
                    status = "SEARCH LASER"
                    laser_ok_count = 0
                    laser_lost_count = 0
                    laser_prev_x0 = laser_prev_x1 = None
                    laser_det.reset_smooth()
                elif not manual_search and state == SEARCH:
                    state = OFF
                    status = "OFF"
                    laser_ok_count = 0
                    laser_lost_count = 0
                    laser_prev_x0 = laser_prev_x1 = None
                elif not manual_search and state in (WAIT_L, WAIT_R):
                    state, status, laser_ok_count, laser_lost_count = (
                        _abort_measure_to_idle(
                            laser_enabled, False, roi, fw, fh,
                            l_roi_w, l_roi_h, laser_det, modbus,
                            seam_tracker=seam_tracker,
                        )
                    )
                    laser_prev_x0 = laser_prev_x1 = None
                    seam_l = seam_r = None
            elif (
                keyboard_plc
                and key in (ord("s"), ord("S"))
                and modbus.error_code == 0
                and not tune.blocks_start()
            ):
                if laser_enabled and state == SEARCH and _search_active(
                    manual_search, modbus, cmd, True
                ):
                    if laser_preview is not None and laser_preview.found:
                        ok_m, new_st, new_status, seam_l, seam_r, cycle_dist_mm = (
                            _start_from_lasers(
                                roi, laser_preview, laser_det, fw, fh,
                                l_roi_w, l_roi_h,
                                seam_off_l_x, seam_off_r_x, seam_off_y,
                                modbus, cmd, use_cfg_dist, cfg_dist_mm,
                                seam_l, seam_r, params,
                                calib_autosave_rois, cfg, l_cfg,
                                seam_tracker=seam_tracker,
                                min_dist_mm=min_dist_mm,
                                max_dist_mm=max_dist_mm,
                            )
                        )
                        state = new_st
                        status = new_status
                        if ok_m:
                            left_armed = True
                            last_evt = time.time() * 1000.0
                            laser_ok_count = 0
                            laser_lost_count = 0
                            laser_prev_x0 = laser_prev_x1 = None
                            print("[laser] START↑ (клавиша S) — гистограммы зафиксированы")
                        else:
                            _clear_laser_rois(roi, fw, fh, l_roi_w, l_roi_h)
                    else:
                        modbus.write_error(ERR_NO_LASER)
                        status = error_status_text(ERR_NO_LASER)
                        laser_grace.reset()
                        print("[laser] START без лазера → ERR 4")
                elif not laser_enabled and state == OFF:
                    ok_m, cycle_dist_mm = _begin_measure(
                        modbus, cmd, use_cfg_dist, cfg_dist_mm, manual=True,
                        min_dist_mm=min_dist_mm, max_dist_mm=max_dist_mm,
                    )
                    if ok_m:
                        seam_l, seam_r = arm_detection_on_start(seam_l, seam_r, params)
                        state = WAIT_L
                        status = "WAIT LEFT"
                        left_armed = True
                        last_evt = time.time() * 1000.0
                        print("[Measure] START (key S, фон bg — заново)")
            if key == ord("v") and laser_ui_mode and not (
                tune.active and tune.sub_mode == "laser_misc"
            ):
                show_laser_mask = not show_laser_mask
            elif key in (ord("b"), ord("B")):
                if roi.active == "right":
                    seam_r = capture_bg_ref(seam_r, r_roi)
                else:
                    seam_l = capture_bg_ref(seam_l, l_roi)
                print(f"BG фон записан ({roi.active} ROI)")

            if state == OFF and prev_state != OFF:
                laser_grace.reset()
            prev_state = state

            if pending_frames and state in (WAIT_L, WAIT_R):
                continue

    except Exception as e:
        print(f"[fatal] {e}")
    finally:
        recorder_raw.close()
        recorder_vis.close()
        stream.close()
        cv2.destroyAllWindows()
        modbus.close()
        print("Exit.")


if __name__ == "__main__":
    main()
