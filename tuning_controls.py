"""
Клавиатурная настройка всех параметров анализа (лазер + стык + ROI).
"""
from typing import Optional

import cv2

from config_utils import save_config


class KeyboardTuning:
    SUB_MODES = ("laser_search", "laser_roi", "laser_seam", "laser_misc")

    def __init__(
        self,
        cfg: dict,
        *,
        laser_det=None,
        laser_enabled: bool = True,
        seam_tracker=None,
        params: Optional[dict] = None,
        blur: int = 3,
        cooldown_ms: int = 200,
        l_main: Optional[dict] = None,
        roi_manager=None,
    ):
        self.cfg = cfg
        self.laser_det = laser_det
        self.laser_enabled = laser_enabled
        self.seam_tracker = seam_tracker
        self.params = params or {}
        self.blur = int(blur)
        self.cooldown_ms = int(cooldown_ms)
        self.l_main = l_main if l_main is not None else {}
        self.roi = roi_manager

        l_cfg = cfg.get("laser", {})
        self.roi_w = int(l_cfg.get("roi_w", 60))
        self.roi_h = int(l_cfg.get("roi_h", 270))
        self.off_l_x = int(l_cfg.get("seam_offset_l_x", 0))
        self.off_r_x = int(l_cfg.get("seam_offset_r_x", 0))
        self.off_y = int(l_cfg.get("seam_offset_y", 0))

        self.active = False
        self.sub_mode = None
        self.drag = None
        self._step = 5
        self._step_big = 10

    def attach_window(self, window_name: str):
        cv2.setMouseCallback(window_name, self._on_mouse)

    def blocks_start(self) -> bool:
        """S не должен стартовать измерение пока крутим параметры."""
        if not self.active:
            return False
        if self.sub_mode in self.SUB_MODES:
            return True
        return self.laser_enabled and self.laser_det is not None

    def blocks_search_toggle(self, key: int) -> bool:
        if key not in (ord("l"), ord("L")):
            return False
        return self.active and self.sub_mode == "laser_seam"

    def toggle(self):
        self.active = not self.active
        if not self.active:
            self.sub_mode = None
            self.drag = None
        return self.active

    def draw_drag_preview(self, frame):
        if self.drag and self.sub_mode == "laser_search":
            xa, xb = sorted((self.drag[0], self.drag[2]))
            ya, yb = sorted((self.drag[1], self.drag[3]))
            cv2.rectangle(frame, (xa, ya), (xb, yb), (0, 220, 255), 1)

    def hud_lines(self) -> list[str]:
        if not self.active:
            return []
        lines = ["TUNE ON  P=save  t=off"]
        if self.sub_mode == "laser_search":
            lines.append(f"MODE search ROI {self._search_rect()}")
            lines.append("WASD move  JL/IW width  IK height  drag")
        elif self.sub_mode == "laser_roi":
            lines.append(f"MODE seam ROI  W={self.roi_w} H={self.roi_h}")
            lines.append("A/D W  W/S H  (Shift x2)")
        elif self.sub_mode == "laser_seam":
            lines.append(
                f"MODE offset Lx={self.off_l_x} Rx={self.off_r_x} Y={self.off_y}"
            )
            lines.append("WASD L  JL R  IK Y")
        elif self.sub_mode == "laser_misc":
            det = self.laser_det
            if det:
                lines.append(
                    f"fill={det.min_col_fill:.2f} "
                    f"gap={det.min_gap_ratio:.2f}-{det.max_gap_ratio:.2f} "
                    f"blur={det.blur} morph={det.morph_k}"
                )
            lines.append(
                f"stable={self.l_main.get('need_found_frames', 5)} "
                f"lost_err={self.l_main.get('lost_error_frames', 15)}"
            )
            lines.append("U/J fill  N/M gap  Z/X blur  C/c morph  [/] stable")
            if self.seam_tracker:
                st = self.seam_tracker
                at_on = st.auto_tune_enabled
                lines.append(
                    f"chg>={st.effective_chg_min():.3f} "
                    f"{'auto ' if at_on else ''}"
                    f"spike>={st.change_spike_min:.3f} "
                    f"warmup={st.warmup_frames} cf={st.confirm_frames}"
                )
                lines.append("Q/q drop%  P/p chg  K/k spike  H/h fdrop  7/8 warmup  9/0 cf")
        else:
            if self.laser_det:
                lines.append(self.laser_det.threshold_label())
            lines.append("w/s thr ,/. peak  g/G manual  m=mode  M=rgb method")
            lines.append("r search  o ROIsize  e offset  y misc")
        p = self.params
        if p:
            lines.append(
                f"diff={p.get('diff_threshold', 0)} "
                f"cline={p.get('min_center_line_ratio', 0):.0%} "
                f"cf={p.get('confirm_frames', 2)}"
            )
            lines.append("[/] diff  +/- cline  ;/' confirm  ( ) zone  f/F blur/morph")
        if not self.laser_enabled and self.roi:
            lines.append("1/2 ROI  WASD move  Shift+WASD size")
        return lines

    def print_help(self):
        print("=== Tuning keys (t=panel) ===")
        print("r=search ROI  o=seam W×H  e=seam offset  y=laser misc")
        print("w/s=porog  ,/.=peak  g/G=manual thr  m=grayscale/rgb  M=rgb method")
        print("[/]=diff  +/-=cline  ;/'=confirm  B=bg  P=save")
        print("A/D W/S in modes: see HUD")

    def handle_key(self, key: int, shifted: bool = False) -> bool:
        if key in (ord("t"), ord("T"), ord("n"), ord("N")):
            on = self.toggle()
            print("TUNING", "ON" if on else "OFF")
            return True

        if key in (ord("p"), ord("P")):
            self.save()
            return True

        if not self.active:
            return False

        step = self._step_big if shifted else self._step

        if key == ord("r"):
            self._toggle_sub("laser_search")
            return True
        if key == ord("o"):
            self._toggle_sub("laser_roi")
            return True
        if key == ord("e"):
            self._toggle_sub("laser_seam")
            return True
        if key == ord("y"):
            self._toggle_sub("laser_misc")
            return True

        if self._handle_seam_keys(key, shifted):
            return True

        if self.laser_det and self._handle_laser_keys(key, shifted, step):
            return True

        if not self.laser_enabled and self.roi and self._handle_manual_roi(key, shifted, step):
            return True

        return False

    def save(self):
        cfg = self.cfg
        if self.laser_det:
            cfg["laser"] = self.laser_det.to_config(cfg.get("laser", {}))
            cfg["laser"]["roi_w"] = self.roi_w
            cfg["laser"]["roi_h"] = self.roi_h
            cfg["laser"]["seam_offset_l_x"] = self.off_l_x
            cfg["laser"]["seam_offset_r_x"] = self.off_r_x
            cfg["laser"]["seam_offset_y"] = self.off_y
        if "laser_main" not in cfg:
            cfg["laser_main"] = {}
        cfg["laser_main"].update(self.l_main)
        if self.laser_det:
            cfg["laser_main"]["min_gap_ratio"] = self.laser_det.min_gap_ratio
            cfg["laser_main"]["max_gap_ratio"] = self.laser_det.max_gap_ratio
        if self.seam_tracker:
            cfg["laser_seam"] = self.seam_tracker.to_config(cfg.get("laser_seam", {}))
        if self.params:
            det = cfg.setdefault("detection", {})
            det.update(self.params)
            det["blur"] = self.blur
            det["cooldown_ms"] = self.cooldown_ms
            det.pop("left", None)
            det.pop("right", None)
        if self.roi and not self.laser_enabled:
            cfg["roi_left"] = self.roi.left
            cfg["roi_right"] = self.roi.right
        save_config(cfg)
        print("Saved config.json")

    def _toggle_sub(self, name: str):
        self.sub_mode = None if self.sub_mode == name else name
        self.drag = None
        print(f"TUNE mode: {self.sub_mode or 'laser_thr+seam'}")

    def _search_rect(self):
        if not self.laser_det:
            return ""
        d = self.laser_det
        return f"{d.sx},{d.sy} {d.sw}x{d.sh}"

    def _on_mouse(self, event, x, y, flags, _param):
        if not self.active or self.sub_mode != "laser_search" or not self.laser_det:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag = (x, y, x, y)
        elif self.drag and event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            self.drag = (self.drag[0], self.drag[1], x, y)
        elif self.drag and event == cv2.EVENT_LBUTTONUP:
            x0, y0, x1, y1 = self.drag
            self.drag = None
            xa, xb = sorted((x0, x1))
            ya, yb = sorted((y0, y1))
            if xb - xa >= 20 and yb - ya >= 20:
                det = self.laser_det
                det.sx, det.sy = xa, ya
                det.sw, det.sh = xb - xa, yb - ya
                det.clamp_search_roi(det.fw, det.fh)
                det.reset_smooth()
                print(f"search_roi: {det.sx},{det.sy} {det.sw}x{det.sh}")

    def _handle_seam_keys(self, key: int, shifted: bool) -> bool:
        p = self.params
        if not p:
            return False
        if key == ord("["):
            p["diff_threshold"] = max(4, int(p["diff_threshold"]) - 2)
            print(f"TUNE diff={p['diff_threshold']}")
            return True
        if key == ord("]"):
            p["diff_threshold"] = min(80, int(p["diff_threshold"]) + 2)
            print(f"TUNE diff={p['diff_threshold']}")
            return True
        if key in (ord("+"), ord("=")):
            p["min_center_line_ratio"] = max(
                0.25, round(p["min_center_line_ratio"] - 0.05, 2)
            )
            print(f"TUNE cline={p['min_center_line_ratio']:.0%}")
            return True
        if key == ord("-"):
            p["min_center_line_ratio"] = min(
                0.9, round(p["min_center_line_ratio"] + 0.05, 2)
            )
            print(f"TUNE cline={p['min_center_line_ratio']:.0%}")
            return True
        if key == ord(";"):
            p["confirm_frames"] = max(1, int(p["confirm_frames"]) - 1)
            print(f"TUNE confirm={p['confirm_frames']}")
            return True
        if key == ord("'"):
            p["confirm_frames"] = min(8, int(p["confirm_frames"]) + 1)
            print(f"TUNE confirm={p['confirm_frames']}")
            return True
        if key == ord("("):
            p["center_zone_ratio"] = max(0.2, round(p["center_zone_ratio"] - 0.05, 2))
            print(f"TUNE zone={p['center_zone_ratio']:.0%}")
            return True
        if key == ord(")"):
            p["center_zone_ratio"] = min(0.6, round(p["center_zone_ratio"] + 0.05, 2))
            print(f"TUNE zone={p['center_zone_ratio']:.0%}")
            return True
        if key == ord("f"):
            p["diff_blur_k"] = max(1, int(p.get("diff_blur_k", 3)) - 1)
            print(f"TUNE diff_blur_k={p['diff_blur_k']}")
            return True
        if key == ord("F"):
            p["vertical_morph_h"] = max(1, int(p.get("vertical_morph_h", 5)) - 1)
            print(f"TUNE vertical_morph_h={p['vertical_morph_h']}")
            return True
        if key == ord("9"):
            p["diff_blur_k"] = min(15, int(p.get("diff_blur_k", 3)) + 1)
            print(f"TUNE diff_blur_k={p['diff_blur_k']}")
            return True
        if key == ord("0"):
            p["vertical_morph_h"] = min(31, int(p.get("vertical_morph_h", 5)) + 1)
            print(f"TUNE vertical_morph_h={p['vertical_morph_h']}")
            return True
        if key in (ord("b"), ord("B")):
            return False  # main handles bg capture
        return False

    def _handle_laser_keys(self, key: int, shifted: bool, step: int) -> bool:
        det = self.laser_det
        if self.sub_mode == "laser_search":
            if key == ord("w"):
                det.move_search_roi(0, -step)
                return True
            if key == ord("s"):
                det.move_search_roi(0, step)
                return True
            if key == ord("a"):
                det.move_search_roi(-step, 0)
                return True
            if key == ord("d"):
                det.move_search_roi(step, 0)
                return True
            if key == ord("j"):
                det.resize_search_roi(-step * 2, 0)
                return True
            if key == ord("l"):
                det.resize_search_roi(step * 2, 0)
                return True
            if key == ord("i"):
                det.resize_search_roi(0, -step * 2)
                return True
            if key == ord("k"):
                det.resize_search_roi(0, step * 2)
                return True
            return False

        if self.sub_mode == "laser_roi":
            if key == ord("a"):
                self.roi_w = max(20, self.roi_w - step)
                print(f"seam ROI W={self.roi_w}")
                return True
            if key == ord("d"):
                self.roi_w = min(400, self.roi_w + step)
                print(f"seam ROI W={self.roi_w}")
                return True
            if key == ord("w"):
                self.roi_h = max(40, self.roi_h - step)
                print(f"seam ROI H={self.roi_h}")
                return True
            if key == ord("s"):
                self.roi_h = min(600, self.roi_h + step)
                print(f"seam ROI H={self.roi_h}")
                return True
            return False

        if self.sub_mode == "laser_seam":
            if key == ord("w"):
                self.off_y -= step
                return True
            if key == ord("s"):
                self.off_y += step
                return True
            if key == ord("a"):
                self.off_l_x -= step
                return True
            if key == ord("d"):
                self.off_l_x += step
                return True
            if key == ord("j"):
                self.off_r_x -= step
                return True
            if key == ord("l"):
                self.off_r_x += step
                return True
            if key == ord("i"):
                self.off_y -= step
                return True
            if key == ord("k"):
                self.off_y += step
                return True
            return False

        if self.sub_mode == "laser_misc":
            if key == ord("u"):
                det.nudge_min_col_fill(0.01)
                det.reset_smooth()
                print(f"min_col_fill={det.min_col_fill:.2f}")
                return True
            if key == ord("j"):
                det.nudge_min_col_fill(-0.01)
                det.reset_smooth()
                print(f"min_col_fill={det.min_col_fill:.2f}")
                return True
            if key == ord("n"):
                det.nudge_gap_ratio("min", 0.01)
                det.reset_smooth()
                return True
            if key == ord("m"):
                det.nudge_gap_ratio("min", -0.01)
                det.reset_smooth()
                return True
            if key in (ord("z"), ord("Z")):
                det.nudge_blur(1 if key == ord("z") else -1)
                det.reset_smooth()
                print(f"blur={det.blur}")
                return True
            if key == ord("c"):
                det.nudge_morph(1)
                det.reset_smooth()
                print(f"morph={det.morph_k}")
                return True
            if key == ord("C"):
                det.nudge_morph(-1)
                det.reset_smooth()
                print(f"morph={det.morph_k}")
                return True
            if key == ord("["):
                self.l_main["need_found_frames"] = max(
                    1, int(self.l_main.get("need_found_frames", 5)) - 1
                )
                print(f"need_found={self.l_main['need_found_frames']}")
                return True
            if key == ord("]"):
                self.l_main["need_found_frames"] = min(
                    30, int(self.l_main.get("need_found_frames", 5)) + 1
                )
                print(f"need_found={self.l_main['need_found_frames']}")
                return True
            if key == ord(";"):
                self.l_main["lost_error_frames"] = max(
                    3, int(self.l_main.get("lost_error_frames", 15)) - 1
                )
                print(f"lost_err={self.l_main['lost_error_frames']}")
                return True
            if key == ord("'"):
                self.l_main["lost_error_frames"] = min(
                    120, int(self.l_main.get("lost_error_frames", 15)) + 1
                )
                print(f"lost_err={self.l_main['lost_error_frames']}")
                return True
            st = self.seam_tracker
            if st:
                if key == ord("q"):
                    st.nudge_drop_ratio(-0.02)
                    st.reset()
                    print(f"peak_drop_ratio={st.peak_drop_ratio:.0%}")
                    return True
                if key == ord("Q"):
                    st.nudge_drop_ratio(0.02)
                    st.reset()
                    print(f"peak_drop_ratio={st.peak_drop_ratio:.0%}")
                    return True
                if key == ord("p"):
                    st.nudge_profile_change(-0.005)
                    st.reset()
                    print(f"profile_change_min={st.profile_change_min:.3f}")
                    return True
                if key == ord("P"):
                    st.nudge_profile_change(0.005)
                    st.reset()
                    print(f"profile_change_min={st.profile_change_min:.3f}")
                    return True
                if key == ord("k"):
                    st.nudge_change_spike(-0.002)
                    st.reset()
                    print(f"change_spike_min={st.change_spike_min:.3f}")
                    return True
                if key == ord("K"):
                    st.nudge_change_spike(0.002)
                    st.reset()
                    print(f"change_spike_min={st.change_spike_min:.3f}")
                    return True
                if key == ord("h"):
                    st.nudge_frame_drop(-0.01)
                    st.reset()
                    print(f"frame_drop_ratio={st.frame_drop_ratio:.0%}")
                    return True
                if key == ord("H"):
                    st.nudge_frame_drop(0.01)
                    st.reset()
                    print(f"frame_drop_ratio={st.frame_drop_ratio:.0%}")
                    return True
                if key == ord("9"):
                    st.nudge_confirm(-1)
                    st.reset()
                    print(f"confirm_frames={st.confirm_frames}")
                    return True
                if key == ord("0"):
                    st.nudge_confirm(1)
                    st.reset()
                    print(f"confirm_frames={st.confirm_frames}")
                    return True
                if key == ord("7"):
                    st.nudge_warmup(-2)
                    st.reset()
                    print(f"warmup_frames={st.warmup_frames}")
                    return True
                if key == ord("8"):
                    st.nudge_warmup(2)
                    st.reset()
                    print(f"warmup_frames={st.warmup_frames}")
                    return True
            return False

        # default laser threshold / peak
        if key == ord("w"):
            det.nudge_percentile(-0.5)
            det.reset_smooth()
            print(det.threshold_label())
            return True
        if key == ord("s"):
            det.nudge_percentile(0.5)
            det.reset_smooth()
            print(det.threshold_label())
            return True
        if key == ord(","):
            det.nudge_peak_prominence(-0.05)
            det.reset_smooth()
            return True
        if key == ord("."):
            det.nudge_peak_prominence(0.05)
            det.reset_smooth()
            return True
        if key == ord("g"):
            det.nudge_bright(-5)
            det.reset_smooth()
            print(det.threshold_label())
            return True
        if key == ord("G"):
            det.nudge_bright(5)
            det.reset_smooth()
            print(det.threshold_label())
            return True
        if key == ord("m"):
            mode = det.cycle_search_mode()
            print(f"search_mode={mode}")
            return True
        if key == ord("M"):
            method = det.cycle_rgb_method()
            print(f"rgb_method={method}")
            return True
        return False

    def _handle_manual_roi(self, key: int, shifted: bool, step: int) -> bool:
        roi = self.roi
        if key == ord("1"):
            roi.active = "left"
            return True
        if key == ord("2"):
            roi.active = "right"
            return True
        big = step * 2
        if key == ord("w"):
            roi.move(0, -step)
            return True
        if key == ord("s"):
            roi.move(0, step)
            return True
        if key == ord("a"):
            roi.move(-step, 0)
            return True
        if key == ord("d"):
            roi.move(step, 0)
            return True
        if key == ord("W"):
            roi.resize(0, big)
            return True
        if key == ord("S"):
            roi.resize(0, -big)
            return True
        if key == ord("A"):
            roi.resize(-big, 0)
            return True
        if key == ord("D"):
            roi.resize(big, 0)
            return True
        return False
