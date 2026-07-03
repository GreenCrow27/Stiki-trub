"""
Проверка лазеров и стыков (jump) на видеофайле.

  python laser_video.py путь/к/video.mp4
  python laser_video.py                    — путь из config.json → camera.video_file

Управление:
  ESC     — выход
  Пробел  — пауза / продолжить
  , / .   — кадр назад / вперёд (на паузе)
  Home    — в начало
  S       — START (зафиксировать ROI, ждать стыки по jump)
  R       — сброс в SEARCH
  v       — маска
  t       — tuning (см. main_laser)
"""
import argparse
import os
import sys
import time

import cv2

from config_utils import load_config
from camera_io import open_video_file, VideoFileStream
from laser_detector import LaserDetector, rois_from_lasers
from laser_seam import LaserSeamTracker
from laser_ui import draw_laser_overlay
from tuning_controls import KeyboardTuning
from vision_utils import SeamResult

SEARCH, WAIT_L, WAIT_R = "SEARCH", "WAIT_LEFT", "WAIT_RIGHT"


def _pick_video_path(args, cam: dict) -> str:
    if args.video:
        return os.path.normpath(args.video)
    vf = str(cam.get("video_file", "")).strip()
    if vf:
        return os.path.normpath(vf)
    return ""


def _frame_delay(stream: VideoFileStream, paused: bool, speed: float) -> float:
    if paused:
        return 1
    fps = stream.fps if stream.fps > 1 else 25.0
    return max(1, int(1000.0 / (fps * max(0.1, speed))))


def main():
    parser = argparse.ArgumentParser(description="Лазер + стык на видеофайле")
    parser.add_argument(
        "video",
        nargs="?",
        default="",
        help="Путь к mp4/avi (или camera.video_file в config.json)",
    )
    parser.add_argument("--no-loop", action="store_true", help="Не зацикливать в конце")
    parser.add_argument("--speed", type=float, default=1.0, help="Скорость 1.0 = как в файле")
    args = parser.parse_args()

    cfg = load_config()
    cam = cfg.get("camera", {})
    l_cfg = cfg.get("laser", {})
    l_seam_cfg = cfg.get("laser_seam", {})

    path = _pick_video_path(args, cam)
    if not path:
        print("Укажите файл: python laser_video.py recordings/pipe_raw_xxx.mp4")
        print("Или задайте camera.video_file в config.json")
        sys.exit(1)
    if not os.path.isfile(path):
        print("Файл не найден:", path)
        sys.exit(1)

    loop = not args.no_loop and bool(cam.get("video_loop", True))
    stream = open_video_file(path, loop=loop)
    fw = stream.width or int(cam.get("width", 1280))
    fh = stream.height or int(cam.get("height", 720))

    det = LaserDetector(l_cfg, fw, fh)
    seam_tracker = LaserSeamTracker(l_seam_cfg)
    tune = KeyboardTuning(
        cfg, laser_det=det, laser_enabled=True, seam_tracker=seam_tracker, params={},
    )
    tune.print_help()

    state = SEARCH
    status = "SEARCH"
    show_mask = False
    paused = False
    speed = float(args.speed)
    fixed_left = fixed_right = None
    left_t = 0.0
    left_armed = True
    right_latched = False
    idle = SeamResult(phase="OFF", debug="")

    print("=== laser_video ===")
    print("Файл:", path)
    print("Кадров:", stream.frame_count, "FPS:", "{:.1f}".format(stream.fps))
    print("mode=", det.search_mode, "seam=", seam_tracker.hud_line())
    print("S=START  R=SEARCH  Пробел=пауза  , .=кадр  Home=начало")

    cv2.namedWindow("Laser Video", cv2.WINDOW_NORMAL)
    tune.attach_window("Laser Video")

    last_frame = None

    try:
        while True:
            if not paused:
                ok, frame = stream.read()
                if not ok or frame is None:
                    if not loop:
                        print("Конец видео.")
                        break
                    continue
                last_frame = frame
            else:
                frame = last_frame
                if frame is None:
                    ok, frame = stream.read()
                    if not ok or frame is None:
                        continue
                    last_frame = frame

            if frame.shape[1] != fw or frame.shape[0] != fh:
                frame = cv2.resize(frame, (fw, fh))

            if state == SEARCH:
                laser = det.detect(frame, fw, fh)
            else:
                active = "left" if state == WAIT_L else "right"
                laser = det.read_seam_peaks(
                    frame, fw, fh,
                    hint_l=seam_tracker.baseline_l,
                    hint_r=seam_tracker.baseline_r,
                    active=active,
                )
            l_res, r_res = idle, idle

            if state == WAIT_L and laser.found:
                _, l_res = seam_tracker.check_left(laser, det)
            elif state == WAIT_R and laser.found:
                _, r_res = seam_tracker.check_right(laser, det)

            pos = stream.position
            total = stream.frame_count
            hud = [
                status,
                "frame {}/{}  {}{}".format(
                    pos, total, "PAUSE" if paused else "",
                    "" if speed == 1.0 else " x{:.1f}".format(speed),
                ),
                "S=START R=SEARCH  , .=step  Home=rewind  v=mask",
            ]
            if state in (WAIT_L, WAIT_R):
                hud.append(seam_tracker.hud_line())
                if state == WAIT_L:
                    hud.append("L " + l_res.debug)
                else:
                    hud.append("R " + r_res.debug)

            draw_seam = state == SEARCH and laser.found
            if state in (WAIT_L, WAIT_R) and fixed_left and fixed_right:
                draw_seam = False

            draw_laser_overlay(
                frame, det, laser, fw, fh,
                tune.roi_w, tune.roi_h,
                tune.off_l_x, tune.off_r_x, tune.off_y,
                active_calib=(state == SEARCH),
                show_mask=show_mask and state == SEARCH,
                draw_seam_rois=draw_seam,
                hud_extra=hud + tune.hud_lines(),
                seam_tracker=seam_tracker,
            )

            if state in (WAIT_L, WAIT_R) and fixed_left and fixed_right:
                for r, col, lab in (
                    (fixed_left, (0, 255, 0), "L"),
                    (fixed_right, (0, 180, 255), "R"),
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

            if state == WAIT_L and l_res.hit and left_armed:
                left_t = time.time() * 1000.0
                state = WAIT_R
                status = "TIMER ON"
                left_armed = False
                right_latched = False
                seam_tracker.rearm_right(laser, det)
                print(">>> LEFT DROP frame={} {}".format(pos, l_res.debug))
            elif state == WAIT_R and r_res.hit and not right_latched:
                right_latched = True
                dt_ms = time.time() * 1000.0 - left_t
                print(">>> RIGHT DROP frame={} dt={:.0f}ms {}".format(
                    pos, dt_ms, r_res.debug,
                ))
                state = SEARCH
                status = "SEARCH — ждём START"
                left_armed = True
                fixed_left = fixed_right = None
                seam_tracker.reset()

            cv2.imshow("Laser Video", frame)
            delay = _frame_delay(stream, paused, speed)
            key = cv2.waitKeyEx(delay)
            if key < 0:
                continue
            ch = key & 0xFF
            ext = (key >> 16) & 0xFF
            shifted = ext not in (0, 255, 254)

            if ch == 27:
                break
            if ch == ord(" "):
                paused = not paused
                print("PAUSE" if paused else "PLAY")
                continue
            if paused and ch == ord(","):
                stream.seek(max(0, stream.position - 2))
                ok, frame = stream.read()
                if ok and frame is not None:
                    last_frame = frame
                continue
            if paused and ch == ord("."):
                ok, frame = stream.read()
                if ok and frame is not None:
                    last_frame = frame
                continue
            if ch == 36 or ch == 2:  # Home
                stream.rewind()
                ok, frame = stream.read()
                if ok and frame is not None:
                    last_frame = frame
                continue
            if ch in (ord("-"), ord("_")):
                speed = max(0.1, speed - 0.25)
                print("speed x{:.2f}".format(speed))
                continue
            if ch in (ord("="), ord("+")):
                speed = min(4.0, speed + 0.25)
                print("speed x{:.2f}".format(speed))
                continue
            if tune.handle_key(ch, shifted):
                continue
            if ch == ord("v"):
                show_mask = not show_mask
                continue
            if ch in (ord("r"), ord("R")) and not tune.blocks_search_toggle(ch):
                state = SEARCH
                status = "SEARCH — ждём START"
                fixed_left = fixed_right = None
                left_armed = True
                right_latched = False
                seam_tracker.reset()
                det.reset_smooth()
                print("SEARCH")
                continue
            if ch in (ord("s"), ord("S")) and not tune.blocks_start():
                if not laser.found:
                    print("START: нет лазера на этом кадре")
                    continue
                if not laser.peaks_live:
                    laser = det.read_seam_peaks(
                        frame, fw, fh,
                        hint_l=laser.peak_l or None,
                        hint_r=laser.peak_r or None,
                        active="both",
                    )
                fixed_left, fixed_right = rois_from_lasers(
                    laser, tune.roi_w, tune.roi_h,
                    tune.off_l_x, tune.off_r_x, tune.off_y, fw, fh,
                )
                seam_tracker.arm(laser, det)
                state = WAIT_L
                status = "WAIT LEFT"
                left_armed = True
                right_latched = False
                print("START frame={} ROI L={} R={}".format(
                    pos, fixed_left, fixed_right,
                ))
    finally:
        stream.close()
        cv2.destroyAllWindows()
        print("Готово.")


if __name__ == "__main__":
    main()
