"""
Тест: две лазерные полоски -> L/R ROI.
Запуск:
  python main_laser.py
  python main_laser.py --video recordings/pipe_raw_xxx.mp4
"""
import argparse
import os

import cv2

from config_utils import load_config
from camera_io import open_media_source, open_video_file
from laser_detector import LaserDetector
from laser_ui import draw_laser_overlay
from tuning_controls import KeyboardTuning


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="", help="Путь к видеофайлу")
    args = parser.parse_args()

    cfg = load_config()
    l_cfg = cfg.get("laser", {})
    cam = dict(cfg.get("camera", {}))
    if args.video:
        cam["video_file"] = args.video

    if cam.get("video_file") and os.path.isfile(str(cam["video_file"])):
        stream = open_video_file(
            str(cam["video_file"]),
            loop=bool(cam.get("video_loop", True)),
        )
    else:
        stream = open_media_source(cam)
    if not stream.isOpened():
        raise RuntimeError(f"Камера не открылась: {stream.label}")

    fw, fh = int(cam.get("width", 1280)), int(cam.get("height", 720))
    if hasattr(stream, "width") and stream.width > 0:
        fw, fh = stream.width, stream.height
    det = LaserDetector(l_cfg, fw, fh)
    show_mask = False

    tune = KeyboardTuning(cfg, laser_det=det, laser_enabled=True, params={})
    tune.print_help()

    print("=== Laser test ===")
    print(f"mode={det.search_mode} rgb={det.rgb_method}")

    cv2.namedWindow("Laser Vision")
    tune.attach_window("Laser Vision")

    try:
        while True:
            ok, frame = stream.read()
            if not ok or frame is None:
                continue
            if frame.shape[1] != fw or frame.shape[0] != fh:
                frame = cv2.resize(frame, (fw, fh))

            tune.draw_drag_preview(frame)
            laser = det.detect(frame, fw, fh)
            draw_laser_overlay(
                frame, det, laser, fw, fh,
                tune.roi_w, tune.roi_h, tune.off_l_x, tune.off_r_x, tune.off_y,
                show_mask=show_mask,
                draw_seam_rois=laser.found,
                hud_extra=["v=mask  t=tune"] + tune.hud_lines(),
            )
            cv2.imshow("Laser Vision", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if tune.handle_key(key):
                continue
            if key == ord("v"):
                show_mask = not show_mask
    finally:
        stream.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
