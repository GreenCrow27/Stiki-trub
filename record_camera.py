"""
Только камера → файл (оригинальный кадр, без оверлея).
Запуск: python record_camera.py
Настройки: config.json → recording
  ESC — выход
  R   — пауза/продолжить запись
"""
import cv2

from config_utils import load_config
from camera_io import open_camera
from video_recorder import RawVideoRecorder


def main():
    cfg = load_config()
    cam = cfg["camera"]
    rec_cfg = dict(cfg.get("recording", {}))
    rec_cfg["enabled"] = True

    stream = open_camera(cam)
    if not stream.isOpened():
        raise RuntimeError("Камера не открылась: {}".format(stream.label))

    recorder = RawVideoRecorder(rec_cfg)
    recorder.set_stream_fps(float(cam.get("fps") or getattr(stream, "fps", 0) or 25))
    paused = False
    show_preview = bool(rec_cfg.get("preview", True))

    print("=== record_camera ===")
    print("Камера:", stream.label)
    print("ESC=выход  R=пауза записи  preview={}".format(show_preview))

    try:
        while True:
            ok, frame = stream.read()
            if not ok or frame is None:
                continue
            h, w = frame.shape[:2]
            if not paused:
                recorder.write(frame)
            if show_preview:
                tag = "REC" if recorder.active and not paused else "PAUSE"
                cv2.putText(
                    frame, tag, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA,
                )
                cv2.imshow("Record (preview only)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in (ord("r"), ord("R")):
                paused = not paused
                print("запись:", "PAUSE" if paused else "ON")
    finally:
        recorder.close()
        stream.close()
        cv2.destroyAllWindows()
        print("Готово.")


if __name__ == "__main__":
    main()
