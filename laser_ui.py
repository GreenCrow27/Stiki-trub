"""Отрисовка лазерного режима (общая для main_laser.py и main.py)."""
import cv2

from laser_detector import LaserDetector, LaserInfo, rois_from_lasers


def _draw_projection_graph(frame, det, laser, seam_tracker=None):
    """Гистограмма col_fill снизу + маркеры пиков L/R."""
    h, w = frame.shape[:2]
    if det.last_proj is None or det.sw <= 40:
        return
    sx = det.sx
    sy = det.sy + det.sh + 8
    if sy >= h - 30:
        return
    pw = min(det.sw, w - sx - 4)
    proj = det.last_proj
    if len(proj) < 2 or pw < 10:
        return

    mx = float(proj.max()) or 1.0
    ph = 36
    cv2.rectangle(
        frame, (sx, sy), (sx + pw - 1, sy + ph + 14),
        (40, 40, 40), 1, cv2.LINE_AA,
    )
    for i in range(pw):
        xi = int(i * (len(proj) - 1) / max(pw - 1, 1))
        val = int(proj[xi] / mx * ph)
        cv2.line(frame, (sx + i, sy + ph), (sx + i, sy + ph - val), (0, 140, 200), 1)

    if laser.found:
        for peak_x, col, lab in (
            (laser.peak_l, (0, 255, 0), "L"),
            (laser.peak_r, (0, 200, 255), "R"),
        ):
            if peak_x <= 0:
                continue
            px = int(sx + (peak_x - det.sx) * (pw - 1) / max(len(proj) - 1, 1))
            px = max(sx, min(sx + pw - 1, px))
            cv2.line(frame, (px, sy), (px, sy + ph + 10), col, 2, cv2.LINE_AA)
            cv2.putText(
                frame, lab, (px + 2, sy + ph + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA,
            )
        live = "live" if laser.peaks_live else "hold"
        cv2.putText(
            frame, "proj peaks ({})".format(live), (sx, sy - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA,
        )
        if seam_tracker is not None:
            cv2.putText(
                frame,
                "dL drop {:.0%} chg {:.3f} sp {:.3f} | "
                "dR drop {:.0%} chg {:.3f} sp {:.3f}".format(
                    seam_tracker.last_drop_l,
                    seam_tracker.last_change_l,
                    seam_tracker.last_spike_l,
                    seam_tracker.last_drop_r,
                    seam_tracker.last_change_r,
                    seam_tracker.last_spike_r,
                ),
                (sx, sy + ph + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1, cv2.LINE_AA,
            )


def draw_laser_hud(frame, det, laser, show_mask=False, extra_lines=None, seam_tracker=None):
    """HUD: порог, статус, диагностика, график проекции с пиками."""
    h, w = frame.shape[:2]
    y = 22
    lines = [det.threshold_label()]
    if extra_lines:
        lines.extend(extra_lines)
    if laser.found:
        lines.append(
            "OK  peaks L={:.0f} R={:.0f}  {}".format(
                laser.peak_l, laser.peak_r,
                "live" if laser.peaks_live else "hold",
            )
        )
        if laser.debug:
            lines.append("    " + laser.debug)
    else:
        lines.append("NO  " + laser.debug)
        if det.last_diag:
            lines.append("    " + det.last_diag)
        lines.append("    fill={:.2f}".format(det.last_col_max))
    if show_mask:
        lines.append("MASK ON")

    for i, text in enumerate(lines):
        if laser.found:
            col = (0, 255, 180) if i == 0 else (200, 200, 200)
        elif i == 0:
            col = (0, 220, 255)
        else:
            col = (0, 100, 255)
        cv2.putText(
            frame, text, (10, y + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1, cv2.LINE_AA,
        )

    _draw_projection_graph(frame, det, laser, seam_tracker=seam_tracker)


def draw_laser_overlay(
    frame,
    det: LaserDetector,
    laser: LaserInfo,
    fw: int,
    fh: int,
    roi_w: int,
    roi_h: int,
    seam_off_l_x: int,
    seam_off_r_x: int,
    seam_off_y: int,
    *,
    active_calib: bool = False,
    show_mask: bool = False,
    draw_seam_rois: bool = True,
    draw_hud: bool = True,
    hud_extra=None,
    seam_tracker=None,
):
    """search_roi, линии лазера, L/R ROI, гистограмма пиков."""
    sx, sy, sw, sh = det.sx, det.sy, det.sw, det.sh
    col_search = (0, 220, 255) if active_calib else (100, 100, 100)
    cv2.rectangle(
        frame, (sx, sy), (sx + sw - 1, sy + sh - 1),
        col_search, 2 if active_calib else 1, cv2.LINE_AA,
    )

    if show_mask and det.last_mask is not None:
        tint = cv2.cvtColor(det.last_mask, cv2.COLOR_GRAY2BGR)
        cv2.addWeighted(tint, 0.4, frame, 0.6, 0, frame)

    if laser.found:
        x0, x1 = int(round(laser.peak_l)), int(round(laser.peak_r))
        bot = int(laser.bottom_y if laser.bottom_y > 0 else laser.y + laser.h - 1)
        cv2.line(frame, (x0, laser.y), (x0, bot), (0, 255, 0), 2, cv2.LINE_AA)
        cv2.line(frame, (x1, laser.y), (x1, bot), (0, 200, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (sx, bot), (sx + sw - 1, bot), (0, 220, 0), 2, cv2.LINE_AA)

        if draw_seam_rois:
            left, right = rois_from_lasers(
                laser, roi_w, roi_h, seam_off_l_x, seam_off_r_x, seam_off_y, fw, fh,
            )
            for r, col in ((left, (0, 255, 0)), (right, (0, 180, 255))):
                cv2.rectangle(
                    frame, (r["x"], r["y"]),
                    (r["x"] + r["w"] - 1, r["y"] + r["h"] - 1),
                    col, 2, cv2.LINE_AA,
                )

    if draw_hud:
        draw_laser_hud(
            frame, det, laser, show_mask=show_mask,
            extra_lines=hud_extra, seam_tracker=seam_tracker,
        )
