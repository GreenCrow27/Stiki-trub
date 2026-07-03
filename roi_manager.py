class ROIManager:
    def __init__(self, left_roi, right_roi):
        self.left = left_roi.copy()
        self.right = right_roi.copy()
        self.active = 'left'   # 'left' или 'right'

    def clamp(self, w, h):
        """Приводит ROI к границам кадра"""
        for roi in (self.left, self.right):
            roi["x"] = max(0, min(roi["x"], w - roi["w"]))
            roi["y"] = max(0, min(roi["y"], h - roi["h"]))
            roi["w"] = max(10, min(roi["w"], w - roi["x"]))
            roi["h"] = max(10, min(roi["h"], h - roi["y"]))

    def move(self, dx, dy):
        """Перемещает активную зону"""
        if self.active == 'left':
            self.left["x"] += dx
            self.left["y"] += dy
        else:
            self.right["x"] += dx
            self.right["y"] += dy

    def resize(self, dw, dh):
        """Изменяет размер активной зоны"""
        if self.active == 'left':
            self.left["w"] += dw
            self.left["h"] += dh
        else:
            self.right["w"] += dw
            self.right["h"] += dh

    def get_active(self):
        return self.left if self.active == 'left' else self.right

    def switch(self):
        self.active = 'right' if self.active == 'left' else 'left'

    def get_rects(self):
        return (self.left["x"], self.left["y"], self.left["w"], self.left["h"]), \
               (self.right["x"], self.right["y"], self.right["w"], self.right["h"])