# Modbus (регистры 0–7)

**Камера → ПЛК:** рег.0 биты — LEFT(1), RIGHT(2), JOINT(4), DONE(8), WDG_REQ(16), ACTIVE(32), READY(64), ERROR(128); рег.1 SPEED×10000; рег.2 ERROR_CODE; рег.3 watchdog timeout.

**ПЛК → камера:** рег.4 — RESET(1), WDG_ACK(2), START(4), WDG_EN(8), DIST_OK(16); рег.5 DISTANCE_MM; рег.6 WDG_SET; рег.7 GRAD_THRESH.

Старт: рег.5=расстояние, рег.4|=16+4 (DIST_OK+START). Сброс ошибки: рег.4 бит0=1.
