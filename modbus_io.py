"""
Modbus TCP: камера (master) ↔ ПЛК (slave).
Все сигналы — Holding Register (UINT16), в т.ч. CMD/STATUS (биты в слове).

Адреса в config.json — смещения/MW (195, 257) или полные 4xxxx (40196).
address_bases.holding_register (обычно 40001):
  адрес_ПЛК_в_документации = config + base  (если config < base)
  адрес_pymodbus (wire)     = адрес_ПЛК - base
"""
import threading
import time

try:
    from pymodbus.client import ModbusTcpClient
except ImportError:  # pymodbus 2.x (Python 3.7)
    from pymodbus.client.sync import ModbusTcpClient

ERR_NONE = 0
ERR_GENERAL = 1
ERR_NO_DIST = 2
ERR_TIMEOUT = 3
ERR_TIMEOUT_RIGHT = 3
ERR_NO_LASER = 4
ERR_INVALID_DIST = 5
ERR_MODBUS = 6
ERR_CAMERA = 7
ERR_LASER_UNSTABLE = 8
ERR_BAD_MEASURE = 9
ERR_BAD_START = 10
ERR_ROI = 11

ERR_NAMES = {
    1: "general",
    2: "no_distance",
    3: "timeout_right",
    4: "no_laser",
    5: "invalid_distance",
    6: "modbus",
    7: "camera",
    8: "laser_unstable",
    9: "bad_measure",
    10: "bad_start",
    11: "roi",
}

ERR_NAMES_RU = {
    0: "OK",
    1: "Общая ошибка",
    2: "Нет дистанции",
    3: "Таймаут правого стыка",
    4: "Нет лазера",
    5: "Дистанция вне диапазона",
    6: "Нет связи Modbus",
    7: "Нет кадра / камера",
    8: "Лазер нестабилен",
    9: "Некорректное измерение",
    10: "Некорректный START",
    11: "Ошибка ROI",
}


def error_status_text(code):
    """Текст для HUD: пусто при 0, иначе «ERR 2» …"""
    c = int(code)
    if c <= 0:
        return ""
    ru = ERR_NAMES_RU.get(c)
    if ru:
        return f"ERR {c} {ru}"
    return f"ERR {c}"


def check_distance_mm(dist, min_mm=10, max_mm=5000):
    """0 = OK, иначе код ERR_NO_DIST / ERR_INVALID_DIST."""
    d = int(dist)
    if d <= 0:
        return ERR_NO_DIST
    if d < int(min_mm) or d > int(max_mm):
        return ERR_INVALID_DIST
    return ERR_NONE


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


_HR = {
    "status_out": 195,
    "speed": 196,
    "error_code": 197,
    "cmd": 257,
    "dist_mm": 258,
}
_BITS_ST = {
    "left": 0,
    "right": 1,
    "joint": 2,
    "done": 3,
    "active": 5,
    "ready": 6,
    "error": 7,
    "alive": 8,
}
_BITS_CMD = {"reset": 0, "search": 1, "start": 2, "dist_ok": 4}

MODBUS_AREA_BASE = {
    "holding_register": 40001,
    "input_register": 30001,
    "coil": 1,
    "discrete_input": 10001,
}


def resolve_modbus_address(
    config_val,
    area="holding_register",
    bases=None,
    legacy_address_base=None,
):
    merged = {**MODBUS_AREA_BASE, **(bases or {})}
    area_base = int(merged[area])
    a = int(config_val)

    if legacy_address_base is not None:
        leg = int(legacy_address_base)
        if a >= 40001:
            wire, plc = a - 40001, a
        else:
            wire = a - leg
            plc = a + 40001
        return max(0, wire), plc

    if a >= area_base:
        return max(0, a - area_base), a
    return a, a + area_base


def build_register_maps(hr_cfg, bases=None, legacy_address_base=None, area="holding_register"):
    wire, plc = {}, {}
    for key, val in hr_cfg.items():
        w, p = resolve_modbus_address(val, area, bases, legacy_address_base)
        wire[key] = w
        plc[key] = p
    return wire, plc


class ModbusIO:
    def __init__(
        self,
        host="10.9.177.30",
        port=502,
        timeout=0.3,
        modbus_cfg=None,
    ):
        c = modbus_cfg or {}
        self.address_bases = {**MODBUS_AREA_BASE, **c.get("address_bases", {})}
        legacy_base = c.get("address_base") if "address_bases" not in c else None
        hr = {**_HR, **c.get("holding_registers", c.get("registers", {}))}
        self.reg_wire, self.reg_plc = build_register_maps(
            hr, self.address_bases, legacy_base, "holding_register"
        )
        self.bits_st = {**_BITS_ST, **c.get("status_bits", {})}
        self.bits_cmd = {**_BITS_CMD, **c.get("cmd_bits", {})}
        self.unit_id = int(c.get("unit_id", 1))
        self.speed_scale = int(c.get("speed_scale", 1000))
        bo = c.get("byte_order", {})
        self.swap_bytes = bool(c.get("swap_bytes", bo.get("enabled", False)))
        self.errors_enabled = bool(c.get("errors_enabled", True))
        hb = c.get("heartbeat", {})
        self.heartbeat_enabled = bool(hb.get("enabled", c.get("heartbeat_enabled", True)))
        self.heartbeat_interval_ms = int(hb.get("interval_ms", c.get("heartbeat_interval_ms", 500)))
        self.status_pulse_ms = int(c.get("status_pulse_ms", 500))
        self.cmd_log_on_change = bool(c.get("cmd_log_on_change", True))
        self.start_on_level = bool(c.get("start_on_level", False))
        self.poll_interval_ms = int(c.get("poll_interval_ms", 100))
        self.retries = int(c.get("retries", 0))
        self._async_enabled = bool(c.get("async_poll", False))
        self._cmd_log_last = -1
        self._cmd_read_fail_last = 0.0
        self._start_level_seen = False
        self._state_lock = threading.RLock()
        self._stop = threading.Event()
        self._poll_thread = None
        self._device_kw = {}
        self._cmd_cache = 0
        self._dist_cache = 0
        self._cmd_ok_cache = False
        self.timeout = float(timeout)
        self._host = host
        self._port = int(port)

        self.client = self._make_client()
        self._read_client = self.client
        self._status = 0
        self._written = -1
        self._err = ERR_NONE
        self._ok = False
        self._cmd_prev = 0
        self._reset_latched = False
        self._start_allowed = False
        self._heartbeat_last_ms = 0.0
        self._heartbeat_on = False
        self._left_pulse_until_ms = 0.0
        self._right_pulse_until_ms = 0.0
        self._app_ready = False
        try:
            self._ok = self.client.connect()
            if self._ok:
                self._bootstrap_device_kw(self.client)
            if self._async_enabled and self._ok:
                if bool(c.get("async_poll_separate_socket", False)):
                    self._read_client = self._make_client()
                    if not self._read_client.connect():
                        print("[Modbus] WARN: read-сокет не подключился, async_poll выкл")
                        self._async_enabled = False
                        self._read_client = self.client
                else:
                    self._read_client = self.client
            hr_base = self.address_bases["holding_register"]
            print(
                f"[Modbus] {'OK' if self._ok else 'FAIL'} {host}:{port} unit={self.unit_id} "
                f"HR base={hr_base} "
                f"status plc={self.reg_plc['status_out']} wire={self.reg_wire['status_out']} "
                f"cmd plc={self.reg_plc['cmd']} wire={self.reg_wire['cmd']}"
                + (" swap_bytes" if self.swap_bytes else "")
                + ("" if self.errors_enabled else " err=OFF")
                + (f" retries={self.retries}" if self.retries else "")
                + (" async=read-socket" if self._read_client is not self.client else " async=same-socket")
            )
            if not legacy_base and self.reg_wire["cmd"] >= 500:
                print(
                    "[Modbus] Внимание: CMD на wire="
                    f"{self.reg_wire['cmd']} — проверьте карту MB_HOLD_REG на ПЛК"
                )
        except Exception as e:
            print(f"[Modbus] {e}")

    def _make_client(self):
        try:
            return ModbusTcpClient(
                self._host,
                port=self._port,
                timeout=self.timeout,
                retries=self.retries,
            )
        except TypeError:
            return ModbusTcpClient(self._host, port=self._port, timeout=self.timeout)

    @property
    def async_enabled(self):
        return self._async_enabled

    def _bootstrap_device_kw(self, client):
        uid = int(self.unit_id) if self.unit_id else 1
        addr = self.reg_wire["cmd"]
        candidates = (
            {"device_id": uid},
            {"slave": uid},
        )
        for kw in candidates:
            try:
                r = client.read_holding_registers(addr, count=1, **kw)
                if r is not None and not r.isError():
                    self._device_kw = kw
                    print(f"[Modbus] unit_id OK ({list(kw.keys())[0]}={uid})")
                    return
            except TypeError:
                continue
            except Exception:
                continue
        self._device_kw = {"device_id": uid}
        print(f"[Modbus] WARN: probe CMD не ответил, используем device_id={uid}")

    def _u16(self, v):
        v = int(v) & 0xFFFF
        if self.swap_bytes:
            v = ((v & 0xFF) << 8) | (v >> 8)
        return v

    def _read_regs(self, client, addr, count=1, default=None):
        if not self._ok:
            return None, False
        if default is None:
            default = [0] * count
        try:
            r = client.read_holding_registers(addr, count=count, **self._device_kw)
            if r is None or r.isError():
                return default, False
            vals = [self._u16(v) for v in r.registers[:count]]
            return vals, True
        except Exception as exc:
            self._warn_read_fail(addr, exc)
            return default, False

    def _read_reg(self, client, addr, default=0):
        vals, ok = self._read_regs(client, addr, 1, default=[default])
        if not ok:
            return default, False
        return vals[0], True

    def _read_cmd_dist_regs(self, client):
        cmd_addr = self.reg_wire["cmd"]
        dist_addr = self.reg_wire["dist_mm"]
        if dist_addr == cmd_addr + 1:
            vals, ok = self._read_regs(client, cmd_addr, 2, default=[0, 0])
            if ok and len(vals) >= 2:
                return int(vals[0]) & 0xFFFF, int(vals[1]), True
        cmd, ok1 = self._read_reg(client, cmd_addr, 0)
        dist, ok2 = self._read_reg(client, dist_addr, 0)
        return int(cmd) & 0xFFFF, int(dist), bool(ok1 and ok2)

    def _read(self, addr, default=0):
        val, _ok = self._read_reg(self.client, addr, default)
        return val

    def _warn_read_fail(self, addr, detail=None):
        now = time.time()
        if now - self._cmd_read_fail_last < 2.0:
            return
        self._cmd_read_fail_last = now
        print(
            f"[Modbus] READ FAIL HR wire={addr} unit={self.unit_id} "
            f"({detail}) — проверьте карту slave и unit_id"
        )

    def _warn_write_fail(self, addr, detail=None):
        now = time.time()
        if now - self._cmd_read_fail_last < 2.0:
            return
        self._cmd_read_fail_last = now
        print(f"[Modbus] WRITE FAIL HR wire={addr} ({detail})")

    def _maybe_reconnect(self, detail=None):
        txt = str(detail or "")
        if "transaction_id" not in txt and "No response" not in txt and "Input/Output" not in txt:
            return
        try:
            self.client.close()
        except Exception:
            pass
        try:
            self._ok = self.client.connect()
            if self._ok:
                print("[Modbus] переподключение write-сокета OK")
        except Exception as exc:
            self._ok = False
            print(f"[Modbus] переподключение write-сокета FAIL: {exc}")
        if self._read_client is not self.client:
            try:
                self._read_client.close()
            except Exception:
                pass
            try:
                rc = self._read_client.connect()
                if rc:
                    print("[Modbus] переподключение read-сокета OK")
            except Exception as exc:
                print(f"[Modbus] переподключение read-сокета FAIL: {exc}")

    def _write(self, addr, val):
        if not self._ok:
            return False
        wire = self._u16(int(val))
        try:
            r = self.client.write_register(addr, wire, **self._device_kw)
            if r is not None and hasattr(r, "isError") and r.isError():
                self._warn_write_fail(addr, r)
                return False
            return True
        except Exception as exc:
            self._warn_write_fail(addr, exc)
            self._maybe_reconnect(exc)
            return False

    @staticmethod
    def has_bit(v, b):
        return bool(v & (1 << b))

    def cmd_rising(self, cmd, key):
        b = self.bits_cmd[key]
        return self.has_bit(cmd, b) and not self.has_bit(self._cmd_prev, b)

    def note_cmd(self, cmd):
        self._cmd_prev = int(cmd) & 0xFFFF

    def arm_cmd_baseline(self, cmd):
        self.note_cmd(cmd)
        st = self.bits_cmd["start"]
        if self.has_bit(cmd, st):
            self._start_allowed = False
            print(
                f"[Modbus] CMD={int(cmd) & 0xFFFF}: START ещё ON — "
                "ждём 0, затем импульс START"
            )
        else:
            self._start_allowed = True
            print(f"[Modbus] CMD={int(cmd) & 0xFFFF}: START OFF, готов к импульсу")

    def _bit(self, b, on):
        if on:
            self._status |= 1 << b
        else:
            self._status &= ~(1 << b)

    def flush(self, force=False):
        if force or self._status != self._written:
            if self._write(self.reg_wire["status_out"], self._status):
                self._written = self._status

    def _flush_locked(self, force=False):
        self.flush(force=force)

    def _locked_io(self):
        return self._state_lock

    def set_bit(self, b, on, flush=True):
        old = self._status
        self._bit(b, on)
        if flush and old != self._status:
            self.flush()

    def camera_ready(self, on=True):
        self._app_ready = bool(on)
        self.set_bit(self.bits_st["ready"], self._app_ready)

    def signal_ready(self):
        """Бит6: программа загружена и готова."""
        with self._locked_io():
            self.camera_ready(True)
            self.set_bit(self.bits_st["ready"], True, flush=False)
            self._flush_locked(force=True)
        print("[Modbus] READY (HR status bit6)")

    def sync_ready(self):
        """Поддержать бит6 после инициализации (ошибка — отдельно бит7)."""
        if self._app_ready:
            with self._locked_io():
                self.set_bit(self.bits_st["ready"], True, flush=False)

    def sync_active(self, on):
        """Бит5: идёт проезд (WAIT_L/R) и лазер виден — без записи каждый кадр."""
        with self._locked_io():
            self.set_bit(self.bits_st["active"], bool(on), flush=False)

    def tick_status(self, now_ms):
        """Импульсы L/R, мигалка alive + запись STATUS при изменении."""
        with self._locked_io():
            changed = self._tick_status_locked(now_ms)
            if changed or self._status != self._written:
                self._flush_locked()

    def _tick_status_locked(self, now_ms):
        changed = False
        bl = self.bits_st.get("left")
        if bl is not None and self._left_pulse_until_ms and now_ms >= self._left_pulse_until_ms:
            self._left_pulse_until_ms = 0.0
            if self.has_bit(self._status, bl):
                self._bit(bl, False)
                changed = True
        br = self.bits_st.get("right")
        if br is not None and self._right_pulse_until_ms and now_ms >= self._right_pulse_until_ms:
            self._right_pulse_until_ms = 0.0
            if self.has_bit(self._status, br):
                self._bit(br, False)
                changed = True
        if self._tick_heartbeat(now_ms):
            changed = True
        return changed

    def _tick_heartbeat(self, now_ms):
        """Бит8: мигалка при штатной работе."""
        if not self.heartbeat_enabled:
            return False
        key = "alive"
        if key not in self.bits_st:
            return False
        if self._heartbeat_last_ms <= 0:
            self._heartbeat_last_ms = now_ms
            return False
        if now_ms - self._heartbeat_last_ms < self.heartbeat_interval_ms:
            return False
        self._heartbeat_last_ms = now_ms
        self._heartbeat_on = not self._heartbeat_on
        old = self._status
        self._bit(self.bits_st[key], self._heartbeat_on)
        return old != self._status

    def tick_heartbeat(self, now_ms):
        self.tick_status(now_ms)

    def joint_found(self):
        with self._locked_io():
            self._bit(self.bits_st["joint"], True)
            self._flush_locked(force=True)
        print("[Modbus] JOINT (status bit2)")

    def pulse_left(self, now_ms=None):
        if now_ms is None:
            now_ms = time.perf_counter() * 1000.0
        with self._locked_io():
            self._bit(self.bits_st["left"], True)
            self._left_pulse_until_ms = now_ms + self.status_pulse_ms
            self._flush_locked(force=True)
        print(f"[Modbus] LEFT (status bit0, {self.status_pulse_ms}ms)")

    def pulse_right(self, now_ms=None):
        if now_ms is None:
            now_ms = time.perf_counter() * 1000.0
        with self._locked_io():
            self._bit(self.bits_st["right"], True)
            self._right_pulse_until_ms = now_ms + self.status_pulse_ms
            self._flush_locked(force=True)
        print(f"[Modbus] RIGHT (status bit1, {self.status_pulse_ms}ms)")

    def clear_pulses(self):
        with self._locked_io():
            self._left_pulse_until_ms = 0.0
            self._right_pulse_until_ms = 0.0
            self._bit(self.bits_st["left"], False)
            self._bit(self.bits_st["right"], False)
            self._flush_locked()

    def measure_active(self, on):
        with self._locked_io():
            self.set_bit(self.bits_st["active"], on, flush=False)
            if on:
                self._flush_locked(force=True)

    def measure_done(self, on):
        with self._locked_io():
            self.set_bit(self.bits_st["done"], on, flush=False)

    def reset_on_start(self):
        with self._locked_io():
            self._left_pulse_until_ms = 0.0
            self._right_pulse_until_ms = 0.0
            for k in ("left", "right", "joint", "done", "active"):
                if k in self.bits_st:
                    self._bit(self.bits_st[k], False)
            self._flush_locked(force=True)

    def reset_cycle(self):
        with self._locked_io():
            self._left_pulse_until_ms = 0.0
            self._right_pulse_until_ms = 0.0
            for k in ("left", "right", "joint", "done", "active"):
                if k in self.bits_st:
                    self._bit(self.bits_st[k], False)
            if "error" in self.bits_st:
                self._bit(self.bits_st["error"], False)
            self._flush_locked(force=True)

    def clear_plc_outputs(self):
        with self._locked_io():
            self._err = ERR_NONE
            self._app_ready = False
            self._status = 0
            self._written = -1
            self._left_pulse_until_ms = 0.0
            self._right_pulse_until_ms = 0.0
            self._write(self.reg_wire["error_code"], 0)
            self._write(self.reg_wire["speed"], 0)
            self._flush_locked(force=True)
        print("[Modbus] выходы ПЛК сброшены (STATUS/SPEED/ERR, ready=0)")

    def write_error(self, code):
        if not self.errors_enabled:
            return
        code = int(code)
        if code <= 0:
            self.clear_error()
            return
        if code == self._err:
            return
        try:
            with self._locked_io():
                self._err = code
                self._write(self.reg_wire["error_code"], code)
                self.set_bit(self.bits_st["error"], True, flush=False)
                self.set_bit(self.bits_st["active"], False, flush=False)
                self._flush_locked(force=True)
        except Exception as exc:
            print(f"[Modbus] write_error({code}) внутренняя ошибка: {exc}")
            return
        name = ERR_NAMES.get(code, "?")
        print(f"[Modbus] ERROR {code} ({name})")

    def clear_error(self):
        if self._err == ERR_NONE:
            return
        with self._locked_io():
            self._err = ERR_NONE
            self._write(self.reg_wire["error_code"], 0)
            self._bit(self.bits_st["error"], False)
            if self._app_ready:
                self.set_bit(self.bits_st["ready"], True, flush=False)
            self._flush_locked(force=True)

    @staticmethod
    def calc_speed_raw(dist_mm, dt_ms, scale=1000):
        if dt_ms <= 0:
            return 0
        return (int(dist_mm) * 1000 * int(scale)) // int(dt_ms)

    @staticmethod
    def calc_speed_reg(dist_mm, dt_ms, scale=1000):
        raw = ModbusIO.calc_speed_raw(dist_mm, dt_ms, scale)
        return max(0, min(65535, raw))

    @staticmethod
    def format_speed_mm_s(reg_val, scale=1000):
        v = int(reg_val)
        if v <= 0:
            return "0"
        if scale == 1000:
            return f"{v // scale}.{v % scale:03d}"
        return f"{v / scale:.3f}"

    format_speed_m_s = format_speed_mm_s

    def write_speed(self, dist_mm, dt_ms):
        raw_full = self.calc_speed_raw(dist_mm, dt_ms, self.speed_scale)
        raw = max(0, min(65535, raw_full))
        if raw_full > 65535:
            print(
                f"[Modbus] Внимание: speed reg={raw_full} > 65535, "
                f"записано {raw} (UINT16)"
            )
        with self._locked_io():
            self._write(self.reg_wire["speed"], raw)
            self.measure_done(True)
            self.set_bit(self.bits_st["active"], False, flush=False)
            self._flush_locked(force=True)
        print(
            f"[Modbus] SPEED HR{self.reg_plc['speed']}={raw} "
            f"({self.format_speed_mm_s(raw, self.speed_scale)} mm/s), DONE bit3"
        )

    def read_cmd(self):
        val, _ok = self.read_cmd_status()
        return val

    def read_cmd_status(self):
        if self._async_enabled and self._poll_thread is not None and self._poll_thread.is_alive():
            with self._state_lock:
                return int(self._cmd_cache) & 0xFFFF, bool(self._cmd_ok_cache)
        val, ok = self._read_reg(self.client, self.reg_wire["cmd"], 0)
        self._log_cmd_if_changed(val, ok)
        return int(val) & 0xFFFF, ok

    def _log_cmd_if_changed(self, val, ok):
        if self.cmd_log_on_change and ok and int(val) != self._cmd_log_last:
            self._cmd_log_last = int(val) & 0xFFFF
            print(
                f"[Modbus] CMD HR wire={self.reg_wire['cmd']} "
                f"plc={self.reg_plc['cmd']} {self.format_cmd(val)}"
            )

    def format_cmd(self, cmd):
        c = int(cmd) & 0xFFFF
        parts = []
        if c & (1 << self.bits_cmd["reset"]):
            parts.append("RESET")
        if c & (1 << self.bits_cmd.get("search", 1)):
            parts.append("SEARCH")
        if c & (1 << self.bits_cmd["start"]):
            parts.append("START")
        if c & (1 << self.bits_cmd["dist_ok"]):
            parts.append("DIST_OK")
        return f"raw={c} " + ("+".join(parts) if parts else "(нет битов)")

    def read_dist_mm(self):
        return self._read(self.reg_wire["dist_mm"])

    def read_dist_mm_status(self):
        if self._async_enabled and self._poll_thread is not None and self._poll_thread.is_alive():
            with self._state_lock:
                return int(self._dist_cache), bool(self._cmd_ok_cache)
        return self._read_reg(self.client, self.reg_wire["dist_mm"], 0)

    def _poll_io_cycle(self):
        """Чтение CMD + STATUS/heartbeat (один сокет — под lock)."""
        with self._state_lock:
            cmd, dist, ok = self._read_cmd_dist_regs(self._read_client)
            self._cmd_cache = cmd
            self._dist_cache = dist
            self._cmd_ok_cache = ok
            self._log_cmd_if_changed(cmd, ok)
            now_ms = time.time() * 1000.0
            changed = self._tick_status_locked(now_ms)
            if changed or self._status != self._written:
                self._flush_locked()

    def _poll_worker(self):
        interval = max(0.05, self.poll_interval_ms / 1000.0)
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                self._poll_io_cycle()
            except Exception as exc:
                print(f"[Modbus] poll: {exc}")
            wait_s = interval - (time.perf_counter() - t0)
            if wait_s > 0:
                self._stop.wait(wait_s)

    def start_async_poll(self, interval_ms=None):
        if not self._async_enabled or not self._ok:
            return
        if interval_ms is not None:
            self.poll_interval_ms = max(50, int(interval_ms))
        if self._poll_thread is not None and self._poll_thread.is_alive():
            return
        self._stop.clear()
        self._poll_io_cycle()
        self._poll_thread = threading.Thread(
            target=self._poll_worker,
            name="modbus-poll",
            daemon=True,
        )
        self._poll_thread.start()
        print(
            f"[Modbus] async poll {self.poll_interval_ms}ms "
            f"(timeout={self.timeout}s, read/write раздельно)"
        )

    def stop_async_poll(self):
        self._stop.set()
        th = self._poll_thread
        if th is not None and th.is_alive():
            th.join(timeout=2.0)
        self._poll_thread = None

    def start_measure_edge(self, cmd):
        if self.errors_enabled and self._err != ERR_NONE:
            return False
        st = self.bits_cmd["start"]
        if not self._start_allowed:
            if not self.has_bit(cmd, st):
                self._start_allowed = True
                self._start_level_seen = False
            return False
        if self.start_on_level:
            if self.has_bit(cmd, st):
                if self._start_level_seen:
                    return False
                self._start_level_seen = True
                return True
            self._start_level_seen = False
            return False
        if not self.cmd_rising(cmd, "start"):
            return False
        return True

    def dist_ok(self, cmd):
        return self.has_bit(cmd, self.bits_cmd["dist_ok"])

    def has_start(self, cmd):
        return self.has_bit(cmd, self.bits_cmd["start"])

    def has_search(self, cmd):
        return self.has_bit(cmd, self.bits_cmd.get("search", 1))

    def search_cmd_allowed(self, cmd):
        if self.errors_enabled and self._err != ERR_NONE:
            return False
        return self.has_search(cmd)

    def reset_errors(self, cmd):
        b = self.bits_cmd["reset"]
        if not self.has_bit(cmd, b):
            self._reset_latched = False
            return False
        if self._err != ERR_NONE:
            if self._reset_latched:
                return False
            self._reset_latched = True
        elif not self.cmd_rising(cmd, "reset"):
            return False
        err_was = self._err
        if err_was:
            print(f"[Modbus] RESET (cmd={int(cmd) & 0xFFFF}, было ERR {err_was})")
        else:
            print(f"[Modbus] RESET (cmd={int(cmd) & 0xFFFF})")
        self.clear_error()
        return True

    @property
    def error_code(self):
        return self._err if self.errors_enabled else ERR_NONE

    @property
    def status(self):
        return self._status

    @property
    def reg(self):
        return self.reg_plc

    def close(self):
        self.stop_async_poll()
        with self._state_lock:
            self.camera_ready(False)
            if "alive" in self.bits_st:
                self._bit(self.bits_st["alive"], False)
                self.flush(force=True)
            try:
                self.client.close()
            except Exception:
                pass
            if self._read_client is not self.client:
                try:
                    self._read_client.close()
                except Exception:
                    pass
