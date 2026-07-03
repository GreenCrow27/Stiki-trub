# Stiki-trub (pipe_vision)

Измерение скорости трубы: RTSP-камера, детекция стыка (лазер / bg_diff), обмен с ПЛК по Modbus TCP.

## Быстрый старт

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate   # Linux
pip install -r requirements.txt
copy config.example.json config.json   # отредактируйте IP, пароль RTSP, Modbus
python main.py
```

Сборка exe: см. [BUILD.md](BUILD.md).

Настройки в runtime: `NASTROYKI.txt`, коды ошибок: `ERRORS.txt`.
