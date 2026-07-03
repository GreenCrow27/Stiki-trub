# Stiki-trub (pipe_vision)

Измерение скорости трубы: RTSP-камера, детекция стыка (лазер / bg_diff), обмен с ПЛК по Modbus TCP.

## Быстрый старт

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate   # Linux
pip install -r requirements.txt
python main.py
```

`config.json` — рабочий конфиг (IP камеры, Modbus, ROI).  
`config.example.json` — шаблон без паролей.

Сборка exe: см. [BUILD.md](BUILD.md) (`dist/` в git не кладём — ~240 МБ, собирается локально).

Настройки в runtime: `NASTROYKI.txt`, коды ошибок: `ERRORS.txt`.
