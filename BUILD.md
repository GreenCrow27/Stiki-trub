# Сборка pipe_vision (PyInstaller)

## Быстрый старт

| ОС | Команда |
|----|---------|
| **Windows** | `build.bat` или `.\build.ps1` |
| **Linux** | `chmod +x build.sh && ./build.sh` |
| **Любая** | `python build_project.py` |

Результат: `dist/pipe_vision/`  
- Windows: `pipe_vision.exe`  
- Linux: `pipe_vision`  

Рядом копируются `config.json`, `NASTROYKI.txt` (и `charuco_calibration.npz`, если есть).

## Опции

```text
python build_project.py --no-deps    # без pip install
python build_project.py --no-clean   # не удалять build/ dist/
```

## Linux: системные библиотеки

```bash
sudo apt-get install -y python3-venv ffmpeg libgl1 libglib2.0-0 libxcb-cursor0
```

## Python на старом Ubuntu (3.7 / 18.04)

`build.sh` сам ищет `python3.10` … `python3.7` и создаёт `.venv`.

```bash
chmod +x build.sh
./build.sh
```

Для **Python 3.7** ставятся старые wheel: numpy 1.21, opencv 4.5, pymodbus 2.x, PyInstaller 5.x  
(см. маркеры в `requirements.txt`).

**Рекомендуется Python 3.8+** (Ubuntu 20.04+ или `sudo apt install python3.8 python3.8-venv`).

```bash
python3.8 -m venv .venv
source .venv/bin/activate
./build.sh --no-deps   # venv уже с нужным Python
python build_project.py
```

## Linux: авто-apt

```bash
PIPE_VISION_APT=1 ./build.sh
```

## Зависимости разработки

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --prefer-binary
```

| Python | numpy | pymodbus | PyInstaller |
|--------|-------|----------|-------------|
| 3.7 | 1.19–1.21 | 2.5.x | 5.13 |
| 3.8 | 1.24 | 3.6+ | 6.14+ |
| 3.9–3.11 | 1.26 | 3.6+ | 6.14+ |
| 3.12+ | 2.x | 3.6+ | 6.14+ |

**Python 3.12+:** не используйте `numpy==1.24` — нет wheel, ошибка `ImpImporter`.

## Запуск

- Исходники: `python main.py`, `python main_laser.py`
- Сборка: `cd dist/pipe_vision && ./pipe_vision` (или `.exe` на Windows)
- RTSP: **ffmpeg** в PATH
- Настройки: править `config.json` рядом с exe; справка — `NASTROYKI.txt`

## Структура

| Файл | Назначение |
|------|------------|
| `build_project.py` | Общая логика сборки |
| `build.bat` | Обёртка Windows |
| `build.ps1` | PowerShell → build_project.py |
| `build.sh` | Linux → build_project.py |
| `pipe_vision.spec` | Конфиг PyInstaller |
