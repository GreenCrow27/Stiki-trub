"""
Сборка pipe_vision (PyInstaller) — Windows и Linux.
Поддержка Python 3.7–3.14 (зависимости подбираются по версии).

Запуск:
  python build_project.py
  build.bat          (Windows)
  ./build.sh         (Linux)
"""
from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent
SPEC = ROOT / "pipe_vision.spec"
DIST_NAME = "pipe_vision"
MIN_PYTHON = (3, 7)

BUNDLE_DATA = (
    "config.json",
    "NASTROYKI.txt",
)


def _venv_python() -> Optional[Path]:
    if platform.system() == "Windows":
        p = ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        p = ROOT / ".venv" / "bin" / "python"
    return p if p.is_file() else None


def _run(cmd: List[str], **kwargs) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True, **kwargs)


def _py_tag() -> str:
    v = sys.version_info
    return "{}.{}.{}".format(v.major, v.minor, v.micro)


def _python_version_tuple(python: str) -> tuple:
    out = subprocess.check_output(
        [python, "-c", "import sys; print(sys.version_info[0], sys.version_info[1])"],
        text=True,
    )
    parts = out.strip().split()
    return int(parts[0]), int(parts[1])


def _check_python(ver: tuple) -> None:
    if ver < MIN_PYTHON:
        raise RuntimeError(
            "Нужен Python {}.{}+, сейчас {}.{}".format(
                MIN_PYTHON[0], MIN_PYTHON[1], ver[0], ver[1]
            )
        )
    if ver < (3, 8):
        print(
            "Внимание: Python 3.7 — pymodbus 2.x, старые numpy/opencv. "
            "Рекомендуется Python 3.8+ (Ubuntu: sudo apt install python3.8 python3.8-venv)."
        )


def install_deps(python: str, py_ver: tuple) -> None:
    """pip + wheel; пакеты только бинарники где возможно."""
    if py_ver < (3, 8):
        boot = [python, "-m", "pip", "install", "--upgrade", "pip<24", "setuptools<70", "wheel"]
    else:
        boot = [python, "-m", "pip", "install", "--upgrade", "pip", "setuptools>=69", "wheel"]
    _run(boot)
    _run(
        [
            python,
            "-m",
            "pip",
            "install",
            "-r",
            "requirements.txt",
            "--upgrade",
            "--prefer-binary",
        ]
    )
    subprocess.run(
        [python, "-m", "pip", "check"],
        cwd=str(ROOT),
        check=False,
    )


def clean_artifacts() -> None:
    for name in ("build", "dist"):
        path = ROOT / name
        if path.is_dir():
            print("remove {}".format(path))
            shutil.rmtree(path)


def copy_runtime_files(dist_dir: Path) -> List[str]:
    copied = []
    for name in BUNDLE_DATA:
        src = ROOT / name
        if not src.is_file():
            continue
        dst = dist_dir / name
        shutil.copy2(str(src), str(dst))
        copied.append(name)
    return copied


def build(
    *,
    clean: bool = True,
    install: bool = True,
    spec: Optional[Path] = None,
) -> Path:
    spec = spec or SPEC
    if not spec.is_file():
        raise FileNotFoundError("Нет spec-файла: {}".format(spec))

    venv_py = _venv_python()
    python = str(venv_py) if venv_py else sys.executable
    if venv_py:
        print("venv: {}".format(venv_py.parent.parent))

    print("=== pipe_vision build ({}) ===".format(platform.system()))
    print("python: {} ({})".format(python, _py_tag()))
    py_ver = _python_version_tuple(python)
    print("target: {}.{}".format(py_ver[0], py_ver[1]))
    _check_python(py_ver)

    if install:
        install_deps(python, py_ver)

    if clean:
        clean_artifacts()

    _run([python, "-m", "PyInstaller", str(spec), "--noconfirm"])

    dist_dir = ROOT / "dist" / DIST_NAME
    if platform.system() == "Windows":
        exe = dist_dir / "pipe_vision.exe"
    else:
        exe = dist_dir / "pipe_vision"

    if not exe.is_file():
        raise RuntimeError("Сборка не создала бинарник: {}".format(exe))

    copied = copy_runtime_files(dist_dir)
    print()
    print("OK: {}".format(exe))
    if copied:
        print("Рядом:", ", ".join(copied))
    print("RTSP: нужен ffmpeg в PATH.")
    return dist_dir


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Сборка pipe_vision (PyInstaller)")
    parser.add_argument("--no-clean", action="store_true", help="Не удалять build/dist")
    parser.add_argument("--no-deps", action="store_true", help="Не ставить pip-зависимости")
    parser.add_argument("--spec", type=Path, default=SPEC, help="Файл .spec")
    args = parser.parse_args(argv)
    try:
        build(clean=not args.no_clean, install=not args.no_deps, spec=args.spec)
    except subprocess.CalledProcessError as e:
        print("Ошибка сборки (код {})".format(e.returncode), file=sys.stderr)
        if sys.version_info >= (3, 12):
            print(
                "Подсказка: для Py3.12+ нужен numpy>=2.0 (см. requirements.txt).",
                file=sys.stderr,
            )
        elif sys.version_info < (3, 8):
            print(
                "Подсказка для Py3.7: python3.8-venv или "
                "pip install -r requirements.txt --prefer-binary",
                file=sys.stderr,
            )
        return e.returncode or 1
    except Exception as e:
        print("Ошибка: {}".format(e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
