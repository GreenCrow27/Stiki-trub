#!/usr/bin/env bash
# Сборка pipe_vision (Linux)
# chmod +x build.sh && ./build.sh

set -euo pipefail
cd "$(dirname "$0")"

echo "=== pipe_vision build (Linux) ==="

# Выбор Python: 3.10 → 3.8 → 3.7 → python3
pick_python() {
  local cmd ver maj min
  for cmd in python3.12 python3.11 python3.10 python3.9 python3.8 python3.7 python3; do
    if command -v "$cmd" >/dev/null 2>&1; then
      ver=$("$cmd" -c 'import sys; print(sys.version_info[0], sys.version_info[1])')
      maj=${ver%% *}
      min=${ver##* }
      if [ "$maj" -eq 3 ] && [ "$min" -ge 7 ]; then
        echo "$cmd"
        return 0
      fi
    fi
  done
  return 1
}

PY=$(pick_python) || {
  echo "Не найден Python 3.7+. Установите: sudo apt install python3 python3-venv"
  exit 1
}

echo "Используем: $PY ($($PY --version))"

if command -v apt-get >/dev/null 2>&1; then
  echo "Системные пакеты (при необходимости):"
  echo "  sudo apt-get install -y python3-venv ffmpeg libgl1 libglib2.0-0"
  if [ "${PIPE_VISION_APT:-0}" = "1" ]; then
    sudo apt-get update
    sudo apt-get install -y \
      python3-venv ffmpeg libgl1 libglib2.0-0 \
      libxcb-cursor0 build-essential || true
  fi
fi

if [ ! -d ".venv" ]; then
  echo "Создаём .venv ..."
  "$PY" -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate

exec python build_project.py "$@"
