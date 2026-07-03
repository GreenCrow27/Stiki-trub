"""Пути к файлам: исходники и собранный PyInstaller exe."""
import os
import sys


def app_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def resource_path(rel: str) -> str:
    """Чтение: рядом с exe, иначе из bundle (_MEIPASS)."""
    rel = rel.replace("\\", "/")
    local = os.path.join(app_dir(), rel)
    if os.path.isfile(local):
        return local
    root = getattr(sys, "_MEIPASS", None)
    if root:
        bundled = os.path.join(root, rel)
        if os.path.isfile(bundled):
            return bundled
    return local


def writable_path(rel: str) -> str:
    """Запись (config после P): всегда каталог exe."""
    return os.path.join(app_dir(), rel.replace("\\", "/"))
