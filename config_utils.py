import json

from paths import resource_path, writable_path


def load_config(path="config.json"):
    with open(resource_path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(cfg, path="config.json"):
    with open(writable_path(path), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)
