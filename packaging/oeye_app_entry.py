from __future__ import annotations

import importlib.machinery
import importlib.util
import runpy
import sys
from pathlib import Path


def _resource_root() -> Path:
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        return Path(frozen_root)
    script_dir = Path(__file__).resolve().parent
    if (script_dir / "protected_sources").exists() or (script_dir / "requirements-standard.txt").exists():
        return script_dir
    return script_dir.parents[0]


def _run_protected_app() -> None:
    root = _resource_root()
    protected_root = root / "protected_sources"
    app_pyc = protected_root / "app.pyc"

    if protected_root.exists():
        sys.path.insert(0, str(protected_root))

    if app_pyc.exists():
        loader = importlib.machinery.SourcelessFileLoader("oeye_protected_app", str(app_pyc))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        if spec is None:
            raise RuntimeError("无法加载 OEye 受保护字节码入口。")
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
        return

    dev_app = root / "app.py"
    if dev_app.exists():
        sys.path.insert(0, str(root))
        runpy.run_path(str(dev_app), run_name="__main__")
        return

    raise FileNotFoundError("未找到受保护 app.pyc 或开发版 app.py。")


_run_protected_app()
