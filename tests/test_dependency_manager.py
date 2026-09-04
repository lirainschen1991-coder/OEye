from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "packaging" / "dependency_manager.py"


def load_dependency_manager():
    spec = importlib.util.spec_from_file_location("oeye_dependency_manager", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_requirements_profiles_parse():
    dm = load_dependency_manager()

    standard = dm.load_profile_requirements("standard", root=ROOT)
    runtime = dm.load_profile_requirements("runtime", root=ROOT)

    assert any(req.name == "scikit-learn" for req in standard)
    assert any(req.name == "stable-baselines3" for req in runtime)
    assert any(req.name == "ray" and req.extras == {"rllib"} for req in runtime)


def test_install_list_only_contains_missing_or_outdated(monkeypatch):
    dm = load_dependency_manager()
    reqs = dm.load_profile_requirements("standard", root=ROOT)

    def fake_version(package_name, python_executable=None):
        versions = {
            "streamlit": "1.40.0",
            "pandas": None,
            "numpy": "1.20.0",
            "scikit-learn": "1.5.0",
            "plotly": "5.20.0",
            "seaborn": "0.13.0",
            "matplotlib": "3.8.0",
            "scipy": "1.12.0",
            "xgboost": "2.1.0",
            "lightgbm": "4.3.0",
            "shap": "0.45.0",
            "lime": "0.2.0.1",
        }
        return versions.get(package_name)

    monkeypatch.setattr(dm, "_version_in_environment", fake_version)
    checks = [dm.check_requirement(req, "standard") for req in reqs]
    install = dm.install_list(checks)

    assert any(item.startswith("pandas>=") for item in install)
    assert any(item.startswith("numpy>=") for item in install)
    assert not any(item.startswith("streamlit") for item in install)
    assert not any(item.startswith("scikit-learn") for item in install)


def test_ray_rllib_is_unsupported_on_python_313(monkeypatch):
    dm = load_dependency_manager()
    ray_req = next(req for req in dm.load_profile_requirements("runtime", root=ROOT) if req.name == "ray")

    monkeypatch.setattr(dm, "_version_in_environment", lambda package_name, python_executable=None: None)
    check = dm.check_requirement(ray_req, "runtime", marker_env={"python_version": "3.13"})

    assert check.status == dm.DependencyStatus.UNSUPPORTED
    assert check.needs_install is False


def test_pip_command_uses_exact_missing_list_and_cache_defaults():
    dm = load_dependency_manager()

    command = dm.build_pip_install_command(
        "python",
        ["numpy>=1.25.0", "stable-baselines3>=2.3.0"],
        index_url="https://pypi.tuna.tsinghua.edu.cn/simple",
    )

    assert command[:4] == ["python", "-m", "pip", "install"]
    assert "--upgrade-strategy" in command
    assert "only-if-needed" in command
    assert "--no-cache-dir" not in command
    assert "numpy>=1.25.0" in command
    assert "stable-baselines3>=2.3.0" in command
