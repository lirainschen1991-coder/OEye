from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import shutil
from dataclasses import dataclass
from enum import Enum
from importlib import metadata
from pathlib import Path
from typing import Iterable

from packaging.markers import default_environment
from packaging.requirements import Requirement


APP_NAME = "OEye"
CONFIG_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / APP_NAME
CONFIG_PATH = CONFIG_DIR / "dependency_config.json"
LOG_DIR = CONFIG_DIR / "logs"


IMPORT_NAME_OVERRIDES = {
    "scikit-learn": "sklearn",
    "stable-baselines3": "stable_baselines3",
    "pillow": "PIL",
    "opencv-python": "cv2",
    "pyyaml": "yaml",
}


ESTIMATES_MB = {
    "streamlit": (35, 120),
    "pandas": (15, 120),
    "numpy": (20, 100),
    "scikit-learn": (12, 90),
    "plotly": (8, 40),
    "seaborn": (1, 8),
    "matplotlib": (12, 55),
    "scipy": (35, 150),
    "xgboost": (120, 450),
    "lightgbm": (40, 160),
    "shap": (4, 25),
    "lime": (1, 5),
    "tensorflow": (500, 1700),
    "torch": (900, 3200),
    "catboost": (90, 350),
    "gymnasium": (3, 15),
    "stable-baselines3": (3, 20),
    "ray": (120, 480),
}


class DependencyStatus(str, Enum):
    SATISFIED = "satisfied"
    MISSING = "missing"
    OUTDATED = "outdated"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class DependencyCheck:
    requirement_line: str
    package_name: str
    display_name: str
    import_name: str
    specifier: str
    status: DependencyStatus
    installed_version: str | None
    profile: str
    optional: bool
    supported: bool
    reason: str
    estimated_download_mb: int
    estimated_install_mb: int

    @property
    def needs_install(self) -> bool:
        return self.status in {DependencyStatus.MISSING, DependencyStatus.OUTDATED} and self.supported

    @property
    def install_requirement(self) -> str:
        spec = self.requirement_line.split(";", 1)[0].strip()
        return spec


def project_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    if (script_dir / "requirements-standard.txt").exists() or (script_dir / "protected_sources").exists():
        return script_dir
    return script_dir.parents[0]


def log_dir() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR


def subprocess_creationflags() -> int:
    if os.name == "nt":
        return getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return 0


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_config(config: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def managed_venv_dir() -> Path:
    return CONFIG_DIR / "runtime" / ".venv"


def managed_python() -> Path:
    if platform.system().lower() == "windows":
        return managed_venv_dir() / "Scripts" / "python.exe"
    return managed_venv_dir() / "bin" / "python"


def _resolve_python_command(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            [*command, "-c", "import sys; print(sys.executable)"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            creationflags=subprocess_creationflags(),
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value if value else None


def discover_python_candidates() -> list[str]:
    candidates: list[str] = []
    commands: list[list[str]] = []

    if not getattr(sys, "frozen", False):
        commands.append([sys.executable])

    py_launcher = shutil.which("py")
    if py_launcher:
        commands.extend([[py_launcher, "-3.12"], [py_launcher, "-3.11"], [py_launcher, "-3.13"], [py_launcher, "-3"]])

    for name in ("python", "python3"):
        found = shutil.which(name)
        if found:
            commands.append([found])

    for command in commands:
        resolved = _resolve_python_command(command)
        if resolved and resolved not in candidates:
            candidates.append(resolved)
    return candidates


def default_base_python() -> str:
    candidates = discover_python_candidates()
    return candidates[0] if candidates else ""


def parse_requirements_file(path: Path, profile: str) -> list[Requirement]:
    requirements: list[Requirement] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(Requirement(line))
    return requirements


def requirement_line(req: Requirement) -> str:
    extras = f"[{','.join(sorted(req.extras))}]" if req.extras else ""
    marker = f"; {req.marker}" if req.marker else ""
    return f"{req.name}{extras}{req.specifier}{marker}"


def import_name_for(package_name: str) -> str:
    normalized = package_name.lower()
    return IMPORT_NAME_OVERRIDES.get(normalized, normalized.replace("-", "_"))


def requirement_supported(req: Requirement, env: dict | None = None) -> tuple[bool, str]:
    if req.marker is None:
        return True, ""
    marker_env = default_environment()
    if env:
        marker_env.update(env)
    supported = bool(req.marker.evaluate(marker_env))
    if supported:
        return True, ""
    return False, f"当前环境不满足依赖标记：{req.marker}"


def _version_in_environment(package_name: str, python_executable: str | Path | None = None) -> str | None:
    if python_executable is None or Path(python_executable).resolve() == Path(sys.executable).resolve():
        try:
            return metadata.version(package_name)
        except metadata.PackageNotFoundError:
            return None

    code = (
        "from importlib import metadata; import sys\n"
        "pkg=sys.argv[1]\n"
        "try:\n"
        "    print(metadata.version(pkg))\n"
        "except metadata.PackageNotFoundError:\n"
        "    sys.exit(2)\n"
    )
    try:
        completed = subprocess.run(
            [str(python_executable), "-c", code, package_name],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
            creationflags=subprocess_creationflags(),
        )
    except Exception:
        return None
    if completed.returncode == 0:
        return completed.stdout.strip() or None
    return None


def check_requirement(
    req: Requirement,
    profile: str,
    python_executable: str | Path | None = None,
    marker_env: dict | None = None,
) -> DependencyCheck:
    supported, unsupported_reason = requirement_supported(req, marker_env)
    package_name = req.name
    import_name = import_name_for(package_name)
    download_mb, install_mb = ESTIMATES_MB.get(package_name.lower(), (10, 40))

    if not supported:
        return DependencyCheck(
            requirement_line=requirement_line(req),
            package_name=package_name,
            display_name=package_name,
            import_name=import_name,
            specifier=str(req.specifier),
            status=DependencyStatus.UNSUPPORTED,
            installed_version=None,
            profile=profile,
            optional=True,
            supported=False,
            reason=unsupported_reason,
            estimated_download_mb=0,
            estimated_install_mb=0,
        )

    installed_version = _version_in_environment(package_name, python_executable)
    if installed_version is None:
        status = DependencyStatus.MISSING
        reason = "未安装"
    elif req.specifier and installed_version not in req.specifier:
        status = DependencyStatus.OUTDATED
        reason = f"当前版本 {installed_version} 不满足 {req.specifier}"
    else:
        status = DependencyStatus.SATISFIED
        reason = "已满足"

    return DependencyCheck(
        requirement_line=requirement_line(req),
        package_name=package_name,
        display_name=package_name,
        import_name=import_name,
        specifier=str(req.specifier),
        status=status,
        installed_version=installed_version,
        profile=profile,
        optional=False,
        supported=True,
        reason=reason,
        estimated_download_mb=download_mb if status != DependencyStatus.SATISFIED else 0,
        estimated_install_mb=install_mb if status != DependencyStatus.SATISFIED else 0,
    )


def load_profile_requirements(profile: str, root: Path | None = None) -> list[Requirement]:
    root = root or project_root()
    if profile == "standard":
        return parse_requirements_file(root / "requirements-standard.txt", profile)
    if profile == "runtime":
        return parse_requirements_file(root / "requirements-runtime.txt", profile)
    raise ValueError(f"未知依赖档位：{profile}")


def check_profile(
    profile: str,
    root: Path | None = None,
    python_executable: str | Path | None = None,
    marker_env: dict | None = None,
) -> list[DependencyCheck]:
    return [
        check_requirement(req, profile, python_executable=python_executable, marker_env=marker_env)
        for req in load_profile_requirements(profile, root=root)
    ]


def install_list(checks: Iterable[DependencyCheck]) -> list[str]:
    seen: set[str] = set()
    requirements: list[str] = []
    for check in checks:
        if not check.needs_install:
            continue
        normalized = check.package_name.lower()
        if normalized in seen:
            continue
        requirements.append(check.install_requirement)
        seen.add(normalized)
    return requirements


def estimate_missing(checks: Iterable[DependencyCheck]) -> tuple[int, int]:
    missing = [check for check in checks if check.needs_install]
    return (
        sum(check.estimated_download_mb for check in missing),
        sum(check.estimated_install_mb for check in missing),
    )


def build_pip_install_command(
    python_executable: str | Path,
    requirements: Iterable[str],
    index_url: str | None = None,
) -> list[str]:
    reqs = [req for req in requirements if req.strip()]
    if not reqs:
        return []
    command = [
        str(python_executable),
        "-m",
        "pip",
        "install",
        "--upgrade-strategy",
        "only-if-needed",
    ]
    if index_url:
        command.extend(["-i", index_url.strip()])
    command.extend(reqs)
    return command


def create_managed_venv(base_python: str | Path | None = None) -> Path:
    target_python = managed_python()
    if target_python.exists():
        return target_python

    base = str(base_python or default_base_python() or sys.executable)
    if getattr(sys, "frozen", False) and Path(base).resolve() == Path(sys.executable).resolve():
        raise RuntimeError("未找到可用于创建 venv 的系统 Python。请先安装 Python 3.11/3.12，或选择一个已有 Python。")
    managed_venv_dir().parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([base, "-m", "venv", str(managed_venv_dir())], check=True, creationflags=subprocess_creationflags())
    return target_python


def run_install_command(command: list[str], log_file: Path | None = None) -> int:
    if not command:
        return 0
    log_file = log_file or (log_dir() / "dependency-install.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write("\n\n$ " + " ".join(command) + "\n")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=subprocess_creationflags(),
        )
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
            handle.flush()
        return int(process.wait())
