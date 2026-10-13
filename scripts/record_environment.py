"""Print reproducibility metadata as JSON; does not install packages."""
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def command(args):
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None

def version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None

devices = {"cuda_available": None, "mps_available": None}
torch_error = None
if version("torch"):
    try:
        import torch
        devices["cuda_available"] = torch.cuda.is_available()
        devices["mps_available"] = bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        devices["cuda_runtime"] = torch.version.cuda
    except Exception as error:
        torch_error = str(error)

report = {
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "python": platform.python_version(),
    "inside_virtual_environment": sys.prefix != sys.base_prefix,
    "os": platform.system(),
    "os_release": platform.release(),
    "macos_version": platform.mac_ver()[0] or None,
    "linux_os_release": command(["cat", "/etc/os-release"]) if platform.system() == "Linux" else None,
    "architecture": platform.machine(),
    "cpu": command(["sysctl", "-n", "machdep.cpu.brand_string"]) if platform.system() == "Darwin" else platform.processor(),
    "packages": {name: version(name) for name in ["torch", "torchvision", "ultralytics", "numpy", "opencv-python"]},
    "available_devices": devices,
    "actual_training_device": None,
    "torch_import_error": torch_error,
    "git_commit": command(["git", "rev-parse", "HEAD"]),
    "git_dirty": bool(command(["git", "status", "--porcelain"])),
    "note": "null means not installed, unknown, or not yet selected; availability does not prove the device was used.",
}
print(json.dumps(report, ensure_ascii=False, indent=2))
