"""Refresh the package payload inside the Kaggle notebook install cell."""

from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "HaemoDiff_Risk_Kaggle.ipynb"
SKIP_DIRS = {".git", ".pytest_cache", "runs", "__pycache__", ".ipynb_checkpoints", "notebooks", "scripts"}


def build_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in ROOT.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT)
            if any(part in SKIP_DIRS for part in relative.parts):
                continue
            if path.suffix in {".pyc", ".pt", ".npy"}:
                continue
            archive.write(path, Path("medAIHemo") / relative)
    return buffer.getvalue()


CELL = """from pathlib import Path
import base64, io, os, shutil, subprocess, sys, zipfile

def is_repo(path):
    return (path / "pyproject.toml").is_file() and (path / "src" / "haemodiff").is_dir()

def find_repo(base):
    if not base.exists():
        return None
    if is_repo(base):
        return base
    skip = {"Videos", "videos", ".git", "__pycache__", ".pytest_cache", "runs"}
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [name for name in dirnames if name not in skip]
        if "pyproject.toml" in filenames and is_repo(Path(dirpath)):
            return Path(dirpath)
    return None

# Cursor sends this cell to the remote Kaggle kernel. The package is inside the cell
# because the kernel cannot see the project folder on the local PC.
BUNDLE = '''__BUNDLE__'''

found = None
for base in [Path("/kaggle/working/medAIHemo"), Path.cwd(), *Path.cwd().parents, Path("/kaggle/working"), Path("/kaggle/input")]:
    found = find_repo(base)
    if found is not None:
        break

if found is None:
    parent = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))) as archive:
        archive.extractall(parent)
    found = parent / "medAIHemo"
    if not is_repo(found):
        found = find_repo(parent)
if found is None or not is_repo(found):
    raise FileNotFoundError("The embedded package did not unpack to pyproject.toml and src/haemodiff.")

root = found
if Path("/kaggle/working").exists():
    destination = Path("/kaggle/working/medAIHemo")
    if found.resolve() != destination.resolve():
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(found, destination, ignore=shutil.ignore_patterns(".git", ".pytest_cache", "runs", "__pycache__", "*.pyc"))
        root = destination

os.chdir(root)
sys.path.insert(0, str(root / "src"))
print("repo", root)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "opacus", "pyyaml", "scipy", "--upgrade-strategy", "only-if-needed"])
"""


def main() -> None:
    payload = base64.b64encode(build_zip()).decode("ascii")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = CELL.replace("__BUNDLE__", payload)
    notebook["cells"][2]["source"] = [line + "\n" for line in source.splitlines()]
    notebook["cells"][2]["outputs"] = []
    notebook["cells"][2]["execution_count"] = None
    NOTEBOOK.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(f"embedded {len(payload)} base64 chars into {NOTEBOOK}")


if __name__ == "__main__":
    main()
