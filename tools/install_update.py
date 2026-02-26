"""
Install & Update tool: git hard reset to origin, then pip install into embedded Python,
then verify imports. Uses GitPython for git (no subprocess), subprocess only for
invoking embedded python -m pip and -c "import ...".
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Callable, Optional, Union


def _default_python_exe(repo_root: Path) -> Path:
    return repo_root.parent / "python_embeded" / "python.exe"


def _default_git_exe(repo_root: Path) -> Path:
    return repo_root.parent / "git_portable" / "mingw64" / "bin" / "git.exe"


def _log(lines: list[str], msg: str, log_fn: Optional[Callable[[str], None]]) -> None:
    line = msg.rstrip()
    lines.append(line)
    if log_fn:
        log_fn(line)


def _run(
    args: list[str],
    cwd: Optional[Path],
    lines: list[str],
    log_fn: Optional[Callable[[str], None]],
    env: Optional[dict] = None,
) -> int:
    """Run command, stream stdout/stderr to log, return returncode."""
    _log(lines, f"> {' '.join(args)}", log_fn)
    r = subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
    )
    if r.stdout:
        for l in r.stdout.rstrip().splitlines():
            _log(lines, l, log_fn)
    if r.stderr:
        for l in r.stderr.rstrip().splitlines():
            _log(lines, l, log_fn)
    return r.returncode


def run_install_update(
    repo_root: Path,
    python_exe: Optional[Union[Path, str]] = None,
    git_exe: Optional[Union[Path, str]] = None,
    cuda_ver: str = "cu121",
    log_fn: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Run: 1) git fetch + reset --hard to origin, 2) pip install into embedded Python,
    3) verify imports. All packages go into embedded Python (no venv).
    Uses GitPython for git (portable git by default); subprocess only for python_exe -m pip and -c.

    Args:
        repo_root: Repo directory (contains .git and requirements.txt).
        python_exe: Path to embedded python.exe; if None, uses repo_root.parent / "python_embeded" / "python.exe".
        git_exe: Path to git.exe (portable); if None, uses repo_root.parent / "git_portable" / "mingw64" / "bin" / "git.exe".
        cuda_ver: PyTorch index suffix (cu118, cu121, cu124).
        log_fn: Optional callback for each log line.

    Returns:
        Full log as single string.
    """
    repo_root = Path(repo_root).resolve()
    py_exe = Path(python_exe) if python_exe else _default_python_exe(repo_root)
    git_bin = Path(git_exe) if git_exe else _default_git_exe(repo_root)
    lines: list[str] = []

    def log(msg: str) -> None:
        _log(lines, msg, log_fn)

    # Check embedded Python exists
    if not py_exe.is_file():
        log(f"Error: Python executable not found: {py_exe}")
        return "\n".join(lines)

    # --- Step 1: Git hard reset (GitPython, portable git) ---
    log("--- Git: fetch + reset --hard (portable git) ---")
    if git_bin.is_file():
        os.environ["GIT_PYTHON_GIT_EXECUTABLE"] = str(git_bin)
    else:
        log(f"Warning: portable git not found at {git_bin}; GitPython may use system git")
    try:
        import git
    except ImportError:
        log("Warning: GitPython not installed; skipping git step. pip install GitPython")
    else:
        git_dir = repo_root / ".git"
        if not git_dir.exists():
            log("Not a git repo (.git missing); skipping git step.")
        else:
            try:
                repo = git.Repo(repo_root)
                origin = repo.remotes.origin
                origin.fetch()
                branch = repo.active_branch.name
                ref = f"origin/{branch}"
                repo.git.reset("--hard", ref)
                log(f"Reset --hard {ref} OK")
            except Exception as e:
                log(f"Git error: {e}")
    log("")

    # --- Step 2: pip (always via python -m pip) ---
    log("--- pip: upgrade pip ---")
    rc = _run(
        [str(py_exe), "-m", "pip", "install", "--upgrade", "pip"],
        cwd=repo_root,
        lines=lines,
        log_fn=log_fn,
    )
    if rc != 0:
        log(f"pip upgrade returned {rc}")
    log("")

    log("--- pip: PyTorch (CUDA) ---")
    rc = _run(
        [
            str(py_exe), "-m", "pip", "install", "torch", "torchvision",
            "--index-url", f"https://download.pytorch.org/whl/{cuda_ver}",
        ],
        cwd=repo_root,
        lines=lines,
        log_fn=log_fn,
    )
    if rc != 0:
        log(f"PyTorch install returned {rc}")
    log("")

    log("--- pip: -r requirements.txt ---")
    req_file = repo_root / "requirements.txt"
    if not req_file.is_file():
        log(f"Warning: {req_file} not found")
    else:
        rc = _run(
            [str(py_exe), "-m", "pip", "install", "-r", str(req_file)],
            cwd=repo_root,
            lines=lines,
            log_fn=log_fn,
        )
        if rc != 0:
            log(f"pip -r requirements.txt returned {rc}")
    log("")

    # --- Step 3: Verify imports ---
    log("--- Verify imports ---")
    imports = [
        "numpy", "yaml", "gradio", "transformers", "diffusers",
        "accelerate", "PIL", "torch",
    ]
    code = "; ".join(f"import {m}" for m in imports)
    rc = _run(
        [str(py_exe), "-c", code],
        cwd=repo_root,
        lines=lines,
        log_fn=log_fn,
    )
    if rc == 0:
        log("Verify imports: OK")
    else:
        log("Verify imports: FAILED")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    repo = Path(__file__).resolve().parent.parent
    cuda = "cu121"
    if len(sys.argv) >= 2:
        cuda = sys.argv[1]
    out = run_install_update(repo, None, None, cuda)
    print(out)
