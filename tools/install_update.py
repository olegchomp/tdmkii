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
    print(line)
    if log_fn:
        log_fn(line)


def _run(
    args: list[str],
    cwd: Optional[Path],
    lines: list[str],
    log_fn: Optional[Callable[[str], None]],
    env: Optional[dict] = None,
) -> int:
    """Run command, stream stdout/stderr to console and log in real time, return returncode."""
    _log(lines, f"> {' '.join(args)}", log_fn)
    proc = subprocess.Popen(
        args,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in iter(proc.stdout.readline, ""):
        line = line.rstrip()
        if line:
            _log(lines, line, log_fn)
    return proc.wait()


def run_install_update(
    repo_root: Path,
    python_exe: Optional[Union[Path, str]] = None,
    git_exe: Optional[Union[Path, str]] = None,
    cuda_ver: str = "cu121",
    log_fn: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
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

    # --- Step 1: Git hard reset (GitPython, portable git) --- [commented out]
    # if progress_callback:
    #     progress_callback(0.05, "Git...")
    # log("--- Git: fetch + reset --hard (portable git) ---")
    # if git_bin.is_file():
    #     os.environ["GIT_PYTHON_GIT_EXECUTABLE"] = str(git_bin)
    # else:
    #     log(f"Warning: portable git not found at {git_bin}; GitPython may use system git")
    # try:
    #     import git
    # except ImportError:
    #     log("Warning: GitPython not installed; skipping git step. pip install GitPython")
    # else:
    #     git_dir = repo_root / ".git"
    #     if not git_dir.exists():
    #         log("Not a git repo (.git missing); skipping git step.")
    #     else:
    #         try:
    #             repo = git.Repo(repo_root)
    #             origin = repo.remotes.origin
    #             origin.fetch()
    #             branch = repo.active_branch.name
    #             ref = f"origin/{branch}"
    #             repo.git.reset("--hard", ref)
    #             log(f"Reset --hard {ref} OK")
    #         except Exception as e:
    #             log(f"Git error: {e}")
    # log("")

    if progress_callback:
        progress_callback(0.15, "pip upgrade...")
    # --- Step 2: pip (always via python -m pip) ---
    log("--- pip: upgrade pip ---")
    rc = _run(
        [str(py_exe), "-m", "pip", "install", "--upgrade", "pip", "--no-warn-script-location"],
        cwd=repo_root,
        lines=lines,
        log_fn=log_fn,
    )
    if rc != 0:
        log(f"pip upgrade returned {rc}")
    log("")

    if progress_callback:
        progress_callback(0.25, "PyTorch (may take several minutes)...")
    log("--- pip: PyTorch (CUDA) ---")
    rc = _run(
        [
            str(py_exe), "-m", "pip", "install", "torch", "torchvision",
            "--index-url", f"https://download.pytorch.org/whl/{cuda_ver}",
            "--no-warn-script-location",
        ],
        cwd=repo_root,
        lines=lines,
        log_fn=log_fn,
    )
    if rc != 0:
        log(f"PyTorch install returned {rc}")
    log("")

    if progress_callback:
        progress_callback(0.48, "Remove stub 'cuda' if present...")
    log("--- pip: uninstall stub 'cuda' (so cuda-python provides cudart) ---")
    _run(
        [str(py_exe), "-m", "pip", "uninstall", "-y", "cuda"],
        cwd=repo_root,
        lines=lines,
        log_fn=log_fn,
    )
    log("")

    if progress_callback:
        progress_callback(0.5, "requirements.txt...")
    log("--- pip: -r requirements.txt ---")
    req_file = repo_root / "requirements.txt"
    if not req_file.is_file():
        log(f"Warning: {req_file} not found")
    else:
        rc = _run(
            [str(py_exe), "-m", "pip", "install", "-r", str(req_file), "--no-warn-script-location"],
            cwd=repo_root,
            lines=lines,
            log_fn=log_fn,
        )
        if rc != 0:
            log(f"pip -r requirements.txt returned {rc}")
    log("")

    if progress_callback:
        progress_callback(0.7, "StreamDiffusion TensorRT...")
    log("--- StreamDiffusion: install-tensorrt (cuda-python, cudnn, trt versions) ---")
    stream_src = (repo_root / "StreamDiffusion" / "src").resolve()
    if not (stream_src / "streamdiffusion").is_dir():
        log(f"Skip install-tensorrt: {stream_src / 'streamdiffusion'} not found")
    else:
        # Run as module with path in -c so subprocess finds streamdiffusion (no PYTHONPATH env)
        bootstrap = (
            f"import sys; sys.path.insert(0, {repr(str(stream_src))}); "
            "import runpy; runpy.run_module('streamdiffusion.tools.install-tensorrt', run_name='__main__')"
        )
        rc = _run(
            [str(py_exe), "-c", bootstrap],
            cwd=repo_root,
            lines=lines,
            log_fn=log_fn,
        )
        if rc != 0:
            log(f"install-tensorrt returned {rc}")
    log("")

    if progress_callback:
        progress_callback(0.85, "Verify imports...")
    # --- Step 3: Verify imports ---
    log("--- Verify imports ---")
    imports = [
        "numpy", "yaml", "gradio", "transformers", "diffusers",
        "accelerate", "PIL", "torch", "tensorrt",
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
    if progress_callback:
        progress_callback(1.0, "Done")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    repo = Path(__file__).resolve().parent.parent
    cuda = "cu121"
    if len(sys.argv) >= 2:
        cuda = sys.argv[1]
    out = run_install_update(repo, None, None, cuda)
    print(out)
