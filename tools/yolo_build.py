"""
Ultralytics YOLO TensorRT engine build.
Official models are downloaded by Ultralytics on first use.
Engines are saved to engines/yolo/ with predictable names.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_yolo_engine(
    model_spec: str,
    imgsz: int | tuple[int, int],
    batch: int,
    half: bool,
    int8: bool,
    dynamic: bool,
    workspace: float | None,
    simplify: bool,
    data: str | None,
    fraction: float,
    engine_output_dir: str | Path,
    progress_callback: Callable[[float, str], None] | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> Path:
    """
    Build TensorRT engine from Ultralytics YOLO model.
    Returns path to the created .engine file in engine_output_dir.
    """
    from ultralytics import YOLO

    engine_output_dir = Path(engine_output_dir)
    engine_output_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        if log_fn:
            log_fn(msg)
        else:
            print(msg)

    if progress_callback:
        progress_callback(0.0, "Loading model...")
    model_spec = (model_spec or "yolo11n.pt").strip()
    log(f"Model: {model_spec}")
    log(f"imgsz={imgsz}, batch={batch}, half={half}, int8={int8}, dynamic={dynamic}")

    model = YOLO(model_spec)
    model_stem = Path(model_spec).stem if "/" in model_spec or "\\" in model_spec else model_spec.replace(".pt", "")

    export_kwargs = {
        "format": "engine",
        "imgsz": imgsz,
        "batch": batch,
        "half": half,
        "int8": int8,
        "dynamic": dynamic,
        "simplify": simplify,
        "workspace": workspace,
        "verbose": False,
    }
    if int8 and data:
        export_kwargs["data"] = data
        export_kwargs["fraction"] = fraction

    if progress_callback:
        progress_callback(0.2, "Exporting to TensorRT engine...")
    log("Exporting (this may take several minutes)...")

    # Export into engine_output_dir so artifacts don't end up in project root
    orig_cwd = os.getcwd()
    try:
        os.chdir(engine_output_dir)
        result_path = model.export(**export_kwargs)
    finally:
        os.chdir(orig_cwd)
    result_path = Path(result_path)
    if not result_path.is_absolute():
        result_path = (engine_output_dir / result_path).resolve()

    imgsz_str = f"{imgsz[0]}x{imgsz[1]}" if isinstance(imgsz, tuple) else str(imgsz)
    engine_name = f"{model_stem}_{imgsz_str}_b{batch}"
    if half:
        engine_name += "_fp16"
    if int8:
        engine_name += "_int8"
    engine_name += ".engine"

    dest_path = engine_output_dir / engine_name
    if result_path.resolve() != dest_path.resolve():
        if progress_callback:
            progress_callback(0.9, "Copying engine to output dir...")
        shutil.copy2(result_path, dest_path)
        if result_path.exists():
            try:
                result_path.unlink()
            except OSError:
                pass
        log(f"Engine saved: {dest_path}")
    else:
        log(f"Engine saved: {dest_path}")

    if progress_callback:
        progress_callback(1.0, "Done")
    return dest_path
