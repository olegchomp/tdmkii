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
    task: str | None = None,
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

    # Ultralytics has no download_dir argument (issue #17513); download goes to cwd. Use repo checkpoints dir.
    yolo_weights_dir = REPO_ROOT / "checkpoints"
    yolo_weights_dir.mkdir(parents=True, exist_ok=True)
    from ultralytics.utils import SETTINGS

    prev_weights_dir = SETTINGS.get("weights_dir")
    prev_cwd = os.getcwd()
    SETTINGS["weights_dir"] = str(yolo_weights_dir)
    os.chdir(yolo_weights_dir)
    try:
        if progress_callback:
            progress_callback(0.0, "Loading model...")
        model_spec = (model_spec or "yolo11n.pt").strip()
        # Fix common typo: yolo8n -> yolov8n (Ultralytics expects yolov8n-seg.pt)
        if model_spec.startswith("yolo8") and not model_spec.startswith("yolov8"):
            model_spec = "yolov8" + model_spec[5:]
        task_str = (task or "").strip().lower() if task else None
        if task_str and task_str not in ("detect", "segment", "pose", "obb"):
            task_str = None
        is_custom_path = "/" in model_spec or "\\" in model_spec or Path(model_spec).exists()
        log(f"Model: {model_spec}")
        log(f"imgsz={imgsz}, batch={batch}, half={half}, int8={int8}, dynamic={dynamic}" + (f", task={task_str}" if task_str else ""))

        # Only pass task for custom paths; standard names (yolo8n-pose.pt etc) let Ultralytics auto-download & infer
        model = YOLO(model_spec, task=task_str) if (task_str and is_custom_path) else YOLO(model_spec)
    finally:
        os.chdir(prev_cwd)
        if prev_weights_dir is not None:
            SETTINGS["weights_dir"] = prev_weights_dir

    model_stem = Path(model_spec).stem if "/" in model_spec or "\\" in model_spec else model_spec.replace(".pt", "")

    # Copy .pt file to engine_output_dir (Ultralytics caches in ~/.config/Ultralytics or similar)
    inner = getattr(model, "model", None)
    pt_source = getattr(model, "ckpt_path", None) or (getattr(inner, "pt_path", None) if inner else None)
    pt_dest = engine_output_dir / f"{model_stem}.pt"
    if pt_source and Path(pt_source).exists() and Path(pt_source).resolve() != pt_dest.resolve():
        shutil.copy2(pt_source, pt_dest)
        log(f"Model copied: {pt_dest}")

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

    # Force export output to engine_output_dir (Ultralytics otherwise saves next to cached .pt)
    engine_base = engine_output_dir / f"{model_stem}.pt"
    old_pt_path = None
    inner = getattr(model, "model", model)
    if hasattr(inner, "pt_path") and inner.pt_path:
        old_pt_path = inner.pt_path
        inner.pt_path = str(engine_base)

    try:
        result_path = model.export(**export_kwargs)
    finally:
        if old_pt_path is not None and hasattr(inner, "pt_path"):
            inner.pt_path = old_pt_path

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
        if result_path.exists():
            shutil.copy2(result_path, dest_path)
            try:
                result_path.unlink()
            except OSError:
                pass
            log(f"Engine saved: {dest_path}")
        else:
            # Fallback: engine may be in engine_output_dir under different name
            candidates = list(engine_output_dir.glob("*.engine"))
            if candidates:
                best = max(candidates, key=lambda p: p.stat().st_mtime)
                if best != dest_path:
                    shutil.copy2(best, dest_path)
                    try:
                        best.unlink()
                    except OSError:
                        pass
                log(f"Engine saved: {dest_path}")
            else:
                raise FileNotFoundError(f"Engine not found at {result_path}; export may have failed")
    else:
        log(f"Engine saved: {dest_path}")

    # Test inference on empty input so we have a clear "generation finished" step
    if progress_callback:
        progress_callback(0.95, "Test inference...")
    import numpy as np
    h, w = (imgsz[0], imgsz[1]) if isinstance(imgsz, tuple) else (imgsz, imgsz)
    dummy = np.zeros((h, w, 3), dtype=np.uint8)
    engine_model = YOLO(str(dest_path))
    _ = engine_model.predict(source=dummy, imgsz=imgsz, half=half, verbose=False)
    log("✓ Test inference OK — engine runs.")

    if progress_callback:
        progress_callback(1.0, "Done")
    return dest_path
