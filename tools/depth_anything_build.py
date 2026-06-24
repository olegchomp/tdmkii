"""
Depth Anything TensorRT engine build (single env, no subprocess).
Checkpoints are downloaded from Hugging Face in code when missing.
Engines are saved to engines/depth/.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TD_DEPTH_ROOT = REPO_ROOT / "TDDepthAnything"

# (model_version, model_size) -> (repo_id, filename, repo_type or None)
# V1: LiheYoung/Depth-Anything is a Space; fallback to per-model repos (pytorch_model.bin)
# V2: separate repo per size, file at root
HF_CHECKPOINT_MAP = {
    (1, "s"): ("LiheYoung/depth_anything_vits14", "pytorch_model.bin", None),
    (1, "b"): ("LiheYoung/depth_anything_vitb14", "pytorch_model.bin", None),
    (1, "l"): ("LiheYoung/depth_anything_vitl14", "pytorch_model.bin", None),
    (2, "s"): ("depth-anything/Depth-Anything-V2-Small", "depth_anything_v2_vits.pth", None),
    (2, "b"): ("depth-anything/Depth-Anything-V2-Base", "depth_anything_v2_vitb.pth", None),
    (2, "l"): ("depth-anything/Depth-Anything-V2-Large", "depth_anything_v2_vitl.pth", None),
    # (2, "g"): Giant "Coming soon" - add when available on HF
}


def adjust_image_size(image_size: int, patch_size: int = 14) -> int:
    w = (image_size // patch_size) * patch_size
    if image_size % patch_size != 0:
        w += patch_size
    return w


def ensure_checkpoint(
    model_version: int,
    model_size: str,
    checkpoints_dir: str | Path,
    log_fn=None,
) -> Path:
    """
    Return path to local .pth. Download from Hugging Face if missing.
    log_fn(msg) is called for progress/errors.
    """
    checkpoints_dir = Path(checkpoints_dir)
    key = (model_version, model_size.lower())
    if key == (2, "g"):
        # V2 Giant not on HF yet (Coming soon)
        local_path = checkpoints_dir / "depth_anything_v2_vitg.pth"
        if local_path.exists():
            if log_fn:
                log_fn(f"Checkpoint found: {local_path}")
            return local_path
        if log_fn:
            log_fn(
                "V2 Giant (vitg) is not on Hugging Face yet. "
                "Download manually from https://github.com/DepthAnything/Depth-Anything-V2 and place depth_anything_v2_vitg.pth in the checkpoints folder."
            )
        raise FileNotFoundError(
            f"Checkpoint not found: {local_path}. V2 Giant not on HF; download manually."
        )
    if key not in HF_CHECKPOINT_MAP:
        raise ValueError(f"Unknown (version, size)={key}. V1: s,b,l. V2: s,b,l,g.")
    entry = HF_CHECKPOINT_MAP[key]
    repo_id = entry[0]
    filename = entry[1]
    repo_type = entry[2] if len(entry) > 2 else None

    local_name = Path(filename).name
    # V1 from per-model repo: we want final path as depth_anything_vit*14.pth for engine naming
    v1_pth_name = f"depth_anything_vit{model_size}14.pth" if key[0] == 1 else None
    candidates = [checkpoints_dir / local_name, checkpoints_dir / filename]
    if v1_pth_name:
        candidates.insert(0, checkpoints_dir / v1_pth_name)
    for candidate in candidates:
        if candidate.exists():
            if log_fn:
                log_fn(f"Checkpoint found: {candidate}")
            return Path(candidate)

    if log_fn:
        log_fn(f"Downloading checkpoint from Hugging Face: {repo_id} / {filename}")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError("huggingface_hub is required for checkpoint download") from e

    try:
        download_kw = dict(
            repo_id=repo_id,
            filename=filename,
            local_dir=checkpoints_dir,
        )
        if repo_type is not None:
            download_kw["repo_type"] = repo_type
        downloaded = hf_hub_download(**download_kw)
    except Exception as e:
        if log_fn:
            log_fn(f"Download failed: {e}")
        raise

    out_path = Path(downloaded)
    # V1: pytorch_model.bin -> copy to depth_anything_vit*14.pth for consistent engine naming
    if v1_pth_name and out_path.name == "pytorch_model.bin":
        import shutil
        dest = checkpoints_dir / v1_pth_name
        shutil.copy2(out_path, dest)
        out_path = dest
        if log_fn:
            log_fn(f"Checkpoint saved: {out_path}")
    elif log_fn:
        log_fn(f"Checkpoint saved: {out_path}")
    return out_path


def build_depth_engine(
    model_version: int,
    model_size: str,
    width: int,
    height: int,
    checkpoint_path: str | Path,
    engine_output_dir: str | Path,
    onnx_dir: str | Path | None = None,
    progress_callback=None,
    log_fn=None,
) -> Path:
    """
    Build ONNX and TensorRT engine for Depth Anything.
    Returns path to the created .engine file.
    Runs in-process; requires TDDepthAnything on sys.path for depth_anything/depth_anything_v2.
    """
    checkpoint_path = Path(checkpoint_path)
    engine_output_dir = Path(engine_output_dir)
    onnx_dir = Path(onnx_dir) if onnx_dir else engine_output_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    engine_output_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str):
        if log_fn:
            log_fn(msg)
        else:
            print(msg)

    # Ensure TDDepthAnything is importable and torch.hub cache is local
    td_path = str(TD_DEPTH_ROOT)
    if td_path not in sys.path:
        sys.path.insert(0, td_path)
    import torch
    torch.hub.set_dir(str(REPO_ROOT / "torchhub"))

    width = adjust_image_size(width)
    height = adjust_image_size(height)
    log(f"Image shape: {width}x{height}")

    encoder = f"vit{model_size}"
    if model_version == 1:
        from depth_anything.dpt import DPT_DINOv2

        if encoder == "vits":
            model = DPT_DINOv2(
                encoder="vits",
                features=64,
                out_channels=[48, 96, 192, 384],
                localhub=False,
            )
        elif encoder == "vitb":
            model = DPT_DINOv2(
                encoder="vitb",
                features=128,
                out_channels=[96, 192, 384, 768],
                localhub=False,
            )
        elif encoder == "vitl":
            model = DPT_DINOv2(
                encoder="vitl",
                features=256,
                out_channels=[256, 512, 1024, 1024],
                localhub=False,
            )
        else:
            model = DPT_DINOv2(
                encoder="vitg",
                features=384,
                out_channels=[1536, 1536, 1536, 1536],
                localhub=False,
            )
    else:
        from depth_anything_v2.dpt import DepthAnythingV2

        model_configs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "encoder": "vitb",
                "features": 128,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
            "vitg": {
                "encoder": "vitg",
                "features": 384,
                "out_channels": [1536, 1536, 1536, 1536],
            },
        }
        model = DepthAnythingV2(**model_configs[encoder])

    from polygraphy.backend.trt import (
        CreateConfig,
        Profile,
        engine_from_network,
        network_from_onnx_path,
        save_engine,
    )
    import tensorrt as trt

    if progress_callback:
        progress_callback(0.05, "Loading checkpoint...")
    try:
        state = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=False
        )
    except TypeError:
        state = torch.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    # Output names for ONNX/engine (match depth_tensorrt.py expectation: input, output)
    base_name = checkpoint_path.stem
    onnx_path = onnx_dir / f"{base_name}_{width}x{height}.onnx"
    engine_path = engine_output_dir / f"{base_name}_{width}x{height}.engine"

    if progress_callback:
        progress_callback(0.2, "Exporting ONNX...")
    log(f"Exporting ONNX: {onnx_path}")

    dummy_input = torch.ones(1, 3, height, width)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        verbose=False,
    )
    log(f"ONNX saved: {onnx_path}")

    if progress_callback:
        progress_callback(0.5, "Building TensorRT engine...")
    log(f"Building TensorRT engine: {engine_path}")

    p = Profile()
    engine = engine_from_network(
        network_from_onnx_path(
            str(onnx_path),
            flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM],
        ),
        config=CreateConfig(
            fp16=True,
            refittable=False,
            profiles=[p],
            load_timing_cache=None,
        ),
        save_timing_cache=None,
    )
    save_engine(engine, path=str(engine_path))
    log(f"Engine saved: {engine_path}")

    if progress_callback:
        progress_callback(1.0, "Done")
    return engine_path
