"""
Flux Klein 4B TensorRT engine build.
Export transformer + VAE encoder + VAE decoder to ONNX, build BF16 TRT engines.
Flux Klein = img2img only, encoder needed for input image -> latents.
Output: transformer_v2.engine, vae_encoder.engine, vae_decoder.engine
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent

BATCH = 1
TEXT_SEQ_LEN = 512


def _image_seq_len(width: int, height: int) -> int:
    """Image sequence length from resolution (latent h*w/2)."""
    latent_h, latent_w = height // 8, width // 8
    return (latent_h * latent_w) // 2


def _workspace_gb(width: int, height: int) -> int:
    """TRT workspace (GB) from resolution."""
    area = width * height
    return max(4, min(16, 2 + area // (512 * 512) * 2))


def _make_transformer_wrapper(transformer):
    """Create nn.Module wrapper for ONNX export of Flux2Klein transformer."""
    import torch.nn as nn

    class TransformerWrapper(nn.Module):
        def __init__(self, t):
            super().__init__()
            self.transformer = t

        def forward(self, hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids):
            return self.transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=None,
                joint_attention_kwargs={},
                return_dict=False,
            )[0]

    return TransformerWrapper(transformer)


def _make_vae_encoder_wrapper(vae):
    """Create nn.Module wrapper for ONNX export of VAE encoder (image -> latent)."""
    import torch.nn as nn

    class VAEEncoderWrapper(nn.Module):
        def __init__(self, v):
            super().__init__()
            self.vae = v

        def forward(self, image):
            posterior = self.vae.encode(image, return_dict=False)[0]
            return posterior.mode()

    return VAEEncoderWrapper(vae)


def _make_vae_decoder_wrapper(vae):
    """Create nn.Module wrapper for ONNX export of VAE decoder (latent -> image)."""
    import torch.nn as nn

    class VAEDecoderWrapper(nn.Module):
        def __init__(self, v):
            super().__init__()
            self.vae = v

        def forward(self, latent):
            return self.vae.decode(latent, return_dict=False)[0]

    return VAEDecoderWrapper(vae)




def _create_dummy_inputs_transformer(pipe, width: int, height: int, device="cpu", dtype=None):
    import torch
    dtype = dtype or torch.float32
    cfg = pipe.transformer.config
    in_ch = cfg.in_channels
    jad = cfg.joint_attention_dim
    seq_len = _image_seq_len(width, height)
    return (
        torch.randn(BATCH, seq_len, in_ch, dtype=dtype, device=device),
        torch.randn(BATCH, TEXT_SEQ_LEN, jad, dtype=dtype, device=device),
        torch.ones(BATCH, dtype=dtype, device=device) * 0.5,
        torch.zeros(BATCH, seq_len, 4, dtype=dtype, device=device),
        torch.zeros(BATCH, TEXT_SEQ_LEN, 4, dtype=dtype, device=device),
    )


def _export_transformer_onnx(pipe, output_path, width: int, height: int, opset=18, log_fn=None):
    import torch

    def _log(msg):
        if log_fn:
            log_fn(msg)
        else:
            print(msg)

    _log("Moving transformer to CUDA float16...")
    transformer = pipe.transformer.eval().cuda().half()
    wrapped = _make_transformer_wrapper(transformer)
    dummy = _create_dummy_inputs_transformer(pipe, width, height, device="cuda", dtype=torch.float16)
    _log(f"Exporting transformer to {output_path} (opset {opset})...")
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            dummy,
            output_path,
            input_names=["hidden_states", "encoder_hidden_states", "timestep", "img_ids", "txt_ids"],
            output_names=["output"],
            opset_version=opset,
            dynamo=True,
        )
    _log(f"Saved: {output_path}")


def _export_vae_encoder_onnx(pipe, output_path, width: int, height: int, opset=18, log_fn=None):
    import torch

    def _log(msg):
        if log_fn:
            log_fn(msg)
        else:
            print(msg)

    vae = pipe.vae.eval().cuda().half()
    wrapped = _make_vae_encoder_wrapper(vae)
    dummy = torch.randn(1, 3, height, width, dtype=torch.float16, device="cuda")
    _log(f"Exporting VAE encoder to {output_path} (opset {opset})...")
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            (dummy,),
            output_path,
            input_names=["image"],
            output_names=["latent"],
            opset_version=opset,
            do_constant_folding=True,
            dynamo=True,
        )
    _log(f"Saved: {output_path}")


def _export_vae_decoder_onnx(pipe, output_path, width: int, height: int, opset=18, log_fn=None):
    import torch

    def _log(msg):
        if log_fn:
            log_fn(msg)
        else:
            print(msg)

    vae = pipe.vae.eval().cuda().half()
    wrapped = _make_vae_decoder_wrapper(vae)
    latent_h, latent_w = height // 8, width // 8
    dummy = torch.randn(1, 32, latent_h, latent_w, dtype=torch.float16, device="cuda")
    _log(f"Exporting VAE decoder to {output_path} (opset {opset})...")
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            (dummy,),
            output_path,
            input_names=["latent"],
            output_names=["image"],
            opset_version=opset,
            do_constant_folding=True,
            dynamo=True,
        )
    _log(f"Saved: {output_path}")


def _build_bf16_engine(onnx_path, engine_path, workspace_gb=4, log_fn=None):
    import tensorrt as trt

    def _log(msg):
        if log_fn:
            log_fn(msg)
        else:
            print(msg)

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    onnx_abs = os.path.abspath(onnx_path)
    _log(f"Parsing ONNX: {onnx_abs}")

    if not parser.parse_from_file(onnx_abs):
        for i in range(parser.num_errors):
            _log(f"  ONNX parse error: {parser.get_error(i)}")
        raise RuntimeError(f"ONNX parse failed for {onnx_path}")

    _log(f"  Network: {network.num_inputs} inputs, {network.num_outputs} outputs, {network.num_layers} layers")

    for i in range(network.num_inputs):
        inp = network.get_input(i)
        inp.dtype = trt.bfloat16
    for i in range(network.num_outputs):
        out = network.get_output(i)
        out.dtype = trt.bfloat16

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))
    config.set_flag(trt.BuilderFlag.BF16)
    config.set_flag(trt.BuilderFlag.TF32)
    config.builder_optimization_level = 3

    _log("Building engine...")
    t0 = time.time()
    engine_bytes = builder.build_serialized_network(network, config)
    elapsed = time.time() - t0

    if engine_bytes is None:
        raise RuntimeError("TensorRT engine build failed!")

    data = bytes(engine_bytes)
    with open(engine_path, "wb") as f:
        f.write(data)
    size_mb = len(data) / (1024 * 1024)
    _log(f"Saved: {engine_path} ({size_mb:.0f} MB) in {elapsed:.0f}s")


def build_flux_klein_engines(
    engine_output_dir: str | Path,
    width: int = 512,
    height: int = 512,
    model_id: str = "black-forest-labs/FLUX.2-klein-4B",
    opset: int = 18,
    progress_callback: Callable[[float, str], None] | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[Path, Path, Path]:
    """
    Build Flux Klein 4B TRT engines: ONNX export (transformer + VAE encoder + VAE decoder) then TRT build.
    width, height — resolution for export and engine.
    workspace (GB) computed automatically from resolution.
    Returns (transformer_engine_path, vae_encoder_engine_path, vae_decoder_engine_path).
    """
    import gc
    import torch

    engine_output_dir = Path(engine_output_dir)
    engine_output_dir.mkdir(parents=True, exist_ok=True)
    w, h = max(64, int(width)), max(64, int(height))
    workspace_gb = _workspace_gb(w, h)

    def _log(msg):
        if log_fn:
            log_fn(msg)
        else:
            print(msg)

    _log(f"Resolution: {w}×{h}, TRT workspace: {workspace_gb} GB (auto)")

    def _progress(p, desc):
        if progress_callback:
            progress_callback(p, desc)

    # Step 1: Load pipeline and export transformer ONNX
    _progress(0.05, "Loading Flux2KleinPipeline...")
    from diffusers_flux2 import Flux2KleinPipeline
    pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.to("cuda")

    transformer_onnx = engine_output_dir / f"transformer_v2_{w}x{h}.onnx"
    _progress(0.08, "Exporting transformer to ONNX...")
    _export_transformer_onnx(pipe, str(transformer_onnx), w, h, opset=opset, log_fn=_log)

    # Step 2: Export VAE encoder ONNX
    vae_encoder_onnx = engine_output_dir / f"vae_encoder_{w}x{h}.onnx"
    _progress(0.22, "Exporting VAE encoder to ONNX...")
    _export_vae_encoder_onnx(pipe, str(vae_encoder_onnx), w, h, opset=opset, log_fn=_log)

    # Step 3: Export VAE decoder ONNX
    vae_decoder_onnx = engine_output_dir / f"vae_decoder_{w}x{h}.onnx"
    _progress(0.35, "Exporting VAE decoder to ONNX...")
    _export_vae_decoder_onnx(pipe, str(vae_decoder_onnx), w, h, opset=opset, log_fn=_log)

    # Free pipeline to save VRAM
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    # Step 4: Build transformer TRT engine
    transformer_engine = engine_output_dir / f"transformer_v2_{w}x{h}.engine"
    _progress(0.5, "Building transformer TRT engine...")
    _build_bf16_engine(str(transformer_onnx), str(transformer_engine), workspace_gb=workspace_gb, log_fn=_log)

    # Step 5: Build VAE encoder TRT engine
    vae_encoder_engine = engine_output_dir / f"vae_encoder_{w}x{h}.engine"
    _progress(0.7, "Building VAE encoder TRT engine...")
    _build_bf16_engine(str(vae_encoder_onnx), str(vae_encoder_engine), workspace_gb=workspace_gb, log_fn=_log)

    # Step 6: Build VAE decoder TRT engine
    vae_decoder_engine = engine_output_dir / f"vae_decoder_{w}x{h}.engine"
    _progress(0.9, "Building VAE decoder TRT engine...")
    _build_bf16_engine(str(vae_decoder_onnx), str(vae_decoder_engine), workspace_gb=workspace_gb, log_fn=_log)

    _progress(1.0, "Done")
    return transformer_engine, vae_encoder_engine, vae_decoder_engine
