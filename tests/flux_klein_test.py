"""
Flux Klein 4B — тестовый клиент для проверки TRT-движков и генерации.

Использование:
    python tests/flux_klein_test.py
    python tests/flux_klein_test.py --image tests/image.jpg -p "make it sunset" -o tests/output_flux_klein.png

Требуется: engines/flux_klein/ (transformer, vae_encoder, vae_decoder) + config.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import tensorrt as trt
from PIL import Image


class TrtEngine:
    """TRT engine wrapper with auto dtype."""

    def __init__(self, engine_path: str, label: str = ""):
        self.logger = trt.Logger(trt.Logger.WARNING)
        print(f"Loading TRT engine ({label}): {engine_path}")
        t0 = time.time()
        runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.io_info = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = list(self.engine.get_tensor_shape(name))
            dtype_trt = self.engine.get_tensor_dtype(name)
            dtype_torch = {
                trt.bfloat16: torch.bfloat16,
                trt.float16: torch.float16,
                trt.float32: torch.float32,
            }.get(dtype_trt, torch.float32)
            is_input = mode == trt.TensorIOMode.INPUT
            self.io_info[name] = {"shape": shape, "dtype": dtype_torch, "input": is_input}
        print(f"  Loaded in {time.time() - t0:.1f}s")

    def run(self, inputs: dict) -> dict:
        device = "cuda"
        for name, tensor in inputs.items():
            info = self.io_info[name]
            t = tensor.detach().to(dtype=info["dtype"]).contiguous().cuda()
            self.context.set_tensor_address(name, t.data_ptr())
            inputs[name] = t
        outputs = {}
        for name, info in self.io_info.items():
            if not info["input"]:
                shape = self.context.get_tensor_shape(name)
                t = torch.empty(list(shape), dtype=info["dtype"], device=device)
                self.context.set_tensor_address(name, t.data_ptr())
                outputs[name] = t
        with torch.cuda.stream(self.stream):
            self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return outputs


class _FakeLatentDist:
    def __init__(self, latent): self._latent = latent
    def mode(self): return self._latent
    def sample(self, generator=None): return self._latent


def patch_transformer(pipe, engine: TrtEngine):
    def trt_forward(hidden_states, encoder_hidden_states=None, timestep=None,
                    img_ids=None, txt_ids=None, guidance=None,
                    joint_attention_kwargs=None, return_dict=True, **kwargs):
        result = engine.run({
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "img_ids": img_ids,
            "txt_ids": txt_ids,
        })
        sample = result["output"].to(dtype=hidden_states.dtype)
        if return_dict:
            from diffusers.models.modeling_outputs import Transformer2DModelOutput
            return Transformer2DModelOutput(sample=sample)
        return (sample,)
    pipe.transformer.forward = trt_forward
    print("Transformer patched with TRT.")


def patch_vae_encoder(pipe, engine: TrtEngine):
    def trt_encode(x, return_dict=True, **kwargs):
        result = engine.run({"image": x})
        latent = result["latent"].to(dtype=x.dtype)
        fake = _FakeLatentDist(latent)
        if return_dict:
            # Pipeline expects .latent_dist.mode() / .sample(); diffusers main lacks AutoencoderKLOutput
            out = type("_Enc", (), {})()
            out.latent_dist = fake
            return out
        return (fake,)
    pipe.vae.encode = trt_encode
    print("VAE encode patched with TRT.")


def patch_vae_decoder(pipe, engine: TrtEngine):
    def trt_decode(z, return_dict=True, **kwargs):
        result = engine.run({"latent": z})
        image = result["image"].to(dtype=z.dtype)
        if return_dict:
            from diffusers.models.autoencoders.vae import DecoderOutput
            return DecoderOutput(sample=image)
        return (image,)
    pipe.vae.decode = trt_decode
    print("VAE decode patched with TRT.")


def main():
    ap = argparse.ArgumentParser(description="Flux Klein 4B — тест TRT engines")
    ap.add_argument("--image", "-i", default=str(REPO_ROOT / "tests" / "image.jpg"), help="Input image")
    ap.add_argument("--prompt", "-p", default="make it sunset", help="Edit prompt")
    ap.add_argument("--output", "-o", default=str(REPO_ROOT / "tests" / "output_flux_klein.png"), help="Output")
    ap.add_argument("--steps", "-s", type=int, default=4, help="Inference steps")
    ap.add_argument("--tensor", "-t", action="store_true", help="Pass image as tensor (GPU path)")
    ap.add_argument("--config", "-c", help="Config YAML (default: engines/flux_klein/config_flux_klein_512x512.yaml)")
    args = ap.parse_args()

    engine_dir = REPO_ROOT / "engines" / "flux_klein"
    config_path = Path(args.config) if args.config else engine_dir / "config_flux_klein_512x512.yaml"

    if not config_path.exists():
        print(f"✗ Config not found: {config_path}")
        print("  Run Build in gradio_prepare Flux Klein tab first.")
        sys.exit(1)

    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    w = cfg.get("width", 512)
    h = cfg.get("height", 512)
    tr_path = Path(cfg["transformer_engine"])
    vae_enc_path = Path(cfg["vae_encoder_engine"]) if cfg.get("vae_encoder_engine") else None
    vae_dec_path = Path(cfg["vae_engine"])

    if not tr_path.exists():
        print(f"✗ Transformer engine not found: {tr_path}")
        sys.exit(1)
    if not vae_dec_path.exists():
        print(f"✗ VAE decoder engine not found: {vae_dec_path}")
        sys.exit(1)
    has_encoder = vae_enc_path and vae_enc_path.exists()

    if not Path(args.image).exists():
        print(f"✗ Image not found: {args.image}")
        sys.exit(1)

    # 1. Load engines
    transformer_trt = TrtEngine(str(tr_path), "transformer")
    vae_dec_trt = TrtEngine(str(vae_dec_path), "VAE decoder")
    vae_enc_trt = TrtEngine(str(vae_enc_path), "VAE encoder") if has_encoder else None

    # 2. Load pipeline
    print("Loading Flux2KleinPipeline...")
    from diffusers_flux2 import Flux2KleinPipeline
    pipe = Flux2KleinPipeline.from_pretrained(
        cfg.get("model_id", "black-forest-labs/FLUX.2-klein-4B"),
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    pipe.transformer.to("cpu")
    pipe.vae.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    pipe.vae.to("cuda")

    # 3. Patch with TRT
    patch_transformer(pipe, transformer_trt)
    if vae_enc_trt:
        patch_vae_encoder(pipe, vae_enc_trt)
    patch_vae_decoder(pipe, vae_dec_trt)

    # 4. Inference
    input_image = Image.open(args.image).convert("RGB")
    orig_size = input_image.size
    input_image = input_image.resize((w, h), Image.LANCZOS)
    print(f"Input: {orig_size} -> {w}x{h}, prompt: {args.prompt!r}, tensor={args.tensor}")

    if args.tensor:
        # GPU path: convert to tensor (0-1), pass directly
        img_np = np.array(input_image).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to("cuda", torch.bfloat16)
        pipe_input = img_t
    else:
        pipe_input = input_image

    print("Warmup...")
    _ = pipe(image=pipe_input, prompt=args.prompt, height=h, width=w, num_inference_steps=2)
    torch.cuda.synchronize()

    print("Inference...")
    t0 = time.time()
    result = pipe(image=pipe_input, prompt=args.prompt, height=h, width=w, num_inference_steps=args.steps)
    elapsed = time.time() - t0
    result.images[0].save(args.output)
    print(f"Saved {args.output} ({elapsed:.2f}s, {args.steps/elapsed:.1f} it/s)")


if __name__ == "__main__":
    main()
