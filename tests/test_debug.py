"""
Диагностический скрипт: загружает конфиг + реальную картинку, прогоняет инференс,
выводит stats тензоров на каждом этапе. Помогает найти, где результат становится чёрным.

Использование:
    python tests/test_debug.py <config.yaml> [--image tests/image.jpg]
"""
import sys
import time
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SD_SRC = REPO_ROOT / "StreamDiffusion" / "src"
if str(SD_SRC) not in sys.path:
    sys.path.insert(0, str(SD_SRC))


def tensor_stats(t, name="tensor"):
    """Print min/max/mean/std of a tensor."""
    print(f"  {name:30s} shape={list(t.shape)}  "
          f"min={t.min().item():.4f}  max={t.max().item():.4f}  "
          f"mean={t.mean().item():.4f}  std={t.std().item():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to config YAML")
    parser.add_argument("--image", default=str(REPO_ROOT / "tests" / "image.jpg"),
                        help="Input image path")
    parser.add_argument("--frames", type=int, default=30, help="Number of frames")
    parser.add_argument("--save-every", type=int, default=5, help="Save every N-th frame")
    args = parser.parse_args()

    import torch
    import yaml
    from PIL import Image
    from torchvision import transforms

    # ── 1. Load config ───────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["compile_engines_only"] = False
    cfg["build_engines_if_missing"] = False

    engine_dir = cfg.get("engine_dir", "engines")
    if not Path(engine_dir).is_absolute():
        cfg["engine_dir"] = str(REPO_ROOT / engine_dir)

    w, h = cfg.get("width", 512), cfg.get("height", 512)
    cfg_type = cfg.get("cfg_type", "none")
    t_list = cfg.get("t_index_list", [])
    n_steps = len(t_list)

    print("=" * 60)
    print(f"Model:       {cfg.get('model_id')}")
    print(f"Resolution:  {w}x{h}")
    print(f"cfg_type:    {cfg_type}")
    print(f"t_index_list:{t_list}  ({n_steps} steps)")
    print(f"guidance:    {cfg.get('guidance_scale')}  delta={cfg.get('delta')}")
    print(f"LoRA:        {cfg.get('lora_dict', {})}")
    print(f"ControlNet:  {cfg.get('use_controlnet')}  nets={len(cfg.get('controlnets', []))}")
    print(f"Prompt:      '{cfg.get('prompt', '')}'")
    print(f"Neg prompt:  '{cfg.get('negative_prompt', '')}'")
    print("=" * 60)

    # ── 2. Load image ────────────────────────────────────────
    img_path = Path(args.image)
    if not img_path.exists():
        print(f"ERROR: image not found: {img_path}")
        sys.exit(1)
    input_img = Image.open(img_path).convert("RGB").resize((w, h))
    print(f"\nInput image: {img_path.name} resized to {w}x{h}")

    to_tensor = transforms.ToTensor()
    input_tensor = to_tensor(input_img).unsqueeze(0).to("cuda", dtype=torch.float16)
    tensor_stats(input_tensor, "input_tensor [0,1]")

    # ── 3. Create pipeline ───────────────────────────────────
    print("\nLoading pipeline...")
    t0 = time.time()
    from streamdiffusion.config import create_wrapper_from_config
    wrapper = create_wrapper_from_config(cfg)
    print(f"Pipeline ready in {time.time() - t0:.1f}s")

    stream = wrapper.stream
    print(f"\nStream internals:")
    print(f"  cfg_type:            {stream.cfg_type}")
    print(f"  batch_size:          {stream.batch_size}")
    print(f"  trt_unet_batch_size: {stream.trt_unet_batch_size}")
    print(f"  frame_bff_size:      {stream.frame_bff_size}")
    print(f"  denoising_steps_num: {stream.denoising_steps_num}")
    print(f"  guidance_scale:      {stream.guidance_scale}")
    if hasattr(stream, 'delta'):
        print(f"  delta:               {stream.delta}")

    # ── 4. Warmup with PIL (как wrapper.img2img ожидает) ─────
    print("\n--- Warmup (batch fill) ---")
    warmup_count = stream.batch_size + 2
    for i in range(warmup_count):
        result = wrapper.img2img(input_img)
        if isinstance(result, torch.Tensor):
            tensor_stats(result, f"warmup[{i}]")
        elif isinstance(result, Image.Image):
            t_tmp = to_tensor(result)
            tensor_stats(t_tmp, f"warmup[{i}] (PIL)")
        else:
            print(f"  warmup[{i}] type={type(result)}")
    print(f"Warmup done ({warmup_count} frames)")

    # ── 5. Inference frames ──────────────────────────────────
    out_dir = Path(args.config).parent if Path(args.config).is_absolute() else REPO_ROOT / Path(args.config).parent
    out_dir = out_dir / "debug_output"
    out_dir.mkdir(exist_ok=True)

    print(f"\n--- Inference ({args.frames} frames) ---")
    print(f"Saving to: {out_dir}")

    t0 = time.time()
    for i in range(args.frames):
        result = wrapper.img2img(input_img)

        if isinstance(result, torch.Tensor):
            if i % args.save_every == 0 or i < 3:
                tensor_stats(result, f"frame[{i}]")
                from torchvision.transforms.functional import to_pil_image
                img_out = to_pil_image(result.squeeze().cpu().clamp(0, 1))
                img_out.save(out_dir / f"frame_{i:03d}.png")
        elif isinstance(result, Image.Image):
            if i % args.save_every == 0 or i < 3:
                t_check = to_tensor(result)
                tensor_stats(t_check, f"frame[{i}] (PIL>tensor)")
                result.save(out_dir / f"frame_{i:03d}.png")

    elapsed = time.time() - t0
    fps = args.frames / elapsed
    print(f"\n{'=' * 60}")
    print(f"Result: {args.frames} frames in {elapsed:.2f}s = {fps:.1f} FPS")
    print(f"Frames saved to: {out_dir}")

    # ── 6. Also test with output_type=pt for raw tensor check
    print("\n--- Direct stream() call test ---")
    wrapper.stream.output_type = "pt"
    x_input = to_tensor(input_img).to("cuda", dtype=torch.float16)
    tensor_stats(x_input, "direct_input [0,1]")

    # Convert to [-1, 1] as StreamDiffusion expects
    x_norm = x_input * 2.0 - 1.0
    tensor_stats(x_norm, "direct_input [-1,1]")

    # Prepare + call
    stream.prepare(x_norm.unsqueeze(0))
    for i in range(warmup_count):
        raw = stream()
        if i == warmup_count - 1:
            tensor_stats(raw, f"stream() raw output")

    # Save
    raw_out = raw.squeeze().cpu().float().clamp(0, 1)
    tensor_stats(raw_out, "raw_out clamped [0,1]")
    from torchvision.transforms.functional import to_pil_image
    to_pil_image(raw_out).save(out_dir / "raw_stream_output.png")
    print(f"Saved raw_stream_output.png")

    # ── Cleanup ──────────────────────────────────────────────
    del wrapper
    torch.cuda.empty_cache()
    print("\nDone. Check the images in debug_output/")


if __name__ == "__main__":
    main()
