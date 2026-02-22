"""
Debug Flux2KleinPipeline API — discover encode_prompt return, timesteps, etc.
Run: D:\TouchDiffusionMKII\.venv\Scripts\python.exe tests/flux_klein_debug.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

def main():
    import torch
    from diffusers_flux2 import Flux2KleinPipeline

    print("Loading Flux2KleinPipeline...")
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4B",
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")

    # 1. Probe encode_prompt
    print("\n--- encode_prompt ---")
    enc = pipe.encode_prompt(prompt="a cat", device="cuda", num_images_per_prompt=1)
    print(f"  type: {type(enc)}")
    if isinstance(enc, (list, tuple)):
        print(f"  len: {len(enc)}")
        for i, x in enumerate(enc):
            print(f"  enc[{i}]: {type(x).__name__} {getattr(x, 'shape', '')}")
    else:
        print(f"  enc: {enc}")

    # 2. Probe get_timesteps
    print("\n--- get_timesteps ---")
    print(f"  has get_timesteps: {hasattr(pipe, 'get_timesteps')}")
    if hasattr(pipe, 'get_timesteps'):
        print("  OK")
    else:
        print("  MISSING - check scheduler")
        print(f"  scheduler type: {type(pipe.scheduler).__name__}")
        print(f"  scheduler methods: {[m for m in dir(pipe.scheduler) if not m.startswith('_')]}")

    # 3. Probe set_timesteps
    print("\n--- scheduler.set_timesteps ---")
    import inspect
    sig = inspect.signature(pipe.scheduler.set_timesteps)
    print(f"  params: {list(sig.parameters.keys())}")

    # 4. Probe prepare_latents
    print("\n--- prepare_latents ---")
    print(f"  has prepare_latents: {hasattr(pipe, 'prepare_latents')}")

    # 5. Probe _unpack_latents
    print("\n--- _unpack_latents ---")
    print(f"  has _unpack_latents: {hasattr(pipe, '_unpack_latents')}")

    # 6. Transformer forward signature
    print("\n--- transformer.forward ---")
    import inspect
    sig = inspect.signature(pipe.transformer.forward)
    print(f"  params: {list(sig.parameters.keys())}")

    # 7. prepare_latents signature
    print("\n--- prepare_latents ---")
    sig = inspect.signature(pipe.prepare_latents)
    print(f"  params: {list(sig.parameters.keys())}")

    # 8. Has _pack / _unpack
    print("\n--- pack/unpack ---")
    for name in ("_pack_latents", "_unpack_latents"):
        print(f"  {name}: {hasattr(pipe, name)}")

    # 9. Patch check_image_input + preprocess and test tensor input
    print("\n--- pipe() with TENSOR (patched) ---")
    def _patch(pipe):
        proc = getattr(pipe, 'image_processor', None)
        if proc is None: return
        if hasattr(proc, 'check_image_input'):
            _o = proc.check_image_input
            def _check(image, *a, **kw):
                if isinstance(image, torch.Tensor): return
                return _o(image, *a, **kw)
            proc.check_image_input = _check
        if hasattr(proc, 'preprocess'):
            _o = proc.preprocess
            def _pre(image, *a, **kw):
                if isinstance(image, torch.Tensor):
                    img = image.to(device=pipe.device)
                    if img.dim() == 3: img = img.unsqueeze(0)
                    if img.max() <= 1.0: img = img * 2.0 - 1.0
                    return img
                return _o(image, *a, **kw)
            proc.preprocess = _pre
    _patch(pipe)
    img_t = torch.rand(1, 3, 512, 512, device="cuda", dtype=torch.bfloat16) * 0.5 + 0.5
    try:
        out = pipe(image=img_t, prompt="a cat", height=512, width=512, num_inference_steps=4, output_type="pt")
        print("  OK, output shape:", out.images[0].shape)
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

    # 10. Full pipe() call (PIL) to see flow
    print("\n--- pipe() with PIL (minimal) ---")
    from PIL import Image
    img = Image.new("RGB", (512, 512), (128, 128, 128))
    try:
        out = pipe(image=img, prompt="a cat", height=512, width=512, num_inference_steps=4)
        print("  OK")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

    print("\nDone.")

if __name__ == "__main__":
    main()
