"""
Gradio UI for StreamDiffusion model preparation.
Builds TRT engines and saves YAML config. Inference is handled by a separate script.
"""
from __future__ import annotations

import io
import json
import re
import sys
import traceback
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
STREAMDIFFUSION_SRC = REPO_ROOT / "StreamDiffusion" / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(STREAMDIFFUSION_SRC) not in sys.path:
    sys.path.insert(0, str(STREAMDIFFUSION_SRC))

import gradio as gr

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"The 'theme' parameter in the Blocks constructor",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"The 'css' parameter in the Blocks constructor",
)

DEFAULT_MODELS = [
    "stabilityai/sd-turbo",
    "stabilityai/sdxl-turbo",
    "KBlueLeaf/kohaku-v2.1",
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-xl-base-1.0",
]


CSS = """
.compact{gap:6px!important}
.act-btn{min-height:40px!important;font-weight:600!important}
.log textarea{font-family:Consolas,'Courier New',monospace!important;font-size:.82em!important}
footer{visibility:hidden!important}
.sec-title{padding:6px 0 4px 6px!important}
/* hide scrollbars on string/text fields */
textarea, input[type="text"]{scrollbar-width:none!important;-ms-overflow-style:none!important}
textarea::-webkit-scrollbar, input[type="text"]::-webkit-scrollbar{display:none!important}
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.orange,
    secondary_hue=gr.themes.colors.gray,
    neutral_hue=gr.themes.colors.gray,
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#0b0f19",
    body_background_fill_dark="#0b0f19",
    block_background_fill="#1a1c2e",
    block_background_fill_dark="#1a1c2e",
    block_border_color="#2e3045",
    block_border_color_dark="#2e3045",
    block_label_text_color="#c5c7d5",
    block_label_text_color_dark="#c5c7d5",
    block_title_text_color="#e5e7eb",
    block_title_text_color_dark="#e5e7eb",
    body_text_color="#c5c7d5",
    body_text_color_dark="#c5c7d5",
    input_background_fill="#111322",
    input_background_fill_dark="#111322",
    input_border_color="#2e3045",
    input_border_color_dark="#2e3045",
    button_primary_background_fill="#f59e0b",
    button_primary_background_fill_dark="#f59e0b",
    button_primary_text_color="#000",
    button_primary_text_color_dark="#000",
    button_secondary_background_fill="#2e3045",
    button_secondary_background_fill_dark="#2e3045",
    button_secondary_text_color="#c5c7d5",
    button_secondary_text_color_dark="#c5c7d5",
    slider_color="#f59e0b",
    slider_color_dark="#f59e0b",
    checkbox_background_color="#111322",
    checkbox_background_color_dark="#111322",
    checkbox_border_color="#2e3045",
    checkbox_border_color_dark="#2e3045",
)


def _model_slug(model_id: str) -> str:
    mid = (model_id or "stabilityai/sd-turbo").strip()
    slug = mid.replace("/", "_").replace("\\", "_")
    return re.sub(r"[^\w\-.]", "_", slug) or "model"


def _config_name(cfg: dict) -> str:
    parts = [
        "config",
        cfg.get("consistency_lora", "lcm"),
        f"{len(cfg.get('t_index_list', [4]))}steps",
        cfg.get("cfg_type", "self"),
        f"{cfg.get('width', 512)}x{cfg.get('height', 512)}",
    ]
    return "_".join(parts) + ".yaml"


LCM_LORA = {"sd15": "latent-consistency/lcm-lora-sdv1-5", "sdxl": "latent-consistency/lcm-lora-sdxl"}
TCD_LORA = {"sd15": "h1t/TCD-SD15-LoRA", "sd21": "h1t/TCD-SD21-base-LoRA", "sdxl": "h1t/TCD-SDXL-LoRA"}


def _model_tier(model_id: str) -> str:
    mid = (model_id or "").lower()
    if "sdxl" in mid or "xl-base" in mid:
        return "sdxl"
    if "2.1" in mid or "sd21" in mid or "sd2.1" in mid:
        return "sd21"
    return "sd15"



IP_ADAPTER_CHOICES = [
    ("SD 1.5 — base", "h94/IP-Adapter/models/ip-adapter_sd15.safetensors"),
    ("SD 1.5 — plus", "h94/IP-Adapter/models/ip-adapter-plus_sd15.safetensors"),
    ("SD 1.5 — light", "h94/IP-Adapter/models/ip-adapter_sd15_light.safetensors"),
    ("SD 1.5 — plus face", "h94/IP-Adapter/models/ip-adapter-plus-face_sd15.safetensors"),
    ("SDXL — base (ViT-bigG)", "h94/IP-Adapter/sdxl_models/ip-adapter_sdxl.safetensors"),
    ("SDXL — ViT-H", "h94/IP-Adapter/sdxl_models/ip-adapter_sdxl_vit-h.safetensors"),
    ("SDXL — plus ViT-H", "h94/IP-Adapter/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors"),
    ("SDXL — plus face ViT-H", "h94/IP-Adapter/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors"),
]
_IP_ADAPTER_MAP = {label: val for label, val in IP_ADAPTER_CHOICES}
IP_ENCODER_CHOICES = [
    ("SD 1.5 (OpenCLIP-ViT-H-14)", "h94/IP-Adapter/models/image_encoder"),
    ("SDXL (OpenCLIP-ViT-bigG-14)", "h94/IP-Adapter/sdxl_models/image_encoder"),
]
_IP_ENCODER_MAP = {label: val for label, val in IP_ENCODER_CHOICES}

RUNTIME_DEFAULTS = {
    "sampler": "normal",
    "guidance_scale": 1.2,
    "delta": 0.7,
    "prompt": "",
    "negative_prompt": "blurry, low quality",
    "seed": 2,
    "conditioning_scale": 1.0,
    "ipadapter_scale": 0.7,
}

DENOISE_PRESETS = {
    1: [45],
    2: [35, 45],
    3: [22, 32, 45],
    4: [16, 28, 38, 45],
    5: [18, 25, 32, 38, 45],
    6: [20, 25, 30, 35, 40, 45],
    7: [20, 24, 28, 32, 36, 40, 45],
    8: [20, 23, 27, 30, 34, 37, 41, 45],
}


def _steps_to_t_index_list(steps: int) -> list[int]:
    steps = max(1, min(8, int(steps)))
    return DENOISE_PRESETS[steps]


def ui_to_config(
    model_id, w, h, cfg_type, scheduler,
    lora0_id, lora0_scale, lora1_id, lora1_scale, lora2_id, lora2_scale,
    cn0_id, cn1_id, cn2_id,
    use_ip, ip_path, ip_enc, ip_tokens, ip_type,
    cached_attn, cache_frames,
    denoise_steps,
):
    base = str(REPO_ROOT / "engines")
    slug = _model_slug(model_id)
    engine_dir = f"{base}/sd/{slug}"

    t_index_list = _steps_to_t_index_list(int(denoise_steps))
    batch = len(t_index_list)

    rt = RUNTIME_DEFAULTS
    cfg_scheduler = "lcm" if scheduler == "none" else scheduler
    scheduler_locked = scheduler != "none"
    cfg = {
        "model_id": model_id or "stabilityai/sd-turbo",
        "mode": "img2img",
        "width": int(w), "height": int(h),
        "cfg_type": cfg_type,
        "use_tiny_vae": True,
        "frame_buffer_size": 1,
        "consistency_lora": scheduler,
        "scheduler_locked": scheduler_locked,
        "scheduler": cfg_scheduler, "sampler": rt["sampler"],
        "device": "cuda", "dtype": "float16", "acceleration": "tensorrt",
        "engine_dir": engine_dir,
        "min_batch_size": batch, "max_batch_size": batch,
        "use_cached_attn": bool(cached_attn),
        "cache_maxframes": int(cache_frames),
        "use_safety_checker": False,
        "prompt": rt["prompt"], "negative_prompt": rt["negative_prompt"],
        "guidance_scale": rt["guidance_scale"], "delta": rt["delta"],
        "num_inference_steps": 50, "seed": rt["seed"],
        "t_index_list": t_index_list,
    }
    ld = {}
    for lid, lsc in [(lora0_id, lora0_scale), (lora1_id, lora1_scale), (lora2_id, lora2_scale)]:
        name = str(lid).strip() if lid else ""
        if name:
            ld[name] = float(lsc) if lsc is not None else 1.0
    if scheduler != "none":
        tier = _model_tier(model_id)
        if scheduler == "tcd":
            ld[TCD_LORA.get(tier, TCD_LORA["sd15"])] = 1.0
        else:
            ld[LCM_LORA.get(tier, LCM_LORA["sd15"])] = 1.0
    if ld:
        cfg["lora_dict"] = ld
    ids = [cn0_id, cn1_id, cn2_id]
    active = [str(mid).strip() for mid in ids if mid and str(mid).strip()]
    cn_count = len(active)
    use_controlnet = cn_count > 0
    cfg["use_controlnet"] = use_controlnet
    cfg["controlnets"] = [
        {
            "model_id": mid,
            "conditioning_scale": rt["conditioning_scale"],
            "preprocessor": "passthrough",
            "conditioning_channels": 3,
            "enabled": True,
        }
        for mid in active
    ]
    resolved_ip = _IP_ADAPTER_MAP.get(ip_path, ip_path or "").strip()
    resolved_enc = _IP_ENCODER_MAP.get(ip_enc, ip_enc or "").strip()
    if use_ip and resolved_ip and resolved_enc:
        cfg["use_ipadapter"] = True
        cfg["ipadapters"] = [{
            "ipadapter_model_path": resolved_ip,
            "image_encoder_path": resolved_enc,
            "scale": rt["ipadapter_scale"],
            "num_image_tokens": int(ip_tokens),
            "type": ip_type, "enabled": True,
        }]
    else:
        cfg["use_ipadapter"] = False
    return cfg


def _sanitize_config_name(name: str) -> str:
    s = str(name).strip()
    if not s:
        return ""
    s = re.sub(r"[^\w\-.]", "_", s)
    return s + ".yaml" if not s.lower().endswith(".yaml") else s


def do_build(*args, progress=gr.Progress()):
    buf = io.StringIO()
    try:
        config_name_input = args[-1]
        cfg = ui_to_config(*args[:-1])

        cfg["compile_engines_only"] = True
        progress(0.1, desc="Building TRT engines...")
        from streamdiffusion.config import create_wrapper_from_config, save_config
        import contextlib, torch
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            build_wrapper = create_wrapper_from_config(cfg)
        del build_wrapper
        torch.cuda.empty_cache()

        cfg_name = _sanitize_config_name(config_name_input) or _config_name(cfg)
        config_path = Path(cfg["engine_dir"]) / cfg_name
        save_config(cfg, str(config_path))
        build_log = buf.getvalue()

        progress(0.8, desc="Test inference...")
        from PIL import Image
        test_cfg = dict(cfg)
        test_cfg["compile_engines_only"] = False
        test_cfg["output_type"] = "pil"
        buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2), contextlib.redirect_stderr(buf2):
            wrapper = create_wrapper_from_config(test_cfg)
        w, h = int(cfg["width"]), int(cfg["height"])
        black = Image.new("RGB", (w, h), (0, 0, 0))
        wrapper.prepare(
            cfg.get("prompt", ""),
            cfg.get("negative_prompt", ""),
            num_inference_steps=cfg.get("num_inference_steps", 50),
            guidance_scale=cfg.get("guidance_scale", 1.2),
            delta=cfg.get("delta", 1.0),
        )
        for _ in range(wrapper.stream.batch_size):
            wrapper.img2img(black)
        result = wrapper.img2img(black)
        if isinstance(result, Image.Image):
            test_status = f"✓ Test OK — {result.size[0]}×{result.size[1]}"
        elif isinstance(result, torch.Tensor):
            test_status = f"✓ Test OK — tensor {tuple(result.shape)}"
        else:
            test_status = f"✓ Test OK — {type(result).__name__}"
        del wrapper
        torch.cuda.empty_cache()

        progress(1.0, desc="Done")
        return build_log + "\n" + test_status + f"\n✓ Config: {config_path}\n✓ Done."
    except Exception:
        return buf.getvalue() + "\n✗ " + traceback.format_exc()


def do_build_depth(
    model_version_str,
    model_size,
    width,
    height,
    progress=gr.Progress(),
):
    """Build Depth Anything TensorRT engine. Checkpoints downloaded from HF if missing."""
    buf = io.StringIO()

    def log(msg):
        buf.write(msg + "\n")

    try:
        model_version = int(model_version_str)
        width = max(14, int(width))
        height = max(14, int(height))
        checkpoints_dir = str(REPO_ROOT / "checkpoints")
        engine_output_dir = REPO_ROOT / "engines" / "depth"
        engine_output_dir.mkdir(parents=True, exist_ok=True)

        progress(0.0, desc="Ensuring checkpoint...")
        from tools.depth_anything_build import ensure_checkpoint, build_depth_engine

        checkpoint_path = ensure_checkpoint(
            model_version, model_size, checkpoints_dir, log_fn=log
        )
        progress(0.05, desc="Building engine...")
        engine_path = build_depth_engine(
            model_version,
            model_size,
            width,
            height,
            checkpoint_path,
            str(engine_output_dir),
            progress_callback=lambda p, desc: progress(p, desc=desc),
            log_fn=log,
        )
        log(f"✓ Engine: {engine_path}")
        log("Engine name: {checkpoint_name}_{width}x{height}.engine (e.g. depth_anything_vits14_518x518.engine).")
        log("Use engine_path in DepthAnythingTensorrtPreprocessor, detect_resolution = engine input size.")
        return buf.getvalue() + "✓ Done."
    except Exception:
        return buf.getvalue() + "\n✗ " + traceback.format_exc()


def do_build_yolo(
    model_spec,
    task,
    imgsz_w,
    imgsz_h,
    batch,
    half,
    int8,
    dynamic,
    data,
    fraction,
    progress=gr.Progress(),
):
    """Build Ultralytics YOLO TensorRT engine. Official models download on first use."""
    buf = io.StringIO()

    def log(msg):
        buf.write(msg + "\n")

    try:
        def _round32(x):
            x = max(320, min(1280, int(x)))
            return (x // 32) * 32
        w = _round32(imgsz_w)
        h = _round32(imgsz_h)
        imgsz = (h, w)
        batch = max(1, min(64, int(batch)))
        fraction_val = 1.0
        if fraction is not None:
            try:
                fraction_val = max(0.01, min(1.0, float(fraction)))
            except (TypeError, ValueError):
                pass
        data_str = (data or "").strip() or None
        engine_output_dir = REPO_ROOT / "engines" / "yolo"
        engine_output_dir.mkdir(parents=True, exist_ok=True)

        from tools.yolo_build import build_yolo_engine

        engine_path = build_yolo_engine(
            model_spec=model_spec or "yolo11n.pt",
            imgsz=imgsz,
            batch=batch,
            half=bool(half),
            int8=bool(int8),
            dynamic=bool(dynamic),
            workspace=None,
            simplify=True,
            data=data_str,
            fraction=fraction_val,
            engine_output_dir=str(engine_output_dir),
            task=(task or "").strip() or None,
            progress_callback=lambda p, desc: progress(p, desc=desc),
            log_fn=log,
        )
        log(f"✓ Engine: {engine_path}")
        log("Engines saved to engines/yolo/. Name: {model}_{h}x{w}_b{batch}[_fp16].engine")
        log("✓ YOLO build and test inference complete.")
        return buf.getvalue() + "✓ Done."
    except Exception:
        return buf.getvalue() + "\n✗ " + traceback.format_exc()


def do_build_flux_klein(flux_width, flux_height, progress=gr.Progress()):
    """Build Flux Klein 4B TRT engines (transformer + VAE). Requires _diffusers_main."""
    buf = io.StringIO()

    def log(msg):
        buf.write(msg + "\n")

    try:
        w = (max(64, min(2048, int(flux_width or 512))) // 64) * 64
        h = (max(64, min(2048, int(flux_height or 512))) // 64) * 64
        engine_output_dir = REPO_ROOT / "engines" / "flux_klein"
        engine_output_dir.mkdir(parents=True, exist_ok=True)

        from tools.flux_klein_build import build_flux_klein_engines

        tr_path, vae_enc_path, vae_dec_path = build_flux_klein_engines(
            engine_output_dir=str(engine_output_dir),
            width=w,
            height=h,
            progress_callback=lambda p, desc: progress(p, desc=desc),
            log_fn=log,
        )
        config_path = engine_output_dir / f"config_flux_klein_{w}x{h}.yaml"
        import yaml
        cfg = {
            "model_id": "black-forest-labs/FLUX.2-klein-4B",
            "transformer_engine": str(tr_path),
            "vae_encoder_engine": str(vae_enc_path),
            "vae_engine": str(vae_dec_path),
            "width": w,
            "height": h,
        }
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, indent=2)
        log(f"✓ Config: {config_path}")
        return buf.getvalue() + "✓ Done."
    except Exception:
        return buf.getvalue() + "\n✗ " + traceback.format_exc()


def build_app():
    with gr.Blocks(title="StreamDiffusion Prep", css=CSS, theme=THEME) as app:
        with gr.Tabs():
            with gr.Tab("StreamDiffusion"):
                with gr.Row():
                    with gr.Column(scale=1):

                        with gr.Group():
                            gr.Markdown("**Checkpoint**", elem_classes=["sec-title"])
                            model_id = gr.Dropdown(
                                DEFAULT_MODELS, value=DEFAULT_MODELS[0],
                                label="Model ID / path", allow_custom_value=True,
                            )

                        with gr.Group():
                            gr.Markdown("**Sampling**", elem_classes=["sec-title"])
                            with gr.Row(elem_classes=["compact"]):
                                width = gr.Slider(256, 1024, 512, step=64, label="Width")
                                height = gr.Slider(256, 1024, 512, step=64, label="Height")
                            denoise_steps = gr.Slider(
                                1, 8, 4, step=1,
                                label="Denoise steps",
                                info="1=turbo, 4=default (like Steps in ComfyUI)",
                            )
                            scheduler = gr.Radio(
                                ["lcm", "tcd", "none"], value="lcm", label="Consistency LoRA",
                                info="lcm/tcd — LoRA baked in; none — no LoRA (turbo, distilled)",
                            )
                            with gr.Row(elem_classes=["compact"]):
                                cfg_type = gr.Dropdown(
                                    ["none", "self", "full", "initialize"],
                                    value="none", label="CFG type",
                                    info="baked (batch structure)",
                                )

                        with gr.Group():
                            gr.Markdown("**LoRA**", elem_classes=["sec-title"])
                            with gr.Row(elem_classes=["compact"]):
                                lora0_id = gr.Textbox("", label="#0 model_id", scale=3)
                                lora0_scale = gr.Number(1.0, label="scale", minimum=0, maximum=2, step=0.05, scale=1)
                            with gr.Row(elem_classes=["compact"]):
                                lora1_id = gr.Textbox("", label="#1 model_id", scale=3)
                                lora1_scale = gr.Number(1.0, label="scale", minimum=0, maximum=2, step=0.05, scale=1)
                            with gr.Row(elem_classes=["compact"]):
                                lora2_id = gr.Textbox("", label="#2 model_id", scale=3)
                                lora2_scale = gr.Number(1.0, label="scale", minimum=0, maximum=2, step=0.05, scale=1)

                    with gr.Column(scale=1):

                        with gr.Group():
                            gr.Markdown("**ControlNet**", elem_classes=["sec-title"])
                            cn0_id = gr.Textbox("", label="#0 model_id")
                            cn1_id = gr.Textbox("", label="#1 model_id")
                            cn2_id = gr.Textbox("", label="#2 model_id")

                        with gr.Group():
                            gr.Markdown("**IP-Adapter**", elem_classes=["sec-title"])
                            use_ipadapter = gr.Checkbox(False, label="Enable")
                            with gr.Row(elem_classes=["compact"]):
                                ip_path = gr.Dropdown(
                                    choices=IP_ADAPTER_CHOICES,
                                    value="h94/IP-Adapter/models/ip-adapter_sd15.safetensors",
                                    allow_custom_value=True,
                                    label="Adapter model path",
                                    scale=1,
                                )
                                ip_enc = gr.Dropdown(
                                    choices=IP_ENCODER_CHOICES,
                                    value="h94/IP-Adapter/models/image_encoder",
                                    allow_custom_value=True,
                                    label="Encoder path (CLIP)",
                                    scale=1,
                                )
                            with gr.Row(elem_classes=["compact"]):
                                ip_tokens = gr.Dropdown(
                                    [4, 16], value=4,
                                    label="Reference detail",
                                    info="4 = base detail, 16 = high (stronger style transfer). Fixed at engine build.",
                                )
                                ip_type = gr.Radio(["regular", "faceid"], value="regular", label="Type")

                        with gr.Group():
                            gr.Markdown("**Cached attention**", elem_classes=["sec-title"])
                            cached_attn = gr.Checkbox(False, label="Cached attn")
                            cache_frames = gr.Slider(1, 16, 1, step=1, label="Cache max frames")

                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("**Output**", elem_classes=["sec-title"])
                            config_name = gr.Textbox(
                                "",
                                label="Config filename",
                                placeholder="config_lcm_4steps_none_512x512 (empty = auto)",
                                info=".yaml is added automatically",
                            )
                            btn_build = gr.Button("Build", variant="primary", elem_classes=["act-btn"])
                            log = gr.Textbox(label="Log", lines=8, max_lines=24, interactive=False, elem_classes=["log"])

                all_fields = [
                    model_id, width, height, cfg_type, scheduler,
                    lora0_id, lora0_scale, lora1_id, lora1_scale, lora2_id, lora2_scale,
                    cn0_id, cn1_id, cn2_id,
                    use_ipadapter, ip_path, ip_enc, ip_tokens, ip_type,
                    cached_attn, cache_frames,
                    denoise_steps,
                    config_name,
                ]
                btn_build.click(fn=do_build, inputs=all_fields, outputs=[log])

            with gr.Tab("Flux Klein"):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("**Resolution**", elem_classes=["sec-title"])
                            with gr.Row(elem_classes=["compact"]):
                                flux_width = gr.Slider(512, 1024, 512, step=64, label="Width")
                                flux_height = gr.Slider(512, 1024, 512, step=64, label="Height")
                            flux_btn_build = gr.Button("Build", variant="primary", elem_classes=["act-btn"])
                            flux_log = gr.Textbox(
                                label="Log", lines=10, max_lines=24, interactive=False, elem_classes=["log"],
                            )
                    with gr.Column(scale=1):
                        flux_engines_dir = REPO_ROOT / "engines" / "flux_klein"
                        gr.Markdown(
                            f"**Engines:** `{flux_engines_dir}`\n\n"
                            "Requires `_diffusers_main` (diffusers main):\n"
                            "`git clone --depth 1 https://github.com/huggingface/diffusers.git _diffusers_main`\n\n"
                            "Flux Klein: BF16, img2img. Workspace — auto. Output: transformer, vae_encoder, vae_decoder engines.",
                            elem_classes=["sec-title"],
                        )
                flux_btn_build.click(
                    fn=do_build_flux_klein,
                    inputs=[flux_width, flux_height],
                    outputs=[flux_log],
                )

            with gr.Tab("DepthAnything"):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("**Model**", elem_classes=["sec-title"])
                            depth_version = gr.Radio(
                                [("Depth Anything v1", "1"), ("Depth Anything v2", "2")],
                                value="1",
                                label="Version",
                            )
                            depth_size = gr.Dropdown(
                                [("Small", "s"), ("Base", "b"), ("Large", "l")],
                                value="s",
                                label="Size",
                                info="v1: Small/Base/Large; v2: + Giant (not on HF yet)",
                            )
                        with gr.Group():
                            gr.Markdown("**Resolution**", elem_classes=["sec-title"])
                            with gr.Row(elem_classes=["compact"]):
                                depth_width = gr.Number(518, label="Width", minimum=14, maximum=2048, step=14)
                                depth_height = gr.Number(518, label="Height", minimum=14, maximum=2048, step=14)
                            gr.Markdown("Rounded to multiple of 14 for ViT.", elem_classes=["sec-title"])
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("**Output**", elem_classes=["sec-title"])
                            depth_btn_build = gr.Button("Build", variant="primary", elem_classes=["act-btn"])
                            depth_log = gr.Textbox(label="Log", lines=10, max_lines=24, interactive=False, elem_classes=["log"])
                depth_engines_dir = REPO_ROOT / "engines" / "depth"
                gr.Markdown(
                    f"**Checkpoints:** `checkpoints/` (next to `engines/`). "
                    f"**Engines:** `{depth_engines_dir}`. "
                    "Engine filename: **{checkpoint_name}_{width}x{height}.engine** (e.g. `depth_anything_vits14_518x518.engine`, `depth_anything_v2_vits_518x518.engine`)."
                )

                def depth_version_change(ver):
                    if ver == "2":
                        return gr.update(
                            choices=[("Small", "s"), ("Base", "b"), ("Large", "l"), ("Giant", "g")],
                            value="s",
                        )
                    return gr.update(
                        choices=[("Small", "s"), ("Base", "b"), ("Large", "l")],
                        value="s",
                    )

                depth_version.change(
                    fn=depth_version_change,
                    inputs=[depth_version],
                    outputs=[depth_size],
                )
                depth_btn_build.click(
                    fn=do_build_depth,
                    inputs=[depth_version, depth_size, depth_width, depth_height],
                    outputs=[depth_log],
                )

            with gr.Tab("YOLO"):
                YOLO_PRESETS = [
                    "yolo11n.pt",
                    "yolo11s.pt",
                    "yolo11m.pt",
                    "yolo11l.pt",
                    "yolo11x.pt",
                    "yolo8n.pt",
                    "yolo8s.pt",
                    "yolo8m.pt",
                    "yolo8l.pt",
                    "yolo8x.pt",
                    "yolo11n-pose.pt",
                    "yolo11s-pose.pt",
                    "yolo8n-pose.pt",
                    "yolo8n-seg.pt",
                ]
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("**Model & input**", elem_classes=["sec-title"])
                            yolo_model = gr.Dropdown(
                                YOLO_PRESETS,
                                value="yolo11n.pt",
                                label="Model",
                                allow_custom_value=True,
                                info="Official name or path to .pt",
                            )
                            yolo_task = gr.Dropdown(
                                ["detect", "segment", "pose", "obb"],
                                value="detect",
                                label="Task",
                                info="Must match model type; set explicitly for custom model names",
                            )
                            with gr.Row(elem_classes=["compact"]):
                                yolo_imgsz_w = gr.Number(640, label="Width", minimum=320, maximum=1280, step=32)
                                yolo_imgsz_h = gr.Number(640, label="Height", minimum=320, maximum=1280, step=32)
                                yolo_batch = gr.Number(1, label="batch", minimum=1, maximum=64, step=1)
                        with gr.Group():
                            gr.Markdown("**Precision & build**", elem_classes=["sec-title"])
                            yolo_half = gr.Checkbox(True, label="FP16 (half)")
                            yolo_dynamic = gr.Checkbox(False, label="Dynamic input/batch")
                        with gr.Group():
                            gr.Markdown("**INT8 calibration**", elem_classes=["sec-title"])
                            yolo_int8 = gr.Checkbox(False, label="Enable INT8")
                            yolo_data = gr.Textbox(
                                "",
                                label="Data YAML",
                                placeholder="coco8.yaml or path to dataset yaml",
                            )
                            yolo_fraction = gr.Number(1.0, label="Calibration fraction", minimum=0.01, maximum=1.0, step=0.1)
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("**Output**", elem_classes=["sec-title"])
                            yolo_btn_build = gr.Button("Build", variant="primary", elem_classes=["act-btn"])
                            yolo_log = gr.Textbox(label="Log", lines=10, max_lines=24, interactive=False, elem_classes=["log"])
                yolo_engines_dir = REPO_ROOT / "engines" / "yolo"
                gr.Markdown(
                    f"**Engines:** `{yolo_engines_dir}`. "
                    "Name: **{model_stem}_{height}x{width}_b{batch}[_fp16].engine**. "
                    "Official models (yolo11n.pt etc.) download on first run."
                )
                yolo_btn_build.click(
                    fn=do_build_yolo,
                    inputs=[
                        yolo_model,
                        yolo_task,
                        yolo_imgsz_w,
                        yolo_imgsz_h,
                        yolo_batch,
                        yolo_half,
                        yolo_int8,
                        yolo_dynamic,
                        yolo_data,
                        yolo_fraction,
                    ],
                    outputs=[yolo_log],
                )

            with gr.Tab("Settings"):
                from tools.install_update import run_install_update

                with gr.Row():
                    with gr.Column(scale=1):
                        settings_btn = gr.Button("Install & Update", variant="primary", elem_classes=["act-btn"])
                    with gr.Column(scale=1):
                        settings_log = gr.Textbox(
                            label="Log",
                            lines=14,
                            max_lines=30,
                            interactive=False,
                            elem_classes=["log"],
                        )

                def do_install_update(progress: gr.Progress = gr.Progress()) -> str:
                    progress_cb = (lambda p, d: progress(p, desc=d)) if progress else None
                    return run_install_update(REPO_ROOT, None, cuda_ver="cu121", progress_callback=progress_cb)

                settings_btn.click(
                    fn=do_install_update,
                    inputs=[],
                    outputs=[settings_log],
                )

    return app


demo = build_app()

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861)
