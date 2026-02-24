# TouchDiffusion MKII

**Repo:** [github.com/olegchomp/tdmkii](https://github.com/olegchomp/tdmkii)

Single Gradio app ([webui.py](webui.py)) to build **TensorRT engines** and YAML configs. Inference runs in a separate process (e.g. TouchDesigner `toe/` or your script).

**Tabs:**

| Tab | Purpose | Build entry | Output |
|-----|---------|-------------|--------|
| **StreamDiffusion** | SD/SDXL pipelines, ControlNet, IP-Adapter | `do_build` → `streamdiffusion.config.create_wrapper_from_config` | `engines/sd/<model_slug>/` + config YAML |
| **Flux Klein** | Flux Klein 4B img2img | [tools/flux_klein_build.build_flux_klein_engines](tools/flux_klein_build.py) | `engines/flux_klein/` + config YAML |
| **Depth Anything** | Depth v1/v2 | [tools/depth_anything_build](tools/depth_anything_build.py) | `engines/depth/`, checkpoints in `checkpoints/` |
| **YOLO** | Ultralytics detect/segment/pose/obb | [tools/yolo_build.build_yolo_engine](tools/yolo_build.py) | `engines/yolo/` |
| **Settings** | *(in development)* | — | — |

---

## Requirements

- Python 3.x, CUDA, NVIDIA GPU
- **Vendored in repo (no pip install):** `StreamDiffusion/`, `diffusers_ipadapter/`, `diffusers_flux2/`. `_diffusers_main/` is in the repo for the Flux Klein tab; if missing, run `git clone --depth 1 https://github.com/huggingface/diffusers.git _diffusers_main`.

---

## Installation (Windows)

1. **Clone the repo** (if you don’t have it):
   ```bash
   git clone https://github.com/olegchomp/tdmkii TouchDiffusionMKII
   cd TouchDiffusionMKII
   ```

2. **Create and activate a virtual environment** in the project folder:
   ```batch
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install PyTorch with CUDA** (match your CUDA version; see [requirements.txt](requirements.txt)):
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
   For CUDA 11.8 use `cu118`, for CUDA 12.4 use `cu124`, etc.

4. **Install dependencies:**
   ```batch
   pip install -r requirements.txt
   ```
   (TensorRT for CUDA 12 is in requirements. On CUDA 11 you need to install TensorRT separately, e.g. via StreamDiffusion’s `install-tensorrt` script.)

5. **Run the app** (with venv activated): `run.bat` (uses `.venv` in the project root).

   UI: **http://0.0.0.0:7861**

---

## Run

From the repo root (venv can be activated or use `run.bat` which uses `.venv` automatically):

```batch
run.bat
```

UI: **http://0.0.0.0:7861**

---

## Project layout

| Path | Description |
|------|-------------|
| [webui.py](webui.py) | Gradio app: all tabs, `do_build` / `do_build_flux_klein` / `do_build_depth` / `do_build_yolo` |
| [tools/](tools/) | Build scripts: `flux_klein_build.py`, `depth_anything_build.py`, `yolo_build.py` |
| `StreamDiffusion/`, `diffusers_ipadapter/`, `diffusers_flux2/` | Vendored; on `sys.path` from webui |
| `_diffusers_main/` | In repo: clone of huggingface/diffusers (Flux Klein tab only) |
| `toe/` | TouchDesigner components (inference) |
| `engines/` | TRT output: `sd/<model_slug>/`, `flux_klein/`, `depth/`, `yolo/` |
| `checkpoints/` | Depth Anything `.pth` (next to `engines/`) |
| `requirements.txt`, `run.bat` | Dependencies and launcher |

---

## UI tabs (what’s in webui)

### StreamDiffusion

- **Checkpoint:** Model ID / path (dropdown + custom).
- **Sampling:** Width/height (256–1024), denoise steps (1–8 → `t_index_list`), Consistency LoRA (lcm / tcd / none), CFG type (none / self / full / initialize).
- **LoRA:** Up to 3 slots: model_id + scale. Consistency LoRA is added automatically when scheduler is lcm/tcd.
- **ControlNet:** Up to 3 model_id textboxes (#0–#2). Conditioning scale fixed at 1.0 in config.
- **IP-Adapter:** Enable, adapter path, encoder path (dropdowns + custom), reference detail (4/16), type (regular/faceid). Scale fixed at 0.7 in config.
- **Cached attention:** Checkbox + cache max frames slider.
- **Output:** Config filename (optional; auto if empty). **Build** compiles TRT engines and writes config to `engines/sd/<model_slug>/config*.yaml`, then runs a short test inference.

**Runtime (not in UI):** Set in `RUNTIME_DEFAULTS` in webui.py; override in inference via `update_stream_params()` or by editing the saved config: `sampler`, `guidance_scale`, `delta`, `prompt`, `negative_prompt`, `seed`, `conditioning_scale`, `ipadapter_scale`.

### Flux Klein

- **Resolution:** Width / height sliders (512–1024, step 64).
- **Build:** Calls `tools.flux_klein_build.build_flux_klein_engines` → ONNX export then TRT (BF16). Writes transformer, VAE encoder, VAE decoder engines and `config_flux_klein_{w}x{h}.yaml` to `engines/flux_klein/`. Requires `_diffusers_main` and vendored `diffusers_flux2`.

### Depth Anything

- **Model:** Version (v1 / v2), size (Small/Base/Large; v2 also Giant).
- **Resolution:** Width / height (default 518; aligned to multiple of 14).
- **Build:** Uses `tools.depth_anything_build` (downloads missing checkpoints to `checkpoints/`), builds engine to `engines/depth/`. Name: `{checkpoint_name}_{width}x{height}.engine`. Use path and resolution in StreamDiffusion’s `DepthAnythingTensorrtPreprocessor`.

### YOLO

- **Model & input:** Model (preset or path), task (detect / segment / pose / obb), width / height / batch.
- **Precision:** FP16 (half), dynamic input/batch.
- **INT8:** Optional; data YAML and calibration fraction.
- **Build:** Uses `tools.yolo_build.build_yolo_engine`; output to `engines/yolo/`. Official models download on first run.

---

## Artifacts

- **StreamDiffusion:** `engines/sd/<model_slug>/config*.yaml` and TRT engines in that folder. Inference: load with `create_wrapper_from_config()`, change runtime with `update_stream_params()`.
- **Flux Klein:** `engines/flux_klein/*.engine` and `config_flux_klein_*x*.yaml`.
- **Depth Anything:** `engines/depth/*.engine`, checkpoints in `checkpoints/`.
- **YOLO:** `engines/yolo/*.engine`.

