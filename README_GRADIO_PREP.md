# Gradio UI — подготовка моделей StreamDiffusion

Интерфейс для сборки TRT-движков и сохранения YAML-конфига. Инференс выполняется отдельным скриптом.

**Запуск:**
```bash
run.bat
# или с hot reload:
python -m gradio gradio_prepare.py
```
UI: **http://0.0.0.0:7861**

---

## Параметры: что запекается, а что — runtime

### В UI только запекаемые параметры

Runtime-параметры **убраны из UI** и задаются хардкодом в `RUNTIME_DEFAULTS` в `gradio_prepare.py`:

| Параметр | Значение по умолчанию |
|----------|-----------------------|
| `sampler` | `"normal"` |
| `guidance_scale` | `1.2` |
| `delta` | `0.7` |
| `prompt` | `""` |
| `negative_prompt` | `"blurry, low quality"` |
| `seed` | `2` |
| `conditioning_scale` (ControlNet) | `1.0` |
| `scale` (IP-Adapter) | `0.7` |

Их можно менять в инференс-скрипте через `update_stream_params()` или в config.yaml после Build.

### Запекаемые в TRT engine (без пересборки не менять)

| Параметр | Описание |
|----------|----------|
| `model_id_or_path` | Базовая модель (SD1.5, SD-Turbo, SDXL...) |
| `width` / `height` | Разрешение (кратно 8) |
| `min_batch_size` / `max_batch_size` | Зависит от t_index_list |
| `lora_dict` | LoRA (consistency + стили) |
| `use_tiny_vae` | TinyVAE |
| `scheduler` | lcm / tcd / none; при none — pipeline scheduler выбирается отдельно |
| `cfg_type` | none / full / self / initialize |
| `use_controlnet` + `controlnet_config[].model_id` | ControlNet модели |
| `use_ipadapter` + пути | IP-Adapter модель и encoder |
| `num_image_tokens` | 4 или 16 для IP-Adapter |
| `use_cached_attn` / `cache_maxframes` | Кеширование внимания |

---

## Структура Gradio UI (gradio_prepare.py)

### Checkpoint

- **model_id** — Dropdown (популярные модели) + custom path

### Sampling

- **denoise_steps** — Slider 1–8 (→ t_index_list)
- **Consistency LoRA** — Radio: lcm / tcd / none (none — без LoRA)
- **Pipeline scheduler (при none)** — Radio: lcm / tcd (виден только при none; какой scheduler использовать для turbo/distilled)
- **cfg_type** — Dropdown: none / full / self / initialize

### Resolution

- **width / height** — Sliders 256–1024
- **use_tiny_vae**, **cached_attn**, **safety** — Checkboxes
- **cache_maxframes** — Slider

### LoRA

- **lora_json** — Textbox `{"hf_id": scale, …}` (consistency LoRA подставляется автоматически по scheduler)

### ControlNet

- **use_controlnet** — Checkbox
- **cn_count** — Slider 0–3
- До 3 слотов: **model_id**, **preprocessor**, **conditioning_channels**, **enabled**
- Scale — хардкод 1.0

### IP-Adapter

- **use_ipadapter** — Checkbox
- **Model path**, **Encoder path** — Textbox
- **Tokens** (4/16), **Type** (regular/faceid)
- Scale — хардкод 0.7

### Engine output

- **Папка engines** — Textbox (default: engines)

### Кнопки

- **Build** — сборка TRT engines и сохранение config.yaml

---

## Артефакты

- **config.yaml** — в `engines/<model_slug>/config.yaml`
- **engines/** — директория с TRT-движками

Инференс-скрипт загружает config через `create_wrapper_from_config()` и может менять runtime-параметры через `update_stream_params()`.
