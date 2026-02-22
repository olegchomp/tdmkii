# diffusers_flux2 (vendored)

Flux2KleinPipeline для Flux Klein 4B. По аналогии с diffusers_ipadapter — локальная копия, подключается через sys.path.

## Источник

- `Flux2KleinPipeline` и зависимости из [huggingface/diffusers](https://github.com/huggingface/diffusers) main
- При первом использовании требуется `_diffusers_main`: клон diffusers main в корне репо
- `tools/flux_klein_build.py` добавляет `_diffusers_main/src` в sys.path перед импортом

## Использование

```python
from diffusers_flux2 import Flux2KleinPipeline
pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", torch_dtype=torch.float32)
```

## Подготовка

Однократно в корне репо:
```
git clone --depth 1 https://github.com/huggingface/diffusers.git _diffusers_main
```
