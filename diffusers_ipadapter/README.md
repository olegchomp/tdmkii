# diffusers_ipadapter (vendored)

Версия пакета [diffusers_ipadapter](https://pypi.org/project/diffusers-ipadapter/) (livepeer), скопированная в репозиторий TouchDiffusionMKII.

## Зачем

- **PyTorch 2.6+**: в `ip_adapter/ip_adapter.py` вызов `torch.load(..., weights_only=False)`, иначе загрузка весов IP-Adapter падает с `UnpicklingError`.
- Версия из PyPI этого не исправляет; патч в репо не теряется при переустановке зависимостей.

## Использование

Корень репозитория добавлен в `sys.path` в `gradio_prepare.py` и в TouchDesigner-скрипте `SD.py`, поэтому `from diffusers_ipadapter import IPAdapter` подхватывает эту локальную копию, а не пакет из site-packages.

## Оригинал

- Автор: livepeer  
- Версия: 0.1.0  
- Изменения в нашей копии: только `weights_only=False` в `torch.load()` в `ip_adapter/ip_adapter.py`.
