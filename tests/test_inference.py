"""
Тестовый клиент: загружает config, создаёт pipeline, прогоняет чёрную картинку.
Никогда не билдит engines — только использует готовые.

Использование:
    python test_inference.py                                        # показать доступные конфиги
    python test_inference.py engines/stabilityai_sd-turbo/config_lcm_4steps_self_512x512.yaml
"""
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
STREAMDIFFUSION_SRC = REPO_ROOT / "StreamDiffusion" / "src"
if str(STREAMDIFFUSION_SRC) not in sys.path:
    sys.path.insert(0, str(STREAMDIFFUSION_SRC))


def list_configs():
    """Найти все config_*.yaml в engines/"""
    engines = REPO_ROOT / "engines"
    if not engines.exists():
        print("✗ Папка engines/ не найдена. Сначала выполните Build.")
        return []
    configs = sorted(engines.rglob("config_*.yaml"))
    if not configs:
        # Совместимость со старым config.yaml
        configs = sorted(engines.rglob("config.yaml"))
    return configs


def main():
    # ── Без аргумента — показать доступные конфиги ──────────
    if len(sys.argv) < 2:
        configs = list_configs()
        if not configs:
            print("✗ Конфиги не найдены. Запустите Build в gradio_prepare.")
            sys.exit(1)
        print("Доступные конфиги:\n")
        for i, c in enumerate(configs):
            print(f"  [{i}] {c.relative_to(REPO_ROOT)}")
        print(f"\nИспользование: python {Path(__file__).name} <путь_к_config.yaml>")
        sys.exit(0)

    # ── 1. Загрузка конфига ─────────────────────────────────
    config_path = Path(sys.argv[1])
    if not config_path.exists():
        print(f"✗ Конфиг не найден: {config_path}")
        sys.exit(1)

    from streamdiffusion.config import load_config, create_wrapper_from_config

    print(f"Конфиг: {config_path}")
    cfg = load_config(config_path)

    cfg["compile_engines_only"] = False
    cfg["build_engines_if_missing"] = False

    w, h = cfg.get("width", 512), cfg.get("height", 512)
    print(f"Модель:     {cfg.get('model_id', '?')}")
    print(f"Разрешение: {w}×{h}")
    print(f"Scheduler:  {cfg.get('scheduler', 'lcm')}  (locked={cfg.get('scheduler_locked', True)})")
    print(f"LoRA:       {cfg.get('consistency_lora', '?')}  dict={cfg.get('lora_dict', {})}")
    print(f"t_index:    {cfg.get('t_index_list', [])}")
    print(f"CFG type:   {cfg.get('cfg_type', 'self')}")

    # ── 2. Создание pipeline ────────────────────────────────
    print("\nЗагрузка engines...")
    t0 = time.time()
    wrapper = create_wrapper_from_config(cfg)
    print(f"Pipeline готов за {time.time() - t0:.1f}s")

    # ── 3. Тестовый инференс ────────────────────────────────
    import torch
    from PIL import Image

    black = Image.new("RGB", (w, h), (0, 0, 0))

    print("Прогрев (5 кадров)...")
    for _ in range(5):
        wrapper.img2img(black)

    print("Инференс...")
    t0 = time.time()
    n_frames = 20
    for _ in range(n_frames):
        result = wrapper.img2img(black)
    elapsed = time.time() - t0
    fps = n_frames / elapsed

    print(f"\n✓ {n_frames} кадров за {elapsed:.2f}s — {fps:.1f} FPS")

    # Сохраняем последний кадр
    out_path = config_path.parent / "test_output.png"
    if isinstance(result, Image.Image):
        result.save(out_path)
        print(f"✓ Результат: {out_path}")
    elif isinstance(result, torch.Tensor):
        from torchvision.transforms.functional import to_pil_image
        img = to_pil_image(result.squeeze().cpu().clamp(0, 1))
        img.save(out_path)
        print(f"✓ Результат: {out_path}")

    del wrapper
    torch.cuda.empty_cache()
    print("✓ VRAM освобождена")


if __name__ == "__main__":
    main()
