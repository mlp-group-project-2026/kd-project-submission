import importlib
import platform
import sys


def try_import(module_name: str):
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"[OK] {module_name} imported (version: {version})")
        return module
    except Exception as error:
        print(f"[FAIL] {module_name} import failed: {error}")
        return None


def main():
    print("=== Environment Smoke Test ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print()

    modules_to_test = [
        "numpy",
        "pandas",
        "torch",
        "torchvision",
        "timm",
        "albumentations",
        "sklearn",
        "transformers",
        "skimage",
        "cv2",
    ]

    imported = {name: try_import(name) for name in modules_to_test}

    torch = imported.get("torch")
    if torch is None:
        print("\n[FAIL] Torch is not available. Cannot run tensor checks.")
        raise SystemExit(1)

    print("\n=== Torch Runtime Check ===")
    print(f"Torch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"CUDA device count: {device_count}")
        for index in range(device_count):
            print(f"- GPU {index}: {torch.cuda.get_device_name(index)}")

    device = "cuda" if cuda_available else "cpu"
    tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
    print(f"Tensor device: {tensor.device}")
    print(f"Tensor mean: {tensor.mean().item():.4f}")

    failed_imports = [name for name, module in imported.items() if module is None]
    if failed_imports:
        print("\n[WARN] Some imports failed:")
        for name in failed_imports:
            print(f"- {name}")
        raise SystemExit(1)

    print("\n[SUCCESS] Environment looks good.")


if __name__ == "__main__":
    main()
