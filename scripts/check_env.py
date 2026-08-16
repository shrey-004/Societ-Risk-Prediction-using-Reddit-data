"""
Phase 0 sanity check — run this FIRST on your actual A100 machine
(not in this sandbox, which has no GPU).

Usage:
    python scripts/check_env.py
"""
import sys
import importlib


REQUIRED = [
    "torch", "transformers", "datasets", "accelerate",
    "sklearn", "pandas", "openpyxl", "numpy", "yaml",
]


def check_packages():
    print("=== Package check ===")
    missing = []
    for pkg in REQUIRED:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  [OK] {pkg:<15} {ver}")
        except ImportError:
            print(f"  [MISSING] {pkg}")
            missing.append(pkg)
    if missing:
        print(f"\nInstall missing packages: pip install {' '.join(missing)}")
    return len(missing) == 0


def check_gpu():
    print("\n=== GPU check ===")
    try:
        import torch
        print(f"  torch version: {torch.__version__}")
        print(f"  torch built with CUDA: {torch.version.cuda}")
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            print(f"  CUDA available: True | {n} device(s)")
            for i in range(n):
                props = torch.cuda.get_device_properties(i)
                total_gb = props.total_memory / (1024 ** 3)
                print(f"  [{i}] {props.name} | {total_gb:.1f} GB total")
                free, total = torch.cuda.mem_get_info(i)
                print(f"      free right now: {free / (1024**3):.1f} GB / {total / (1024**3):.1f} GB")
                if free / (1024 ** 3) < 8:
                    print(f"      WARNING: <8GB free — someone else may be using this shared A100. "
                          f"Check `nvidia-smi` before launching a big training job.")
        else:
            print("  CUDA available: False — likely installed the CPU-only torch wheel by "
                  "mistake. Re-run: pip install torch==2.4.1 --index-url "
                  "https://download.pytorch.org/whl/cu121")
    except ImportError:
        print("  torch not installed — cannot check GPU")


def check_data():
    print("\n=== Data check ===")
    import os
    path = "data/raw/train.xlsx"
    if os.path.exists(path):
        print(f"  [OK] found {path}")
    else:
        print(f"  [MISSING] {path} — copy train.xlsx into data/raw/ before Phase 1")


if __name__ == "__main__":
    ok = check_packages()
    check_gpu()
    check_data()
    print("\nDone." if ok else "\nFix missing packages before proceeding to Phase 1.")
    sys.exit(0 if ok else 1)
