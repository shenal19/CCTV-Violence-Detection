"""Phase 0 environment check.

Run from the project root:
    python -m src.check_env

This is the Phase 0 "done when": confirm CUDA is available before training.
"""
import sys


def main():
    print(f"Python: {sys.version.split()[0]}")
    try:
        import torch
        import torchvision
        print(f"torch: {torch.__version__}")
        print(f"torchvision: {torchvision.__version__}")
        cuda = torch.cuda.is_available()
        print(f"CUDA available: {cuda}")
        if cuda:
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            total = torch.cuda.get_device_properties(idx).total_memory / 1024 ** 3
            print(f"GPU: {name}")
            print(f"VRAM: {total:.1f} GB")
            if total < 5:
                print("  -> ~4GB card: keep batch_size at 2-4 with AMP; train VideoMAE on Kaggle.")
        else:
            print("  -> No CUDA GPU detected. CPU training is impractically slow; use Kaggle.")
    except ImportError as e:
        print(f"PyTorch not installed correctly: {e}")
        print("Install it from https://pytorch.org/get-started/locally/ (choose your CUDA version).")
        return

    print("--- supporting libraries ---")
    for mod, pkg in [("cv2", "opencv-python"), ("decord", "decord"),
                     ("numpy", "numpy"), ("sklearn", "scikit-learn"), ("yaml", "pyyaml")]:
        try:
            __import__(mod)
            print(f"{pkg}: OK")
        except ImportError:
            print(f"{pkg}: MISSING  (pip install {pkg})")


if __name__ == "__main__":
    main()
