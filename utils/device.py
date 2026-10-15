"""Device selection shared by training and evaluation."""
import torch


def resolve_device(requested="auto"):
    """Prefer CUDA, then Apple Metal, then CPU; explicit devices must exist."""
    mps_available = (hasattr(torch.backends, "mps")
                     and torch.backends.mps.is_available())
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("mps" if mps_available else "cpu")
    device = torch.device(requested)
    if device.type == "mps" and not mps_available:
        raise RuntimeError("MPS is unavailable. Use native arm64 Python and an "
                           "MPS-enabled PyTorch on macOS, or choose --device cpu.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Choose --device auto, mps or cpu.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {device.index} does not exist.")
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported training device: {device}")
    return device
