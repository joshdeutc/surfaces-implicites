"""Détection automatique du matériel : GPU CUDA si dispo, sinon CPU."""
import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # GPU Apple (Mac M1/M2/M3) si présent
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
