try:
    import torch
except ImportError:
    raise ImportError(
        "ml4vision.ml requires the pytorch library. Please run: pip install ml4vision-py[ml]"
    ) from None
