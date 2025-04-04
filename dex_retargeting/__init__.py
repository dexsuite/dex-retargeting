try:
    import torch
except ImportError:
    raise ImportError(
        "Torch is not installed. You can install it using the following command:\n"
        "pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        "Note that cpu-only torch already works for dex-retargeting."
    )
