"""LoopQuant: CLDC, ILPA, and LCTC for looped transformers."""

from .hub_compat import patch_hub_pagination

# Install before importing datasets/transformers through the pipeline.
patch_hub_pagination()

from .config import LoopQuantConfig
from .pipeline import LoopQuantPipeline, quantize_looped_transformer

__version__ = "1.0.0"

__all__ = ["LoopQuantConfig", "LoopQuantPipeline", "quantize_looped_transformer"]
