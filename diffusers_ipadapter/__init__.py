"""
IPAdapter implementation for HuggingFace Diffusers (vendored).

This package provides an alternative implementation of the IPAdapter models
for Huggingface Diffusers with enhanced features including:
- Support for multiple input images
- Image weighting capabilities
- Negative input image support
- Streamlined workflow with unified IPAdapter class

Vendored in TouchDiffusionMKII with torch.load(..., weights_only=False) for PyTorch 2.6+.
"""

from .ip_adapter.ip_adapter import IPAdapter

__version__ = "0.1.0"
__author__ = "livepeer"
__all__ = ["IPAdapter"]
