"""Audio feature extractors for multimodal reasoning."""

from __future__ import annotations

from .base import (
    FeatureExtractor,
    clear_cache,
    get_cache_path,
    set_cache_subdir,
)
from .cqt import CQTExtractor
from .f0 import F0Extractor
from .stereo import StereoExtractor

# Registry of all available feature extractors
FEATURE_REGISTRY: dict[str, FeatureExtractor] = {
    "cqt": CQTExtractor(),
    "f0": F0Extractor(),
    "stereo": StereoExtractor(),
}

# List of all available feature names
AVAILABLE_FEATURES = list(FEATURE_REGISTRY.keys())


def get_features(names: list[str]) -> list[FeatureExtractor]:
    """Get feature extractors by name.

    Args:
        names: List of feature names to retrieve.

    Returns:
        List of corresponding FeatureExtractor instances.

    Raises:
        ValueError: If an unknown feature name is provided.
    """
    extractors = []
    for name in names:
        if name not in FEATURE_REGISTRY:
            raise ValueError(
                f"Unknown feature: {name}. Available: {AVAILABLE_FEATURES}"
            )
        extractors.append(FEATURE_REGISTRY[name])
    return extractors


__all__ = [
    "FeatureExtractor",
    "CQTExtractor",
    "F0Extractor",
    "StereoExtractor",
    "FEATURE_REGISTRY",
    "AVAILABLE_FEATURES",
    "get_features",
    "get_cache_path",
    "clear_cache",
    "set_cache_subdir",
]
