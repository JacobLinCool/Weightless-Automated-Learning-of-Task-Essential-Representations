"""Base class for audio feature extractors with file-based caching."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

# Default cache directory (relative to audio file location)
_CACHE_SUBDIR = ".features"


def set_cache_subdir(subdir: str) -> None:
    """Set the subdirectory name for feature cache.

    Args:
        subdir: Subdirectory name (e.g., ".features" or "features").
    """
    global _CACHE_SUBDIR
    _CACHE_SUBDIR = subdir


def get_cache_path(audio_path: Path, feature_name: str) -> Path:
    """Get the cache file path for a feature.

    Cache files are stored alongside audio files:
    - audio/sample.wav -> audio/.features/sample.waveform.png
    - audio/sample.wav -> audio/.features/sample.cqt.png

    Args:
        audio_path: Path to the audio file.
        feature_name: Name of the feature extractor.

    Returns:
        Path to the cached PNG file.
    """
    cache_dir = audio_path.parent / _CACHE_SUBDIR
    return cache_dir / f"{audio_path.stem}.{feature_name}.png"


def clear_cache(audio_dir: Path | None = None) -> int:
    """Clear cached feature images.

    Args:
        audio_dir: Directory to clear cache from. If None, raises error.

    Returns:
        Number of cache files deleted.
    """
    if audio_dir is None:
        raise ValueError("Must specify audio_dir to clear cache")

    cache_dir = Path(audio_dir) / _CACHE_SUBDIR
    if not cache_dir.exists():
        return 0

    count = 0
    for f in cache_dir.glob("*.png"):
        f.unlink()
        count += 1
    return count


class FeatureExtractor(ABC):
    """Base class for audio feature extractors.

    Feature extractors take an audio file and generate a visualization
    (as PNG bytes) that aids the model in understanding specific aspects
    of the audio.

    Results are automatically cached as PNG files alongside the audio:
    - audio/sample.wav -> audio/.features/sample.{feature_name}.png
    """

    name: str
    description: str

    def extract(self, audio_path: Path, use_cache: bool = True) -> bytes | None:
        """Extract feature visualization from audio (with file caching).

        Args:
            audio_path: Path to the audio file (WAV format).
            use_cache: Whether to use cached results if available.

        Returns:
            PNG image bytes representing the feature visualization,
            or None if extraction fails.
        """
        audio_path = Path(audio_path).resolve()
        cache_path = get_cache_path(audio_path, self.name)

        # Check cache
        if use_cache and cache_path.exists():
            try:
                return cache_path.read_bytes()
            except Exception:
                pass  # Cache read failed, regenerate

        # Generate feature
        result = self._extract_impl(audio_path)

        # Save to cache
        if result is not None:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(result)
            except Exception as e:
                print(
                    f"Warning: Failed to cache {self.name} for {audio_path.name}: {e}"
                )

        return result

    @abstractmethod
    def _extract_impl(self, audio_path: Path) -> bytes | None:
        """Implementation of feature extraction (to be overridden).

        Args:
            audio_path: Absolute path to the audio file.

        Returns:
            PNG image bytes representing the feature visualization,
            or None if extraction fails.
        """
        pass
