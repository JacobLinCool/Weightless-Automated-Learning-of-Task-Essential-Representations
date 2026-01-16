"""CQT and Chromagram visualization for music theory analysis."""

from __future__ import annotations

import io
from pathlib import Path

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from .base import FeatureExtractor


class CQTExtractor(FeatureExtractor):
    """Visualizes Constant-Q Transform (CQT) and Chromagram.

    Purpose: Maps audio to musical pitches with logarithmic frequency resolution
    tuned to musical notes.

    Best for:
    - "What chord is this?"
    - "Is the melody rising?"
    - "Is this Major or Minor?"
    - Any music theory or harmonic analysis questions.
    """

    name = "cqt"
    description = "Constant-Q Transform and Chromagram showing musical pitch content. Useful for identifying chords, melody direction, and musical tonality."

    def _extract_impl(self, audio_path: Path) -> bytes | None:
        """Generate CQT and Chromagram visualization."""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=22050)

            # Create figure with two subplots
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=100)

            # CQT
            C = np.abs(
                librosa.cqt(y, sr=sr, hop_length=512, n_bins=84, bins_per_octave=12)
            )
            C_db = librosa.amplitude_to_db(C, ref=np.max)

            img1 = librosa.display.specshow(
                C_db,
                sr=sr,
                hop_length=512,
                x_axis="time",
                y_axis="cqt_note",
                ax=axes[0],
                cmap="magma",
            )
            axes[0].set_title("Constant-Q Transform (Musical Pitch Representation)")
            fig.colorbar(img1, ax=axes[0], format="%+2.0f dB")

            # Chromagram
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)

            img2 = librosa.display.specshow(
                chroma,
                sr=sr,
                hop_length=512,
                x_axis="time",
                y_axis="chroma",
                ax=axes[1],
                cmap="coolwarm",
            )
            axes[1].set_title("Chromagram (12-tone Pitch Classes)")
            fig.colorbar(img2, ax=axes[1])

            plt.tight_layout()

            # Save to bytes
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            return buf.read()

        except Exception as e:
            print(f"Error extracting CQT: {e}")
            return None
