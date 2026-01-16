"""Fundamental frequency (F0) contour for speaker and emotion analysis."""

from __future__ import annotations

import io
from pathlib import Path

import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from .base import FeatureExtractor


class F0Extractor(FeatureExtractor):
    """Visualizes fundamental frequency (pitch) contour.

    Purpose: Strips away noise and timbre, leaving only the "melody" of the voice.

    Best for:
    - "Is this a question?" (Rising tail)
    - "Are they singing or speaking?"
    - "Is the tone monotonous or excited?"
    - Any intonation, speaker emotion, or prosody questions.
    """

    name = "f0"
    description = "Fundamental frequency (F0) contour showing pitch over time. Useful for detecting question intonation, emotional expressiveness, and speech patterns."

    def _extract_impl(self, audio_path: Path) -> bytes | None:
        """Generate F0 contour visualization."""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=22050)

            # Extract F0 using pyin (probabilistic YIN)
            f0, voiced_flag, voiced_prob = librosa.pyin(
                y,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sr,
                hop_length=512,
            )

            # Create time axis
            times = librosa.times_like(f0, sr=sr, hop_length=512)

            # Create figure
            fig, axes = plt.subplots(2, 1, figsize=(12, 6), dpi=100)

            # F0 contour
            ax1 = axes[0]
            # Plot voiced regions
            voiced_times = times[voiced_flag]
            voiced_f0 = f0[voiced_flag]

            if len(voiced_f0) > 0:
                ax1.scatter(
                    voiced_times,
                    voiced_f0,
                    c=voiced_prob[voiced_flag],
                    cmap="viridis",
                    s=10,
                    alpha=0.7,
                )
                ax1.plot(
                    voiced_times, voiced_f0, color="#4CAF50", linewidth=1, alpha=0.5
                )

            ax1.set_xlabel("Time (seconds)")
            ax1.set_ylabel("Frequency (Hz)")
            ax1.set_title("F0 (Pitch) Contour")
            ax1.set_xlim(0, times[-1] if len(times) > 0 else 1)
            ax1.grid(True, alpha=0.3)

            # F0 in semitones (more intuitive for pitch changes)
            ax2 = axes[1]
            if len(voiced_f0) > 0:
                # Convert to semitones relative to mean
                mean_f0 = np.nanmean(voiced_f0)
                semitones = 12 * np.log2(voiced_f0 / mean_f0)
                ax2.plot(voiced_times, semitones, color="#FF5722", linewidth=1.5)
                ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
                ax2.fill_between(voiced_times, 0, semitones, alpha=0.3, color="#FF5722")

            ax2.set_xlabel("Time (seconds)")
            ax2.set_ylabel("Semitones (relative to mean)")
            ax2.set_title("Pitch Variation (Intonation Pattern)")
            ax2.set_xlim(0, times[-1] if len(times) > 0 else 1)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save to bytes
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            return buf.read()

        except Exception as e:
            print(f"Error extracting F0: {e}")
            return None
