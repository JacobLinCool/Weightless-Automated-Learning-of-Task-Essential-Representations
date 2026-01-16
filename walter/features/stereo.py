"""Stereo phase scope and panning visualization for spatial audio analysis."""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav

matplotlib.use("Agg")

from .base import FeatureExtractor


class StereoExtractor(FeatureExtractor):
    """Visualizes stereo field and panning information.

    Purpose: Shows the spatial distribution of audio between left and right channels.
    Most AI audio models process mono audio and cannot "hear" spatial direction.

    Best for:
    - "Is the train moving left to right?"
    - "Is the sound coming from behind?" (binaural)
    - Any questions about sound direction or spatial movement.
    """

    name = "stereo"
    description = "Stereo phase scope and panning plot showing L/R channel relationship. Useful for detecting sound direction and spatial movement."

    def _extract_impl(self, audio_path: Path) -> bytes | None:
        """Generate stereo visualization."""
        try:
            sample_rate, data = wav.read(audio_path)

            # Check if stereo
            if len(data.shape) == 1:
                # Mono audio - create a simple message plot
                fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
                ax.text(
                    0.5,
                    0.5,
                    "Mono Audio\n(No stereo information available)",
                    ha="center",
                    va="center",
                    fontsize=14,
                    transform=ax.transAxes,
                )
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis("off")
                ax.set_title("Stereo Analysis")

                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight")
                plt.close(fig)
                buf.seek(0)
                return buf.read()

            left = data[:, 0].astype(np.float32)
            right = data[:, 1].astype(np.float32)

            # Normalize
            max_val = max(np.abs(left).max(), np.abs(right).max())
            if max_val > 0:
                left = left / max_val
                right = right / max_val

            # Create figure with multiple visualizations
            fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=100)

            # 1. Lissajous / Phase scope (top-left)
            ax1 = axes[0, 0]
            # Subsample for visualization
            step = max(1, len(left) // 10000)
            ax1.scatter(left[::step], right[::step], alpha=0.1, s=1, c="#2196F3")
            ax1.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
            ax1.axvline(x=0, color="gray", linestyle="-", alpha=0.3)
            # Draw diagonal lines for mono reference
            ax1.plot([-1, 1], [-1, 1], "g--", alpha=0.5, label="Mono (center)")
            ax1.plot([-1, 1], [1, -1], "r--", alpha=0.5, label="Side (L-R)")
            ax1.set_xlabel("Left Channel")
            ax1.set_ylabel("Right Channel")
            ax1.set_title("Stereo Phase Scope (Lissajous)")
            ax1.set_xlim(-1.1, 1.1)
            ax1.set_ylim(-1.1, 1.1)
            ax1.set_aspect("equal")
            ax1.legend(loc="upper right", fontsize=8)

            # 2. Panning over time (top-right)
            ax2 = axes[0, 1]
            # Calculate panning: (R - L) / (R + L + epsilon)
            window_size = sample_rate // 20  # 50ms windows
            n_windows = len(left) // window_size
            panning = []
            times = []

            for i in range(n_windows):
                start = i * window_size
                end = start + window_size
                l_energy = np.sqrt(np.mean(left[start:end] ** 2))
                r_energy = np.sqrt(np.mean(right[start:end] ** 2))
                total = l_energy + r_energy + 1e-10
                pan = (r_energy - l_energy) / total  # -1 = full left, +1 = full right
                panning.append(pan)
                times.append((start + end) / 2 / sample_rate)

            ax2.plot(times, panning, color="#FF5722", linewidth=1.5)
            ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax2.fill_between(times, 0, panning, alpha=0.3, color="#FF5722")
            ax2.set_xlabel("Time (seconds)")
            ax2.set_ylabel("Pan Position (L ← 0 → R)")
            ax2.set_title("Stereo Panning Over Time")
            ax2.set_ylim(-1, 1)
            ax2.set_xlim(0, len(left) / sample_rate)
            ax2.grid(True, alpha=0.3)

            # 3. L-R waveform comparison (bottom-left)
            ax3 = axes[1, 0]
            time = np.linspace(0, len(left) / sample_rate, len(left))
            ax3.plot(
                time, left, color="#2196F3", alpha=0.7, linewidth=0.5, label="Left"
            )
            ax3.plot(
                time, right, color="#F44336", alpha=0.7, linewidth=0.5, label="Right"
            )
            ax3.set_xlabel("Time (seconds)")
            ax3.set_ylabel("Amplitude")
            ax3.set_title("Left vs Right Channel Waveforms")
            ax3.legend(loc="upper right")
            ax3.set_xlim(0, len(left) / sample_rate)
            ax3.grid(True, alpha=0.3)

            # 4. Mid-Side representation (bottom-right)
            ax4 = axes[1, 1]
            mid = (left + right) / 2
            side = (left - right) / 2

            # Calculate energy over time
            mid_energy = []
            side_energy = []
            for i in range(n_windows):
                start = i * window_size
                end = start + window_size
                mid_energy.append(np.sqrt(np.mean(mid[start:end] ** 2)))
                side_energy.append(np.sqrt(np.mean(side[start:end] ** 2)))

            ax4.plot(
                times, mid_energy, color="#4CAF50", linewidth=1.5, label="Mid (Center)"
            )
            ax4.plot(
                times,
                side_energy,
                color="#9C27B0",
                linewidth=1.5,
                label="Side (Stereo)",
            )
            ax4.set_xlabel("Time (seconds)")
            ax4.set_ylabel("Energy")
            ax4.set_title("Mid/Side Energy (Center vs Stereo Content)")
            ax4.legend(loc="upper right")
            ax4.set_xlim(0, len(left) / sample_rate)
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save to bytes
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            return buf.read()

        except Exception as e:
            print(f"Error extracting stereo: {e}")
            return None
