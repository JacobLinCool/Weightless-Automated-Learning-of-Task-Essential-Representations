"""CLI for MMAR audio prediction with async concurrency support."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .features import AVAILABLE_FEATURES, FeatureExtractor, get_features
from .gemini.client import generate_prediction_async
from .mmar import MMARSample, load_mmar_meta


def load_existing_ids(output_path: Path) -> set[str]:
    """Load sample IDs that have already been processed.

    Args:
        output_path: Path to the JSONL output file.

    Returns:
        Set of sample IDs already in the output file.
    """
    existing_ids: set[str] = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        if "id" in data:
                            existing_ids.add(data["id"])
                    except json.JSONDecodeError:
                        continue
    return existing_ids


class AsyncResultWriter:
    """Thread-safe async writer for JSONL output."""

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self._lock = asyncio.Lock()

    async def write(self, data: dict) -> None:
        """Write a result to the output file."""
        async with self._lock:
            with open(self.output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                f.flush()


class ProgressTracker:
    """Track progress across concurrent tasks."""

    def __init__(self, total: int):
        self.total = total
        self.processed = 0
        self.errors = 0
        self._lock = asyncio.Lock()

    async def success(self, sample_id: str, answer: str) -> None:
        async with self._lock:
            self.processed += 1
            print(
                f"  [{self.processed + self.errors}/{self.total}] {sample_id}: ✓ {answer[:50]}..."
            )

    async def error(self, sample_id: str, error: str) -> None:
        async with self._lock:
            self.errors += 1
            print(
                f"  [{self.processed + self.errors}/{self.total}] {sample_id}: ✗ {error}"
            )


async def process_sample(
    sample: MMARSample,
    data_dir: Path,
    extractors: list[FeatureExtractor],
    model: str,
    writer: AsyncResultWriter,
    progress: ProgressTracker,
    semaphore: asyncio.Semaphore,
) -> None:
    """Process a single sample with rate limiting."""
    async with semaphore:
        try:
            # Resolve audio path
            audio_path = data_dir / sample.audio_path.lstrip("./")
            if not audio_path.exists():
                await progress.error(sample.id, f"audio not found at {audio_path}")
                return

            # Load audio (I/O bound, could use aiofiles but keeping simple)
            audio_bytes = await asyncio.to_thread(audio_path.read_bytes)

            # Extract features (CPU bound, run in thread pool)
            feature_images: list[tuple[str, bytes]] = []
            for extractor in extractors:
                img_bytes = await asyncio.to_thread(extractor.extract, audio_path)
                if img_bytes:
                    feature_images.append((extractor.name, img_bytes))

            # Generate prediction (async API call)
            result = await generate_prediction_async(
                model=model,
                sample=sample,
                audio_bytes=audio_bytes,
                feature_images=feature_images if feature_images else None,
            )

            # Resolve answer prediction letter (A, B, C, D) to actual choice value
            answer_raw = result["answer_prediction"].strip().upper()
            # Convert letter to index: A=0, B=1, C=2, D=3, etc.
            if len(answer_raw) == 1 and answer_raw.isalpha():
                answer_idx = ord(answer_raw) - ord("A")
                if 0 <= answer_idx < len(sample.choices):
                    answer_value = sample.choices[answer_idx]
                else:
                    # Letter out of range, store raw
                    answer_value = result["answer_prediction"]
            else:
                # Not a single letter, store raw prediction
                answer_value = result["answer_prediction"]

            # Write result
            output_data = {
                "id": sample.id,
                "thinking_prediction": result["thinking_prediction"],
                "answer_prediction": answer_value,
            }
            await writer.write(output_data)
            await progress.success(sample.id, answer_value)

        except Exception as e:
            await progress.error(sample.id, str(e))


async def run_async(args: argparse.Namespace) -> int:
    """Run the inference pipeline asynchronously."""
    # Parse features
    feature_names: list[str] = []
    if args.features:
        if args.features == "all":
            feature_names = AVAILABLE_FEATURES
        else:
            feature_names = [f.strip() for f in args.features.split(",") if f.strip()]

    # Validate features
    if feature_names:
        try:
            extractors = get_features(feature_names)
            print(f"Enabled features: {', '.join(feature_names)}")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    else:
        extractors = []
        print("No visual features enabled (audio only)")

    # Load metadata
    meta_path = Path(args.meta)
    if not meta_path.exists():
        print(f"Error: Metadata file not found: {meta_path}", file=sys.stderr)
        return 1

    samples = load_mmar_meta(meta_path)
    print(f"Loaded {len(samples)} samples from {meta_path}")

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Auto-generate filename based on model and features
        model_name = args.model.replace("/", "_").replace(":", "_")
        if feature_names:
            features_suffix = "_" + "-".join(sorted(feature_names))
        else:
            features_suffix = ""
        output_path = Path(f"{model_name}{features_suffix}.jsonl")
        print(f"Output file: {output_path}")

    # Load existing IDs for continuation
    existing_ids = load_existing_ids(output_path)
    if existing_ids:
        print(f"Found {len(existing_ids)} already processed samples, will skip them")

    # Filter samples
    pending_samples = [s for s in samples if s.id not in existing_ids]
    if args.max_items is not None:
        pending_samples = pending_samples[: args.max_items]

    if not pending_samples:
        print("No samples to process.")
        return 0

    print(
        f"Processing {len(pending_samples)} samples with concurrency={args.concurrency}..."
    )

    # Set up async components
    data_dir = Path(args.data_dir)
    writer = AsyncResultWriter(output_path)
    progress = ProgressTracker(len(pending_samples))
    semaphore = asyncio.Semaphore(args.concurrency)

    # Create and run tasks
    tasks = [
        process_sample(
            sample=sample,
            data_dir=data_dir,
            extractors=extractors,
            model=args.model,
            writer=writer,
            progress=progress,
            semaphore=semaphore,
        )
        for sample in pending_samples
    ]

    await asyncio.gather(*tasks)

    print(f"\nDone! Processed {progress.processed} samples, {progress.errors} errors")
    print(f"Results saved to: {output_path}")

    return 0 if progress.errors == 0 else 1


def main() -> int:
    """Main entry point for the CLI."""
    p = argparse.ArgumentParser(
        description="Generate predictions for MMAR audio multiple-choice questions"
    )
    p.add_argument(
        "--meta",
        type=str,
        default="data/MMAR-meta.json",
        help="Path to MMAR-meta.json",
    )
    p.add_argument(
        "--model",
        type=str,
        default="gemini-3-flash-preview",
        help="Gemini model to use",
    )
    p.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output JSONL file path (default: auto-generated from model and features)",
    )
    p.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Maximum number of items to process (None for all)",
    )
    p.add_argument(
        "--features",
        "-F",
        type=str,
        default="",
        help=f"Comma-separated list of features to enable. Available: {', '.join(AVAILABLE_FEATURES)}. Default: audio only (no visual features)",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Base directory for audio files (default: data)",
    )
    p.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=4,
        help="Number of concurrent requests (default: 4)",
    )

    args = p.parse_args()

    return asyncio.run(run_async(args))


if __name__ == "__main__":
    sys.exit(main())
