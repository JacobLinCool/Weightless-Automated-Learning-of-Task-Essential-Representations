"""Scoring module for MMAR prediction evaluation.

Computes accuracy metrics broken down by category, modality, and overall.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .mmar import load_mmar_meta, MMARSample


@dataclass
class ScoreStats:
    """Statistics for a scoring group."""

    correct: int = 0
    total: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    def add(self, is_correct: bool) -> None:
        self.total += 1
        if is_correct:
            self.correct += 1


@dataclass
class ScoreResults:
    """Complete scoring results."""

    overall: ScoreStats = field(default_factory=ScoreStats)
    by_category: dict[str, ScoreStats] = field(default_factory=lambda: defaultdict(ScoreStats))
    by_sub_category: dict[str, ScoreStats] = field(default_factory=lambda: defaultdict(ScoreStats))
    by_modality: dict[str, ScoreStats] = field(default_factory=lambda: defaultdict(ScoreStats))
    missing_ids: list[str] = field(default_factory=list)
    extra_ids: list[str] = field(default_factory=list)


def load_predictions(path: Path) -> dict[str, dict]:
    """Load predictions from JSONL file.

    Args:
        path: Path to the JSONL file.

    Returns:
        Dictionary mapping sample IDs to prediction data.
    """
    predictions: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    if "id" in data:
                        predictions[data["id"]] = data
                except json.JSONDecodeError:
                    continue
    return predictions


def is_correct(prediction: str, answer: str) -> bool:
    """Check if prediction matches answer.

    Handles some normalization for comparison.
    """
    # Normalize both strings
    pred = prediction.strip().lower()
    ans = answer.strip().lower()

    # Direct match
    if pred == ans:
        return True

    # Handle multi-line answers (some answers have multiple valid values)
    if "\n" in ans:
        valid_answers = [a.strip().lower() for a in ans.split("\n")]
        return pred in valid_answers

    return False


def compute_scores(
    samples: list[MMARSample],
    predictions: dict[str, dict],
) -> ScoreResults:
    """Compute scoring metrics.

    Args:
        samples: List of MMAR samples with ground truth.
        predictions: Dictionary mapping sample IDs to predictions.

    Returns:
        ScoreResults with all metrics.
    """
    results = ScoreResults()

    sample_ids = {s.id for s in samples}
    pred_ids = set(predictions.keys())

    # Track missing and extra predictions
    results.missing_ids = list(sample_ids - pred_ids)
    results.extra_ids = list(pred_ids - sample_ids)

    # Score each sample that has a prediction
    for sample in samples:
        if sample.id not in predictions:
            continue

        pred_data = predictions[sample.id]
        pred_answer = pred_data.get("answer_prediction", "")

        correct = is_correct(pred_answer, sample.answer)

        # Update overall
        results.overall.add(correct)

        # Update by category
        category = sample.category or "Unknown"
        results.by_category[category].add(correct)

        # Update by sub-category
        sub_category = sample.sub_category or "Unknown"
        results.by_sub_category[sub_category].add(correct)

        # Update by modality
        modality = sample.modality or "Unknown"
        results.by_modality[modality].add(correct)

    return results


def format_results(results: ScoreResults, show_sub_categories: bool = False) -> str:
    """Format results as a human-readable string.

    Args:
        results: The scoring results.
        show_sub_categories: Whether to show sub-category breakdown.

    Returns:
        Formatted string with results.
    """
    lines: list[str] = []

    # Overall
    lines.append("=" * 60)
    lines.append("OVERALL RESULTS")
    lines.append("=" * 60)
    lines.append(
        f"Accuracy: {results.overall.accuracy:.2%} "
        f"({results.overall.correct}/{results.overall.total})"
    )
    lines.append("")

    # By Modality
    lines.append("-" * 60)
    lines.append("BY MODALITY")
    lines.append("-" * 60)
    for modality in sorted(results.by_modality.keys()):
        stats = results.by_modality[modality]
        lines.append(
            f"  {modality:30s} {stats.accuracy:6.2%} ({stats.correct:4d}/{stats.total:4d})"
        )
    lines.append("")

    # By Category
    lines.append("-" * 60)
    lines.append("BY CATEGORY")
    lines.append("-" * 60)
    for category in sorted(results.by_category.keys()):
        stats = results.by_category[category]
        lines.append(
            f"  {category:30s} {stats.accuracy:6.2%} ({stats.correct:4d}/{stats.total:4d})"
        )
    lines.append("")

    # By Sub-Category (optional)
    if show_sub_categories:
        lines.append("-" * 60)
        lines.append("BY SUB-CATEGORY")
        lines.append("-" * 60)
        for sub_cat in sorted(results.by_sub_category.keys()):
            stats = results.by_sub_category[sub_cat]
            lines.append(
                f"  {sub_cat:40s} {stats.accuracy:6.2%} ({stats.correct:4d}/{stats.total:4d})"
            )
        lines.append("")

    # Missing/Extra
    if results.missing_ids:
        lines.append("-" * 60)
        lines.append(f"MISSING PREDICTIONS: {len(results.missing_ids)} samples")
        lines.append("-" * 60)

    if results.extra_ids:
        lines.append("-" * 60)
        lines.append(f"EXTRA PREDICTIONS: {len(results.extra_ids)} samples (not in metadata)")
        lines.append("-" * 60)

    return "\n".join(lines)


def results_to_json(results: ScoreResults) -> dict:
    """Convert results to a JSON-serializable dictionary."""
    return {
        "overall": {
            "accuracy": results.overall.accuracy,
            "correct": results.overall.correct,
            "total": results.overall.total,
        },
        "by_modality": {
            modality: {
                "accuracy": stats.accuracy,
                "correct": stats.correct,
                "total": stats.total,
            }
            for modality, stats in sorted(results.by_modality.items())
        },
        "by_category": {
            category: {
                "accuracy": stats.accuracy,
                "correct": stats.correct,
                "total": stats.total,
            }
            for category, stats in sorted(results.by_category.items())
        },
        "by_sub_category": {
            sub_cat: {
                "accuracy": stats.accuracy,
                "correct": stats.correct,
                "total": stats.total,
            }
            for sub_cat, stats in sorted(results.by_sub_category.items())
        },
        "missing_count": len(results.missing_ids),
        "extra_count": len(results.extra_ids),
    }


def main() -> int:
    """Main entry point for the scoring CLI."""
    parser = argparse.ArgumentParser(
        description="Score MMAR prediction results against ground truth"
    )
    parser.add_argument(
        "predictions",
        type=str,
        help="Path to the predictions JSONL file",
    )
    parser.add_argument(
        "--meta",
        type=str,
        default="data/MMAR-meta.json",
        help="Path to MMAR-meta.json (default: data/MMAR-meta.json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (default: print to stdout)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of human-readable format",
    )
    parser.add_argument(
        "--sub-categories",
        "-s",
        action="store_true",
        help="Include sub-category breakdown in human-readable output",
    )

    args = parser.parse_args()

    # Load metadata
    meta_path = Path(args.meta)
    if not meta_path.exists():
        print(f"Error: Metadata file not found: {meta_path}", file=sys.stderr)
        return 1

    samples = load_mmar_meta(meta_path)
    print(f"Loaded {len(samples)} samples from {meta_path}", file=sys.stderr)

    # Load predictions
    pred_path = Path(args.predictions)
    if not pred_path.exists():
        print(f"Error: Predictions file not found: {pred_path}", file=sys.stderr)
        return 1

    predictions = load_predictions(pred_path)
    print(f"Loaded {len(predictions)} predictions from {pred_path}", file=sys.stderr)

    # Compute scores
    results = compute_scores(samples, predictions)

    # Format output
    if args.json:
        output = json.dumps(results_to_json(results), indent=2, ensure_ascii=False)
    else:
        output = format_results(results, show_sub_categories=args.sub_categories)

    # Write output
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(output, encoding="utf-8")
        print(f"Results saved to: {output_path}", file=sys.stderr)
    else:
        print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
