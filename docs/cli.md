# CLI Usage

The `walter` CLI generates predictions for MMAR audio multiple-choice questions using Gemini AI.

## Basic Usage

```bash
# Run inference on all samples
uv run -m walter

# Run on first 10 samples
uv run -m walter --max-items 10

# Specify output file
uv run -m walter --output my_results.jsonl
```

## Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--meta` | | `data/MMAR-meta.json` | Path to MMAR metadata file |
| `--model` | | `gemini-3-flash-preview` | Gemini model to use |
| `--output` | `-o` | `results.jsonl` | Output JSONL file path |
| `--max-items` | | None (all) | Maximum samples to process |
| `--features` | `-F` | (none) | Comma-separated feature list |
| `--data-dir` | | `data` | Base directory for audio files |
| `--concurrency` | `-c` | `4` | Number of concurrent API requests |

## Features

Enable visual feature extractors to help the model reason about specific audio aspects:

```bash
# Enable CQT for music theory questions  
uv run -m walter -F cqt

# Enable F0 for speaker/emotion analysis
uv run -m walter -F f0

# Enable stereo for spatial audio
uv run -m walter -F stereo

# Enable multiple features
uv run -m walter -F cqt,f0,stereo
```

See [features.md](./features.md) for detailed feature descriptions.

## Output Format

The output is a JSONL file where each line is a JSON object:

```json
{
  "id": "sample_id_here",
  "thinking_prediction": "The model's chain-of-thought reasoning...",
  "answer_prediction": "A"
}
```

## Continuation

The CLI automatically continues from where it left off. If you interrupt and restart:

```bash
# First run - processes some samples
uv run -m walter --output results.jsonl --max-items 50
# <Ctrl+C or crash>

# Resume - skips already processed samples
uv run -m walter --output results.jsonl
```

Already-processed sample IDs are read from the existing output file and skipped.

## Examples

### Process specific subset

```bash
# First 100 samples with stereo and CQT features
uv run -m walter \
  --max-items 100 \
  --features stereo,cqt \
  --output results_with_features.jsonl
```

### Use different model

```bash
uv run -m walter --model gemini-3-pro-preview
```

### Custom data location

```bash
uv run -m walter \
  --meta /path/to/MMAR-meta.json \
  --data-dir /path/to/data
```
