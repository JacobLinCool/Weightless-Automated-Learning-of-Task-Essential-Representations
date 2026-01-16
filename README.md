# Weightless Automated Learning of Task Essential Representations

This repo prototypes **zero-shot prompt optimization via feature search** over the MMAR dataset metadata.

- Algorithm overview: see `docs/algorithm.md`
- Implementation details (current prototype): see `docs/implementation.md`

## Quickstart (metadata-only)

This machine setup intentionally does **not** load audio bytes. The prototype runs purely on `data/MMAR-meta.json`.

```bash
export GEMINI_API_KEY="..."
python -m walter --meta data/MMAR-meta.json --model gemini-3-flash-preview
```
