# Audio Feature Extractors

This document describes the audio feature extractors available for multimodal reasoning with the MMAR dataset.

## Overview

The feature extractors convert audio signals into visual representations that help AI models reason about specific aspects of audio content. Each feature targets a specific type of question.

## Available Features

### 1. CQT / Chromagram (`cqt`)

**Purpose**: Music theory analysis with pitch-aligned frequency representation.

**What it shows**:
- **CQT (Constant-Q Transform)**: Logarithmic frequency bins aligned to musical notes (7 octaves)
- **Chromagram**: 12 pitch classes (C, C#, D, ..., B) over time

**Best for**:
- "What chord is this?"
- "Is the melody rising?"
- "Is this Major or Minor?"
- Any music theory or harmonic analysis

**Example usage**:
```bash
uv run -m walter --features cqt
```

---

### 2. F0 Contour (`f0`)

**Purpose**: Speaker and emotion analysis through pitch tracking.

**What it shows**:
- **F0 contour**: Fundamental frequency (pitch) over time in Hz
- **Semitone plot**: Relative pitch variation showing intonation patterns

**Best for**:
- "Is this a question?" (Rising tail = question intonation)
- "Are they singing or speaking?"
- "Is the tone monotonous or excited?"
- Any prosody, intonation, or speaker emotion questions

**Example usage**:
```bash
uv run -m walter --features f0
```

---

### 3. Stereo Phase Scope (`stereo`)

**Purpose**: Spatial audio analysis for direction and movement.

**What it shows**:
- **Lissajous (Phase Scope)**: L/R channel relationship scatter plot
- **Panning over time**: Sound position moving between L and R
- **Mid/Side energy**: Center vs stereo content balance

**Best for**:
- "Is the train moving left to right?"
- "Is the sound coming from behind?"
- Any spatial direction or movement questions

> **Note**: Most AI audio models process audio in mono and cannot perceive stereo information. This feature provides spatial data they would otherwise miss.

**Example usage**:
```bash
uv run -m walter --features stereo
```

---

## Combining Features

You can enable multiple features by separating them with commas:

```bash
# Enable CQT and F0 for a music counting question
uv run -m walter --features cqt,f0

# Enable all features
uv run -m walter --features all
```

## Feature Selection Guide

| Question Type | Recommended Features |
|--------------|---------------------|
| Counting events | `cqt` |
| Music/chords/melody | `cqt` |
| Speaker emotion/intonation | `f0` |
| Spatial/direction | `stereo` |
| General audio | (none - audio only) |

## Technical Details

All feature extractors:
- Accept WAV files (mono or stereo)
- Output PNG images at 100 DPI
- Handle errors gracefully (return `None` on failure)
- Are implemented in `walter/features/`
