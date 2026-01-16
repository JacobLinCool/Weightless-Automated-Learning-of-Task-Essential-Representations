"""Gemini API client for audio question answering."""

from __future__ import annotations

import json

from dotenv import load_dotenv
from google import genai
from google.genai.types import Content, GenerateContentConfig, Part, ThinkingConfig
from pydantic import BaseModel, Field

from ..mmar import MMARSample

load_dotenv()


client = genai.Client()


class PredictionResponse(BaseModel):
    """Structured response from the prediction model."""

    thinking: str = Field(description="The model's chain-of-thought reasoning process")
    answer: str = Field(
        description="The final answer choice (e.g., 'A', 'B', 'C', 'D' or the actual answer text)"
    )


def _build_content_and_config(
    sample: MMARSample,
    audio_bytes: bytes,
    feature_images: list[tuple[str, bytes]] | None = None,
    enable_thinking: bool = False,
) -> tuple[list[Content], GenerateContentConfig]:
    """Build content parts and config for the API call.

    This is shared between sync and async versions.
    """
    # Build the prompt
    choices_text = "\n".join(
        f"{chr(65 + i)}. {choice}" for i, choice in enumerate(sample.choices)
    )

    prompt = f"""You are an expert audio analyst answering multiple-choice questions about audio content.

Question: {sample.question}

Choices:
{choices_text}

Listen carefully to the audio and analyze any provided feature visualizations to determine the correct answer.

Think step by step about what you hear in the audio and what you observe in the visualizations (if any).
Then provide your final answer as just the letter (A, B, C, D, etc.) or the exact text of the correct choice.
"""

    # Build content parts
    parts: list[Part] = []

    # Add audio
    parts.append(
        Part.from_bytes(
            data=audio_bytes,
            mime_type="audio/wav",
        )
    )

    # Add feature images if provided
    if feature_images:
        for feature_name, image_bytes in feature_images:
            parts.append(
                Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/png",
                )
            )
            parts.append(Part.from_text(text=f"[{feature_name} visualization above]"))

    # Add the text prompt
    parts.append(Part.from_text(text=prompt))

    # Create content
    contents = [Content(role="user", parts=parts)]

    # Build config
    if enable_thinking:
        config = GenerateContentConfig(
            thinking_config=ThinkingConfig(include_thoughts=True),
            response_mime_type="application/json",
            response_schema=PredictionResponse,
        )
    else:
        config = GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=PredictionResponse,
        )

    return contents, config


def _parse_response(response) -> dict[str, str]:
    """Parse the API response into thinking and answer."""
    thinking = ""
    answer = ""

    if response.candidates and response.candidates[0].content:
        for part in response.candidates[0].content.parts:
            if hasattr(part, "thought") and part.thought:
                thinking += part.text or ""
            elif hasattr(part, "text") and part.text:
                try:
                    parsed = json.loads(part.text)
                    if isinstance(parsed, dict):
                        thinking = parsed.get("thinking", thinking)
                        answer = parsed.get("answer", "")
                except json.JSONDecodeError:
                    answer = part.text

    return {
        "thinking_prediction": thinking,
        "answer_prediction": answer,
    }


def generate_prediction(
    model: str,
    sample: MMARSample,
    audio_bytes: bytes,
    feature_images: list[tuple[str, bytes]] | None = None,
    enable_thinking: bool = False,
) -> dict[str, str]:
    """Generate a prediction for an audio multiple-choice question (sync version).

    Args:
        model: The Gemini model to use.
        sample: The MMAR sample containing question and choices.
        audio_bytes: Raw audio file bytes.
        feature_images: Optional list of (feature_name, png_bytes) tuples.
        enable_thinking: Whether to enable thinking mode (requires compatible model).

    Returns:
        Dictionary with 'thinking_prediction' and 'answer_prediction' keys.
    """
    contents, config = _build_content_and_config(
        sample, audio_bytes, feature_images, enable_thinking
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )

    return _parse_response(response)


async def generate_prediction_async(
    model: str,
    sample: MMARSample,
    audio_bytes: bytes,
    feature_images: list[tuple[str, bytes]] | None = None,
    enable_thinking: bool = False,
) -> dict[str, str]:
    """Generate a prediction for an audio multiple-choice question (async version).

    Args:
        model: The Gemini model to use.
        sample: The MMAR sample containing question and choices.
        audio_bytes: Raw audio file bytes.
        feature_images: Optional list of (feature_name, png_bytes) tuples.
        enable_thinking: Whether to enable thinking mode (requires compatible model).

    Returns:
        Dictionary with 'thinking_prediction' and 'answer_prediction' keys.
    """
    contents, config = _build_content_and_config(
        sample, audio_bytes, feature_images, enable_thinking
    )

    response = await client.aio.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )

    return _parse_response(response)


if __name__ == "__main__":
    # Test the client
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[Content(role="user", parts=[Part(text="Hello, how are you?")])],
        config=GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=PredictionResponse,
        ),
    )
    print(response)
