"""Client for the local Ollama API.

The model name, base URL, and timeout come from the environment. This module
does not call any paid model API.
"""

import requests
from loguru import logger

from app.config import get_settings
from app.exceptions import LLMResponseError, OllamaUnavailableError


def generate_response(prompt: str) -> str:
    """Send a prompt to the configured Ollama model and return its text."""

    settings = get_settings()
    url = f"{settings.ollama_base_url}/api/generate"
    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "keep_alive": "10m",
        "options": {
            "temperature": 0,
            "num_predict": 4096,
        },
    }

    logger.info(
        "Calling Ollama model={} timeout={}s",
        settings.ollama_model,
        settings.ollama_timeout_seconds,
    )

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=settings.ollama_timeout_seconds,
        )
    except requests.ConnectionError as exc:
        logger.error("Ollama connection failed url={}", url)
        raise OllamaUnavailableError(
            "The local Ollama service is unavailable. Start Ollama and try again."
        ) from exc
    except requests.Timeout as exc:
        logger.error("Ollama request timed out model={}", settings.ollama_model)
        raise OllamaUnavailableError(
            "The local model took too long to respond. Try again in a moment."
        ) from exc
    except requests.RequestException as exc:
        logger.error("Ollama request failed: {}", type(exc).__name__)
        raise OllamaUnavailableError(
            "The request to the local Ollama service failed."
        ) from exc

    if response.status_code == 404:
        logger.error("Ollama model was not found model={}", settings.ollama_model)
        raise OllamaUnavailableError(
            f"The model '{settings.ollama_model}' is not available in local Ollama. "
            "Pull it with `ollama pull` and try again."
        )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        logger.error("Ollama returned HTTP {}", response.status_code)
        raise OllamaUnavailableError(
            "The local Ollama service returned an error while generating a response."
        ) from exc

    try:
        body = response.json()
    except ValueError as exc:
        logger.error("Ollama returned a non-JSON response")
        raise LLMResponseError("The local model returned an unreadable response.") from exc

    text = body.get("response")
    if not isinstance(text, str) or not text.strip():
        logger.error("Ollama returned an empty response")
        raise LLMResponseError("The local model returned an empty response.")

    done_reason = body.get("done_reason")
    if done_reason not in (None, "stop"):
        logger.warning("Ollama stopped before completion reason={}", done_reason)
    logger.info("Ollama response received characters={}", len(text))
    return text
