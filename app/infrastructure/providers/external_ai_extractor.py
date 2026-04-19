"""Adapter for extracting structured information from AI API responses."""

import json
import re
from typing import Any

import httpx
from pydantic import ValidationError

from core.logging import get_logger
from domain.exceptions.ai_extractor import (
    AINetworkError,
    AIProviderError,
    AIResponseValidationError,
)
from domain.ports.ai_extractor import (
    AIExtractedInfo,
    AIExtractor,
    AIResponse,
    ChatMessage,
)

logger = get_logger(__name__)


class ExternalAIExtractor(AIExtractor):
    """External adapter for AI-based information extraction."""

    _MAX_RETRIES = 3
    _TIMEOUT = 10.0

    def __init__(self, api_url: str, api_key: str) -> None:
        """Initialize the AI extractor with API endpoint and credentials.

        Args:
            api_url: Base URL of the AI provider.
            api_key: API key for authentication.
        """
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key

    async def request_extraction(self, messages: list[ChatMessage]) -> AIResponse:
        """Call the AI provider and return its typed response.

        Args:
            messages: List of messages to send to the AI provider.

        Returns:
            AIResponse: The structured response from the provider.

        Raises:
            AINetworkError: If there is a network error connecting to the provider.
            AIProviderError: If the provider returns an error response (HTTP non-200).
            AIResponseValidationError: If response JSON cannot be parsed.
        """
        headers = {
            "X-API-Key": self._api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "messages": [m.model_dump() for m in messages],
        }

        logger.debug(
            f"AI extraction | action=request_start | messages_count={len(messages)}"
        )

        try:
            async with httpx.AsyncClient(timeout=self._TIMEOUT) as client:
                response = await client.post(
                    f"{self._api_url}/v1/ai/extract",
                    json=payload,
                    headers=headers,
                )
        # Network/timeout errors - not retryable at this level
        except httpx.TimeoutException as error:
            logger.warning(
                f"AI extraction | action=request_failed | reason=timeout | error={error!s}"
            )
            raise AINetworkError("AI provider timeout") from error
        except httpx.HTTPError as error:
            logger.warning(
                f"AI extraction | action=request_failed | reason=connection_error | error={error!s}"
            )
            raise AINetworkError("AI provider connection error") from error

        # Success response
        if response.status_code == 200:
            try:
                data = response.json()
                ai_response = AIResponse.model_validate(data)
                logger.debug(
                    f"AI extraction | action=request_success | status_code=200 | choices_count={len(ai_response.choices)}"
                )
                return ai_response
            except (ValueError, KeyError, Exception) as error:
                logger.warning(
                    f"AI extraction | action=request_success | reason=response_parse_failed | error={error!s}"
                )
                raise AIResponseValidationError(
                    "Invalid JSON response from AI provider"
                ) from error

        # Any non-200 response is a provider error
        logger.warning(
            f"AI extraction | action=request_failed | status_code={response.status_code}"
        )
        raise AIProviderError(f"AI provider returned status {response.status_code}")

    async def extract(self, user_input: str) -> AIExtractedInfo:
        """Extract structured info from natural language input with retries.

        This method attempts to extract structured notification data from free-form
        user input. It handles validation failures with retry logic, optionally adding
        refinement prompts to guide the AI toward correct output format.

        Args:
            user_input: Natural language input describing the notification.

        Returns:
            AIExtractedInfo: Validated structured extraction data.

        Raises:
            AIResponseValidationError: If all extraction attempts fail.
            AIProviderError: If provider returns an error.
            AINetworkError: If network connectivity fails.
        """
        logger.info(
            f"AI extraction | action=extract_start | user_input_len={len(user_input)}"
        )

        system_prompt = self._build_system_prompt()
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_input),
        ]

        last_error: Exception | None = None
        extracted_text_for_retry: str = ""

        for attempt in range(self._MAX_RETRIES):
            # Request extraction - provider errors are not retryable
            try:
                ai_response = await self.request_extraction(messages)
            except AIProviderError:
                logger.error(
                    f"AI extraction | action=extract_failed | attempt={attempt + 1} | reason=provider_error"
                )
                raise  # Don't retry provider errors
            except AINetworkError:
                logger.error(
                    f"AI extraction | action=extract_failed | attempt={attempt + 1} | reason=network_error"
                )
                raise  # Don't retry network errors

            # Extraction and parsing - errors here are retryable (content validation)
            try:
                extracted_text = self._extract_message_content(ai_response)
                extracted_text_for_retry = extracted_text
                cleaned_text = self._clean_response(extracted_text)
                parsed = self._parse_extracted_json(cleaned_text)
                result = AIExtractedInfo.model_validate(parsed)
                logger.info(
                    f"AI extraction | action=extract_success | attempt={attempt + 1} | to={result.to} | type={result.type}"
                )
                return result
            except (AIResponseValidationError, ValidationError) as error:
                last_error = error
                logger.warning(
                    f"AI extraction | action=extract_retry | attempt={attempt + 1} | max_attempts={self._MAX_RETRIES} | error={type(error).__name__}"
                )
                # Add refinement prompt for next attempt if retrying
                if attempt < self._MAX_RETRIES - 1:
                    refinement_prompt = (
                        f"User request: {user_input}\n\n"
                        "Extract information and respond ONLY with this JSON structure:\n"
                        '{"to": "<recipient>", "message": "<content>", "type": "email|sms|push"}'
                    )
                    if extracted_text_for_retry:
                        messages.append(
                            ChatMessage(
                                role="assistant", content=extracted_text_for_retry
                            )
                        )
                    messages.append(ChatMessage(role="user", content=refinement_prompt))

        logger.error(
            f"AI extraction | action=extract_failed | reason=max_attempts_exhausted | attempts={self._MAX_RETRIES}"
        )
        raise AIResponseValidationError(
            f"Failed to extract information after {self._MAX_RETRIES} attempts"
        ) from last_error

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the AI extractor.

        Returns:
            The system prompt string.
        """
        return (
            "You are an expert notification assistant. "
            "Extract structured information from user input to create notifications. "
            "Always respond with valid JSON containing exactly these fields: "
            "'to' (recipient email, phone, or identifier), "
            "'message' (the notification content), "
            "'type' (one of: email, sms, push). "
            "Ensure the JSON is valid and all fields are non-empty strings."
        )

    def _extract_message_content(self, ai_response: AIResponse) -> str:
        """Extract the message content from the AI response.

        Args:
            ai_response: The structured response from the AI provider.

        Returns:
            The message content string.

        Raises:
            AIResponseValidationError: If message content cannot be extracted.
        """
        if not ai_response.choices:
            raise AIResponseValidationError("AI response contains no choices")

        choice = ai_response.choices[0]
        if not choice.message or not choice.message.content:
            raise AIResponseValidationError("AI response contains no message content")

        return choice.message.content

    def _clean_response(self, text: str) -> str:
        """Clean and normalize AI response for JSON parsing.

        Removes markdown formatting (backticks, code blocks) and fixes common
        formatting issues that would prevent JSON parsing.

        Args:
            text: Raw text from AI provider.

        Returns:
            Cleaned text suitable for JSON parsing.
        """
        # Remove markdown code block markers (```json, ```, etc.)
        text = re.sub(r"```(?:json)?\s*\n?", "", text)
        text = re.sub(r"```\s*$", "", text)

        # Remove inline backticks
        text = re.sub(r"`", "", text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def _parse_extracted_json(self, text: str) -> dict[str, Any]:
        """Parse JSON from text, attempting to fix common issues.

        Handles malformed JSON by attempting basic fixes like:
        - Fixing unquoted or incorrectly quoted keys
        - Fixing trailing commas
        - Normalizing quotes

        Args:
            text: Text expected to contain JSON.

        Returns:
            Parsed JSON as a dictionary.

        Raises:
            AIResponseValidationError: If JSON cannot be parsed or fixed.
        """
        # First attempt: direct JSON parse
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                logger.debug("AI extraction | action=json_parse_success | attempt=1")
                return parsed
            raise AIResponseValidationError("AI response must be a JSON object")
        except json.JSONDecodeError:
            pass

        # Second attempt: try to fix common issues
        try:
            # Fix trailing commas before closing braces/brackets
            text = re.sub(r",(\s*[}\]])", r"\1", text)

            # Attempt parse again after basic fixes
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                logger.debug(
                    "AI extraction | action=json_parse_success | attempt=2 | reason=trailing_comma_fixed"
                )
                return parsed
            raise AIResponseValidationError("AI response must be a JSON object")
        except json.JSONDecodeError as error:
            logger.warning(
                f"AI extraction | action=json_parse_failed | error={error!s}"
            )
            raise AIResponseValidationError(
                f"Failed to parse JSON from AI response: {error!s}"
            ) from error
