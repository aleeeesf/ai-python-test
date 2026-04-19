"""Unit tests for ExternalAIExtractor."""

import httpx
import pytest
import respx
from domain.exceptions.ai_extractor import (
    AINetworkError,
    AIProviderError,
    AIResponseValidationError,
)
from domain.ports.ai_extractor import ChatMessage
from infrastructure.providers.external_ai_extractor import ExternalAIExtractor


class TestExternalAIExtractorRequestExtraction:
    """Test suite for request_extraction method."""

    @pytest.mark.asyncio
    async def test_returns_ai_response_when_provider_succeeds(self):
        """request_extraction returns parsed response on HTTP 200."""
        # Arrange
        async with respx.mock:
            respx.post("http://provider:3001/v1/ai/extract").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "id": "response-1",
                        "object": "text_completion",
                        "created": 1234567890,
                        "model": "gpt-4",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": '{"to": "user@example.com", "message": "Hello", "type": "email"}',
                                },
                                "finish_reason": "stop",
                            }
                        ],
                    },
                )
            )
            extractor = ExternalAIExtractor(
                api_url="http://provider:3001",
                api_key="test-dev-2026",
            )

            # Act
            messages = [
                ChatMessage(role="user", content="Send email to user@example.com")
            ]
            result = await extractor.request_extraction(messages)

            # Assert
            assert result.id == "response-1"
            assert result.model == "gpt-4"
            assert len(result.choices) == 1
            assert (
                result.choices[0].message.content
                == '{"to": "user@example.com", "message": "Hello", "type": "email"}'
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status_code",
        [401, 429, 500, 422, 418],
    )
    async def test_raises_provider_error_for_non_200_status(self, status_code):
        """request_extraction raises AIProviderError for any non-200 response."""
        # Arrange
        async with respx.mock:
            respx.post("http://provider:3001/v1/ai/extract").mock(
                return_value=httpx.Response(status_code, json={"error": "Test error"})
            )
            extractor = ExternalAIExtractor(
                api_url="http://provider:3001",
                api_key="test-dev-2026",
            )

            # Act / Assert
            with pytest.raises(AIProviderError):
                messages = [ChatMessage(role="user", content="Test")]
                await extractor.request_extraction(messages)

    @pytest.mark.asyncio
    async def test_raises_network_error_on_timeout(self):
        """request_extraction raises AINetworkError on timeout."""
        # Arrange
        async with respx.mock:
            respx.post("http://provider:3001/v1/ai/extract").mock(
                side_effect=httpx.TimeoutException("timeout")
            )
            extractor = ExternalAIExtractor(
                api_url="http://provider:3001",
                api_key="test-dev-2026",
            )

            # Act / Assert
            with pytest.raises(AINetworkError):
                messages = [ChatMessage(role="user", content="Test")]
                await extractor.request_extraction(messages)

    @pytest.mark.asyncio
    async def test_raises_validation_error_on_invalid_response_json(self):
        """request_extraction raises AIResponseValidationError on malformed JSON."""
        # Arrange
        async with respx.mock:
            respx.post("http://provider:3001/v1/ai/extract").mock(
                return_value=httpx.Response(200, text="not json")
            )
            extractor = ExternalAIExtractor(
                api_url="http://provider:3001",
                api_key="test-dev-2026",
            )

            # Act / Assert
            with pytest.raises(AIResponseValidationError):
                messages = [ChatMessage(role="user", content="Test")]
                await extractor.request_extraction(messages)

    @pytest.mark.asyncio
    async def test_raises_validation_error_on_missing_required_fields(self):
        """request_extraction validates response schema."""
        # Arrange
        async with respx.mock:
            respx.post("http://provider:3001/v1/ai/extract").mock(
                return_value=httpx.Response(200, json={"id": "resp-1"})
            )
            extractor = ExternalAIExtractor(
                api_url="http://provider:3001",
                api_key="test-dev-2026",
            )

            # Act / Assert
            with pytest.raises(AIResponseValidationError):
                messages = [ChatMessage(role="user", content="Test")]
                await extractor.request_extraction(messages)


class TestExternalAIExtractorExtract:
    """Test suite for extract method."""

    @pytest.mark.asyncio
    async def test_extracts_info_on_valid_response(self):
        """extract returns AIExtractedInfo on successful AI response."""
        # Arrange
        async with respx.mock:
            respx.post("http://provider:3001/v1/ai/extract").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "id": "response-1",
                        "object": "text_completion",
                        "created": 1234567890,
                        "model": "gpt-4",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": '{"to": "user@example.com", "message": "Hello world", "type": "email"}',
                                },
                                "finish_reason": "stop",
                            }
                        ],
                    },
                )
            )
            extractor = ExternalAIExtractor(
                api_url="http://provider:3001",
                api_key="test-dev-2026",
            )

            # Act
            result = await extractor.extract(
                "Send email to user@example.com saying hello"
            )

            # Assert
            assert result.to == "user@example.com"
            assert result.message == "Hello world"
            assert result.type == "email"

    @pytest.mark.asyncio
    async def test_handles_markdown_formatted_json(self):
        """extract removes markdown formatting from JSON response."""
        # Arrange
        async with respx.mock:
            respx.post("http://provider:3001/v1/ai/extract").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "id": "response-1",
                        "object": "text_completion",
                        "created": 1234567890,
                        "model": "gpt-4",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": "```json\n"
                                    '{"to": "user@example.com", "message": "Test", "type": "sms"}\n'
                                    "```",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                    },
                )
            )
            extractor = ExternalAIExtractor(
                api_url="http://provider:3001",
                api_key="test-dev-2026",
            )

            # Act
            result = await extractor.extract("Send SMS")

            # Assert
            assert result.to == "user@example.com"
            assert result.message == "Test"
            assert result.type == "sms"

    @pytest.mark.asyncio
    async def test_handles_trailing_commas_in_json(self):
        """extract fixes trailing commas in JSON."""
        # Arrange
        async with respx.mock:
            respx.post("http://provider:3001/v1/ai/extract").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "id": "response-1",
                        "object": "text_completion",
                        "created": 1234567890,
                        "model": "gpt-4",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": '{"to": "user@example.com", "message": "Test", "type": "email",}',
                                },
                                "finish_reason": "stop",
                            }
                        ],
                    },
                )
            )
            extractor = ExternalAIExtractor(
                api_url="http://provider:3001",
                api_key="test-dev-2026",
            )

            # Act
            result = await extractor.extract("Send email")

            # Assert
            assert result.to == "user@example.com"

    @pytest.mark.asyncio
    async def test_handles_alias_choices_in_fields(self):
        """extract accepts alternative field names via AliasChoices."""
        # Arrange
        async with respx.mock:
            respx.post("http://provider:3001/v1/ai/extract").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "id": "response-1",
                        "object": "text_completion",
                        "created": 1234567890,
                        "model": "gpt-4",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": '{"recipient": "user@example.com", "body": "Hello", "channel": "email"}',
                                },
                                "finish_reason": "stop",
                            }
                        ],
                    },
                )
            )
            extractor = ExternalAIExtractor(
                api_url="http://provider:3001",
                api_key="test-dev-2026",
            )

            # Act
            result = await extractor.extract("Send email")

            # Assert
            assert result.to == "user@example.com"
            assert result.message == "Hello"
            assert result.type == "email"

    @pytest.mark.asyncio
    async def test_retries_with_refinement_prompt_on_validation_failure(self):
        """extract retries with refinement prompt when extraction fails."""
        # Arrange
        attempt_count = 0

        def response_callback(request: httpx.Request) -> httpx.Response:
            nonlocal attempt_count
            attempt_count += 1

            if attempt_count == 1:
                # First attempt: invalid JSON (triggers retry with refinement)
                return httpx.Response(
                    200,
                    json={
                        "id": "response-1",
                        "object": "text_completion",
                        "created": 1234567890,
                        "model": "gpt-4",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": "Invalid response",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                    },
                )
            else:
                # Second attempt: valid response
                return httpx.Response(
                    200,
                    json={
                        "id": "response-2",
                        "object": "text_completion",
                        "created": 1234567890,
                        "model": "gpt-4",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": '{"to": "user@example.com", "message": "Hello", "type": "email"}',
                                },
                                "finish_reason": "stop",
                            }
                        ],
                    },
                )

        async with respx.mock:
            respx.post("http://provider:3001/v1/ai/extract").mock(
                side_effect=response_callback
            )
            extractor = ExternalAIExtractor(
                api_url="http://provider:3001",
                api_key="test-dev-2026",
            )

            # Act
            result = await extractor.extract("Send email")

            # Assert
            assert result.to == "user@example.com"
            assert attempt_count == 2  # Verify retry happened

    @pytest.mark.asyncio
    async def test_fails_after_max_retries(self):
        """extract raises error after exhausting all retries."""
        # Arrange
        async with respx.mock:
            respx.post("http://provider:3001/v1/ai/extract").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "id": "response-1",
                        "object": "text_completion",
                        "created": 1234567890,
                        "model": "gpt-4",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": "Always invalid",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                    },
                )
            )
            extractor = ExternalAIExtractor(
                api_url="http://provider:3001",
                api_key="test-dev-2026",
            )

            # Act / Assert
            with pytest.raises(AIResponseValidationError) as exc_info:
                await extractor.extract("Send email")

            assert "Failed to extract information after 3 attempts" in str(
                exc_info.value
            )

    @pytest.mark.asyncio
    async def test_propagates_provider_error_without_retry(self):
        """extract raises AIProviderError immediately for provider errors."""
        # Arrange
        attempt_count = 0

        def response_callback(request: httpx.Request) -> httpx.Response:
            nonlocal attempt_count
            attempt_count += 1
            return httpx.Response(422, json={"error": "Invalid request"})

        async with respx.mock:
            respx.post("http://provider:3001/v1/ai/extract").mock(
                side_effect=response_callback
            )
            extractor = ExternalAIExtractor(
                api_url="http://provider:3001",
                api_key="test-dev-2026",
            )

            # Act / Assert
            with pytest.raises(AIProviderError):
                await extractor.extract("Send email")

            assert attempt_count == 1  # Should not retry

    @pytest.mark.asyncio
    async def test_propagates_provider_error_on_http_error_without_retry(self):
        """extract raises AIProviderError immediately for HTTP errors."""
        # Arrange
        attempt_count = 0

        def response_callback(request: httpx.Request) -> httpx.Response:
            nonlocal attempt_count
            attempt_count += 1
            return httpx.Response(500, json={"error": "Server error"})

        async with respx.mock:
            respx.post("http://provider:3001/v1/ai/extract").mock(
                side_effect=response_callback
            )
            extractor = ExternalAIExtractor(
                api_url="http://provider:3001",
                api_key="test-dev-2026",
            )

            # Act / Assert
            with pytest.raises(AIProviderError):
                await extractor.extract("Send email")

            assert attempt_count == 1  # Should not retry

    @pytest.mark.asyncio
    async def test_propagates_network_error_without_retry(self):
        """extract raises AINetworkError immediately without retry."""
        # Arrange
        attempt_count = 0

        def response_callback(request: httpx.Request) -> httpx.Response:
            nonlocal attempt_count
            attempt_count += 1
            raise httpx.NetworkError("Connection refused")

        async with respx.mock:
            respx.post("http://provider:3001/v1/ai/extract").mock(
                side_effect=response_callback
            )
            extractor = ExternalAIExtractor(
                api_url="http://provider:3001",
                api_key="test-dev-2026",
            )

            # Act / Assert
            with pytest.raises(AINetworkError):
                await extractor.extract("Send email")

            assert attempt_count == 1  # Should not retry


class TestExternalAIExtractorHelpers:
    """Test suite for internal helper methods."""

    def test_clean_response_removes_markdown_code_blocks(self):
        """_clean_response removes markdown code block markers."""
        # Arrange
        extractor = ExternalAIExtractor(
            api_url="http://provider:3001",
            api_key="test-dev-2026",
        )

        # Act
        result = extractor._clean_response(
            "```json\n"
            '{"to": "user@example.com", "message": "Hello", "type": "email"}\n'
            "```"
        )

        # Assert
        assert (
            result == '{"to": "user@example.com", "message": "Hello", "type": "email"}'
        )

    def test_clean_response_removes_inline_backticks(self):
        """_clean_response removes inline backticks."""
        # Arrange
        extractor = ExternalAIExtractor(
            api_url="http://provider:3001",
            api_key="test-dev-2026",
        )

        # Act
        result = extractor._clean_response("`test` `json`")

        # Assert
        assert result == "test json"

    def test_clean_response_strips_whitespace(self):
        """_clean_response strips leading/trailing whitespace."""
        # Arrange
        extractor = ExternalAIExtractor(
            api_url="http://provider:3001",
            api_key="test-dev-2026",
        )

        # Act
        result = extractor._clean_response("   \n  test  \n   ")

        # Assert
        assert result == "test"

    def test_parse_extracted_json_parses_valid_json(self):
        """_parse_extracted_json parses valid JSON."""
        # Arrange
        extractor = ExternalAIExtractor(
            api_url="http://provider:3001",
            api_key="test-dev-2026",
        )

        # Act
        result = extractor._parse_extracted_json('{"key": "value", "number": 42}')

        # Assert
        assert result == {"key": "value", "number": 42}

    def test_parse_extracted_json_fixes_trailing_comma(self):
        """_parse_extracted_json fixes trailing commas."""
        # Arrange
        extractor = ExternalAIExtractor(
            api_url="http://provider:3001",
            api_key="test-dev-2026",
        )

        # Act
        result = extractor._parse_extracted_json('{"key": "value",}')

        # Assert
        assert result == {"key": "value"}

    def test_parse_extracted_json_raises_on_invalid_json(self):
        """_parse_extracted_json raises AIResponseValidationError on unfixable JSON."""
        # Arrange
        extractor = ExternalAIExtractor(
            api_url="http://provider:3001",
            api_key="test-dev-2026",
        )

        # Act / Assert
        with pytest.raises(AIResponseValidationError):
            extractor._parse_extracted_json("not valid json at all {")
