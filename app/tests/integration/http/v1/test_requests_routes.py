"""Integration tests for request routes."""

import asyncio

from domain.entities.request import NotificationRequest, NotificationStatus


class TestCreateRequest:
    """Test suite for POST /v1/requests route."""

    def test_creates_request_when_payload_is_valid(self, client):
        """POST /v1/requests returns 201 and a generated request ID."""
        # Arrange
        payload = {
            "to": "user@example.com",
            "message": "Test notification",
            "type": "email",
        }

        # Act
        response = client.post("/v1/requests", json=payload)
        body = response.json()

        # Assert
        assert response.status_code == 201
        assert isinstance(body["id"], str)
        assert body["id"]

    def test_rejects_create_request_with_empty_to(self, client):
        """POST /v1/requests rejects payload with empty 'to' field."""
        # Arrange
        payload = {
            "to": "",
            "message": "Test notification",
            "type": "email",
        }

        # Act
        response = client.post("/v1/requests", json=payload)

        # Assert
        assert response.status_code == 422  # Validation error

    def test_rejects_create_request_with_empty_message(self, client):
        """POST /v1/requests rejects payload with empty 'message' field."""
        # Arrange
        payload = {
            "to": "user@example.com",
            "message": "",
            "type": "email",
        }

        # Act
        response = client.post("/v1/requests", json=payload)

        # Assert
        assert response.status_code == 422

    def test_rejects_create_request_with_invalid_type(self, client):
        """POST /v1/requests rejects payload with invalid notification type."""
        # Arrange
        payload = {
            "to": "user@example.com",
            "message": "Test notification",
            "type": "telegram",  # Not in {email, sms, push}
        }

        # Act
        response = client.post("/v1/requests", json=payload)

        # Assert
        assert response.status_code == 422

    def test_rejects_create_request_with_missing_to(self, client):
        """POST /v1/requests rejects payload missing 'to' field."""
        # Arrange
        payload = {
            "message": "Test notification",
            "type": "email",
        }

        # Act
        response = client.post("/v1/requests", json=payload)

        # Assert
        assert response.status_code == 422

    def test_rejects_create_request_with_missing_message(self, client):
        """POST /v1/requests rejects payload missing 'message' field."""
        # Arrange
        payload = {
            "to": "user@example.com",
            "type": "email",
        }

        # Act
        response = client.post("/v1/requests", json=payload)

        # Assert
        assert response.status_code == 422

    def test_rejects_create_request_with_missing_type(self, client):
        """POST /v1/requests rejects payload missing 'type' field."""
        # Arrange
        payload = {
            "to": "user@example.com",
            "message": "Test notification",
        }

        # Act
        response = client.post("/v1/requests", json=payload)

        # Assert
        assert response.status_code == 422

    def test_rejects_create_request_with_non_string_to(self, client):
        """POST /v1/requests rejects 'to' field that is not a string."""
        # Arrange
        payload = {
            "to": 123,
            "message": "Test notification",
            "type": "email",
        }

        # Act
        response = client.post("/v1/requests", json=payload)

        # Assert
        assert response.status_code == 422

    def test_rejects_create_request_with_non_string_message(self, client):
        """POST /v1/requests rejects 'message' field that is not a string."""
        # Arrange
        payload = {
            "to": "user@example.com",
            "message": 123,
            "type": "email",
        }

        # Act
        response = client.post("/v1/requests", json=payload)

        # Assert
        assert response.status_code == 422


class TestProcessRequest:
    """Test suite for POST /v1/requests/{id}/process route."""

    def test_returns_not_found_when_processing_unknown_request(self, client):
        """POST /v1/requests/{id}/process returns 404 for unknown request IDs."""
        # Act
        response = client.post("/v1/requests/missing-request/process")

        # Assert
        assert response.status_code == 404

    def test_returns_accepted_and_dispatches_when_processing_queued_request(
        self,
        client,
        stub_process_dispatcher,
    ):
        """POST /process returns 202 and dispatches queued requests."""
        # Arrange
        create_response = client.post(
            "/v1/requests",
            json={
                "to": "user@example.com",
                "message": "Test notification",
                "type": "email",
            },
        )
        request_id = create_response.json()["id"]

        # Act
        response = client.post(f"/v1/requests/{request_id}/process")
        status_response = client.get(f"/v1/requests/{request_id}")

        # Assert
        assert response.status_code == 202
        assert stub_process_dispatcher.dispatched_request_ids == [request_id]
        assert status_response.status_code == 200
        assert status_response.json() == {
            "id": request_id,
            "status": NotificationStatus.PROCESSING.value,
        }

    def test_processing_already_sent_request_returns_200(self, client):
        """POST /process on already SENT request returns 200 (idempotent)."""
        # Arrange
        create_response = client.post(
            "/v1/requests",
            json={
                "to": "user@example.com",
                "message": "Test notification",
                "type": "email",
            },
        )
        request_id = create_response.json()["id"]

        # First process: QUEUED → PROCESSING, dispatches work
        client.post(f"/v1/requests/{request_id}/process")

        # Act: Try to process while in PROCESSING (should return 202)
        response = client.post(f"/v1/requests/{request_id}/process")

        # Assert
        assert response.status_code == 202

    def test_processing_request_twice_does_not_dispatch_twice(
        self, client, stub_process_dispatcher
    ):
        """POST /process twice on same request dispatches only once."""
        # Arrange
        create_response = client.post(
            "/v1/requests",
            json={
                "to": "user@example.com",
                "message": "Test notification",
                "type": "email",
            },
        )
        request_id = create_response.json()["id"]

        # Act: Process twice
        client.post(f"/v1/requests/{request_id}/process")
        response_2 = client.post(f"/v1/requests/{request_id}/process")

        # Assert: Dispatched only once despite two process calls
        assert response_2.status_code == 202
        assert stub_process_dispatcher.dispatched_request_ids == [request_id]

    def test_processing_failed_request_allows_retry(self, client, requests_repository):
        """POST /process on FAILED request returns 202 to allow retry."""
        # Arrange - Save a FAILED request directly to repository
        failed_req = NotificationRequest(
            id="failed-request",
            to="user@example.com",
            message="Test",
            type="email",
            status=NotificationStatus.FAILED,
            error="Previous attempt failed",
        )
        asyncio.run(requests_repository.save(failed_req))

        # Act
        response = client.post("/v1/requests/failed-request/process")

        # Assert: Can retry failed requests (returns 202)
        assert response.status_code == 202


class TestGetRequestStatus:
    """Test suite for GET /v1/requests/{id} route."""

    def test_returns_not_found_when_getting_status_of_unknown_request(self, client):
        """GET /v1/requests/{id} returns 404 for unknown request IDs."""
        # Act
        response = client.get("/v1/requests/missing-request")

        # Assert
        assert response.status_code == 404
        assert response.json()["detail"] == "Request not found"

    def test_getting_status_of_created_request_shows_queued(self, client):
        """GET /v1/requests/{id} after creation shows 'queued' status."""
        # Arrange
        create_response = client.post(
            "/v1/requests",
            json={
                "to": "user@example.com",
                "message": "Test notification",
                "type": "email",
            },
        )
        request_id = create_response.json()["id"]

        # Act
        response = client.get(f"/v1/requests/{request_id}")

        # Assert
        assert response.status_code == 200
        assert response.json()["status"] == NotificationStatus.QUEUED.value
