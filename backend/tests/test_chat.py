"""Tests for /api/chat endpoint and validation rules."""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_chat_valid_english():
    """Verify chat endpoint returns 200 with agentic response structure (Step 4)."""
    payload = {
        "message": "What is the procedure for mutual consent divorce under the Hindu Marriage Act?",
        "language": "en",
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("success", "error")
    assert data["language"] == "en"
    # Step 4: endpoint now routes through the agentic pipeline
    assert data["is_placeholder"] is False
    assert "mutual consent divorce" in data["received_message"]
    assert data["reply"]  # Non-empty reply
    assert "disclaimer" in data
    assert "retrieved_passages" in data
    assert "agent_steps" in data
    assert isinstance(data["agent_steps"], list)


def test_chat_valid_hindi():
    """Verify chat endpoint works with Hindi language code (Step 4)."""
    payload = {
        "message": "सीआरपीसी धारा 125 के तहत भरण-पोषण के नियम क्या हैं?",
        "language": "hi",
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("success", "error")
    assert data["language"] == "hi"
    # Reply must be non-empty; exact wording depends on corpus state
    assert data["reply"]


def test_chat_valid_kannada():
    """Verify chat endpoint works with Kannada language code (Step 4)."""
    payload = {
        "message": "ಮಕ್ಕಳ ಪಾಲನೆಯ ನಿಯಮಗಳು ಯಾವುವು?",
        "language": "kn",
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("success", "error")
    assert data["language"] == "kn"
    assert data["reply"]


def test_chat_empty_message_validation():
    """Verify empty message triggers 422 validation error."""
    payload = {
        "message": "   ",
        "language": "en",
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 422


def test_chat_missing_message_validation():
    """Verify missing message triggers 422 error."""
    payload = {
        "language": "en",
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 422


def test_chat_invalid_language_validation():
    """Verify unsupported language triggers 422 error."""
    payload = {
        "message": "What is maintenance?",
        "language": "es",  # Spanish is not one of en, hi, kn
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 422


def test_cors_headers_allowed_origin():
    """Verify CORS headers are returned for allowed Vite origin."""
    response = client.options(
        "/api/chat",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
