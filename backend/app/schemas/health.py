"""Health check response schema."""

from datetime import datetime, timezone
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health status response payload."""

    status: str = Field(..., description="Current service health status", json_schema_extra={"example": "healthy"})
    app_name: str = Field(..., description="Service name")
    version: str = Field(..., description="Semantic version of the application")
    environment: str = Field(..., description="Operational environment")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the health check",
    )
