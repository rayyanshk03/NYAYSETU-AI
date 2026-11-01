"""Main FastAPI application entrypoint for NyayaSetu AI."""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import api_router
from app.core.config import settings


def create_application() -> FastAPI:
    """Factory creating and configuring the FastAPI app."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Backend API for NyayaSetu AI — Multilingual Family Law Information and Decision-Support System",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Configure CORS - restricted strictly to allowed frontend origins (e.g. Vite dev server)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "Accept"],
    )

    # Include routes under the configured prefix (e.g. /api)
    app.include_router(api_router, prefix=settings.API_PREFIX)

    @app.get("/", tags=["Root"])
    async def root():
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "documentation": "/docs",
            "health": f"{settings.API_PREFIX}/health",
        }

    return app


app = create_application()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
