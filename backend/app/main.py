from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager

from backend.app.core.logging import configure_logging
from backend1.app.core.config import Settings, get_settings

API_PREFIX = "/api/v1"
def create_app(settings_override: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """ Initialize and cleanup application state for one FastAPI instance. """
        settings = settings_override or get_settings()
        configure_logging(settings.log_level)
        app.state.settings = settings
        # FastAPI serves requests while execution is paused here.
        yield

    application = FastAPI(
            title="Agentic RAG Knowledge Assistant",
            description="Source-grounded document question answering API",
            version="0.1.0",
            docs_url=f"{API_PREFIX}/docs",
            redoc_url=f"{API_PREFIX}/redoc",
            openapi_url=f"{API_PREFIX}/openapi.json",
            lifespan=lifespan,
        )


    @application.get("/")
    def read_root():
        return {"Hello": "CeCe"}

    return application

app = create_app()