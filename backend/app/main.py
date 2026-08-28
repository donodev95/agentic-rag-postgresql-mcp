from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager

from backend.app.api import health
from backend.app.core.errors import register_exception_handlers
from backend.app.core.logging import configure_logging
from backend.app.core.config import Settings, get_settings
from backend.app.db.session import Database

API_PREFIX = "/api/v1"
def create_app(settings_override: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """ Initialize and cleanup application state for one FastAPI instance. """
        settings = settings_override or get_settings()
        configure_logging(settings.log_level)
        app.state.settings = settings
        app.state.database = Database(settings.database_url)
        yield
        # try:
        #     async with create_checkpointer(settings) as checkpointer:
        #         app.state.checkpointer = checkpointer
        #         yield
        # finally:
        #     await app.state.database.close()

    application = FastAPI(
            title="Agentic RAG Knowledge Assistant",
            description="Source-grounded document question answering API",
            version="0.1.0",
            docs_url=f"{API_PREFIX}/docs",
            redoc_url=f"{API_PREFIX}/redoc",
            openapi_url=f"{API_PREFIX}/openapi.json",
            lifespan=lifespan,
        )

    register_exception_handlers(application)
    application.include_router(health.router) # Checking the Liveness of the backend application and the Readiness of the database connection.
    # application.include_router(auth.router, prefix=API_PREFIX)
    # application.include_router(users.router, prefix=API_PREFIX)
    # application.include_router(threads.router, prefix=API_PREFIX)
    # application.include_router(documents.router, prefix=API_PREFIX)
    # application.include_router(data_sources.router, prefix=API_PREFIX)
    # application.include_router(retrieval.router, prefix=API_PREFIX)
    # application.include_router(chat.router, prefix=API_PREFIX)
    # application.include_router(metrics.router, prefix=API_PREFIX)

    # @application.get("/")
    # def read_root():
    #     return {"Hello": "CeCe"}

    return application

app = create_app()