"""Lazy embedding providers with deterministic test support."""

import asyncio
import hashlib
import math
import re
from itertools import pairwise
from typing import Any, Protocol

from backend.app.core.config import Settings


class EmbeddingProvider(Protocol):
    """Batch embedding interface used by ingestion and retrieval."""

    dimension: int

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed text values in input order."""
        ...


class DeterministicEmbeddingProvider:
    """Offline lexical vectors for tests and local development."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate repeatable normalized vectors without external services."""
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        words = re.findall(r"[\w'-]+", text.casefold(), flags=re.UNICODE)
        features = words + [f"{left}::{right}" for left, right in pairwise(words)]
        vector = [0.0] * self.dimension
        for feature in features:
            digest = hashlib.sha256(feature.encode()).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimension
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1
        return [value / norm for value in vector]


class OpenAIEmbeddingProvider:
    """OpenAI embeddings with SDK retries plus an application timeout."""

    def __init__(self, api_key: str, model: str, dimension: int) -> None:
        from openai import AsyncOpenAI

        self.dimension = dimension
        self.model = model
        self.client: Any = AsyncOpenAI(api_key=api_key, max_retries=3, timeout=30)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed one batch and validate the provider's vector dimension."""
        async with asyncio.timeout(45):
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self.dimension,
            )
        vectors = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
        validate_embeddings(vectors, len(texts), self.dimension)
        return vectors

class OllamaEmbeddingProvider:
    """Generate embeddings using Ollama's OpenAI-compatible API."""

    def __init__(
        self,
        model: str,
        dimension: int,
        base_url: str,
    ) -> None:
        from openai import AsyncOpenAI

        self.dimension = dimension
        self.model = model

        self.client: Any = AsyncOpenAI(
            api_key="ollama",  # required by SDK, ignored by Ollama
            base_url=base_url,
            max_retries=3,
            timeout=120,
        )

    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Embed a batch using local Ollama."""

        async with asyncio.timeout(180):
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self.dimension,
            )

        vectors = [
            item.embedding
            for item in sorted(
                response.data,
                key=lambda item: item.index,
            )
        ]

        validate_embeddings(
            vectors,
            len(texts),
            self.dimension,
        )

        return vectors

def validate_embeddings(vectors: list[list[float]], expected_count: int, dimension: int) -> None:
    """Reject missing, non-finite, or incorrectly sized vectors before database writes."""
    if len(vectors) != expected_count:
        raise ValueError("Embedding provider returned the wrong number of vectors")
    if any(len(vector) != dimension for vector in vectors):
        raise ValueError("Embedding provider returned an unexpected dimension")
    if any(not math.isfinite(value) for vector in vectors for value in vector):
        raise ValueError("Embedding provider returned a non-finite value")


def create_embedding_provider(
    settings: Settings,
) -> EmbeddingProvider:
    """Instantiate the configured provider only when needed."""

    if settings.embedding_provider == "fake":
        return DeterministicEmbeddingProvider(
            settings.embedding_dimension
        )

    if settings.embedding_provider == "openai":
        if (
            settings.embedding_api_key is None
            or not settings.embedding_model
        ):
            raise ValueError(
                "OpenAI embedding model and API key are required"
            )

        return OpenAIEmbeddingProvider(
            settings.embedding_api_key.get_secret_value(),
            settings.embedding_model,
            settings.embedding_dimension,
        )

    if settings.embedding_provider == "ollama":
        if not settings.embedding_model:
            raise ValueError(
                "Ollama embedding model is required"
            )

        return OllamaEmbeddingProvider(
            model=settings.embedding_model,
            dimension=settings.embedding_dimension,
            base_url=(
                settings.embedding_base_url
                or "http://host.docker.internal:11434/v1"
            ),
        )

    raise ValueError(
        f"Unsupported embedding provider: "
        f"{settings.embedding_provider}"
    )
