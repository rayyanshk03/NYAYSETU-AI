"""Pretrained multilingual embedding service for NyayaSetu AI.

Supports English, Hindi, and Kannada using a pretrained SentenceTransformer model:
- Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- Source: Hugging Face (https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- License: Apache 2.0
- Model Size: ~471 MB
- Dimensions: 384
- Multilingual Support: 50+ languages, with explicit teacher-student distillation for Indic languages (hi, kn, etc.)
- Normalization: L2 cosine normalized vectors (unit sphere) for cosine similarity.
"""

from __future__ import annotations

import logging
from typing import List, Optional
import numpy as np
from app.core.config import settings

logger = logging.getLogger(__name__)


class MultilingualEmbedder:
    """Singleton wrapper for loading and executing multilingual embeddings."""

    _instance: Optional[MultilingualEmbedder] = None
    _model = None

    def __init__(self, model_name: Optional[str] = None, use_mock: bool = False):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.use_mock = use_mock
        self._dim = 384

    @classmethod
    def get_instance(cls, model_name: Optional[str] = None, use_mock: bool = False) -> MultilingualEmbedder:
        """Retrieve or create the singleton embedder instance."""
        if cls._instance is None:
            cls._instance = cls(model_name=model_name, use_mock=use_mock)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (primarily for isolated test fixtures)."""
        cls._instance = None
        cls._model = None

    def _load_model(self):
        """Lazy load the underlying SentenceTransformer model."""
        if self.use_mock:
            return None

        if MultilingualEmbedder._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading pretrained multilingual embedding model: %s", self.model_name)
                MultilingualEmbedder._model = SentenceTransformer(self.model_name)
                logger.info("Successfully loaded embedding model: %s", self.model_name)
            except Exception as e:
                logger.warning(
                    "Could not load SentenceTransformer model '%s': %s. Falling back to deterministic mock embeddings.",
                    self.model_name,
                    e,
                )
                self.use_mock = True
        return MultilingualEmbedder._model

    def encode_text(self, text: str) -> List[float]:
        """
        Encode a single query or legal text into a normalized 384-dimensional vector.
        """
        if not text or not text.strip():
            return [0.0] * self._dim

        if self.use_mock:
            return self._mock_encode(text)

        model = self._load_model()
        if self.use_mock or model is None:
            return self._mock_encode(text)

        try:
            vector = model.encode(text.strip(), normalize_embeddings=True)
            return vector.tolist() if hasattr(vector, "tolist") else list(vector)
        except Exception as e:
            logger.error("Error during text embedding generation: %s", e)
            return self._mock_encode(text)

    def encode_passages(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Encode a batch of legal passages into normalized vectors.
        """
        if not texts:
            return []

        if self.use_mock:
            return [self._mock_encode(t) for t in texts]

        model = self._load_model()
        if self.use_mock or model is None:
            return [self._mock_encode(t) for t in texts]

        try:
            cleaned = [t.strip() if t and t.strip() else "empty" for t in texts]
            embeddings = model.encode(
                cleaned,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return embeddings.tolist() if hasattr(embeddings, "tolist") else [list(e) for e in embeddings]
        except Exception as e:
            logger.error("Error during batch embedding generation: %s", e)
            return [self._mock_encode(t) for t in texts]

    def _mock_encode(self, text: str) -> List[float]:
        """
        Deterministic pseudo-embedding for testing or offline environments.
        Computes a normalized vector based on character hash values.
        """
        import hashlib

        # Seed pseudo-random generator with text hash
        text_hash = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(text_hash[:4], "big")
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self._dim)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


# Convenience accessor
embedder = MultilingualEmbedder.get_instance()
