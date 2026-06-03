"""
Embedding service using BGE-M3 via FlagEmbedding (primary) or sentence-transformers (fallback).
Supports CUDA auto-detection and CPU fallback.
"""
from typing import List, Optional
import numpy as np

from .utils import logger
import config


class EmbeddingService:
    """Local embedding service using BAAI/bge-m3."""

    def __init__(self):
        self.model = None
        self.model_name = config.EMBEDDING_MODEL_NAME
        self.device = config.EMBEDDING_DEVICE
        self.use_fp16 = config.EMBEDDING_USE_FP16
        self.dimension = 1024  # bge-m3 dense embedding dimension
        self._initialized = False
        self._backend = None  # 'flagembedding' or 'sentence-transformers'

    def _init_flagembedding(self) -> bool:
        """Try loading via FlagEmbedding (BGEM3FlagModel)."""
        from FlagEmbedding import BGEM3FlagModel
        self.model = BGEM3FlagModel(
            self.model_name,
            use_fp16=self.use_fp16,
            device=self.device,
        )
        self._backend = "flagembedding"
        return True

    def _init_sentence_transformers(self) -> bool:
        """Try loading via sentence-transformers."""
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
            trust_remote_code=True,
        )
        self._backend = "sentence-transformers"
        return True

    def initialize(self) -> bool:
        """Initialize the embedding model. Returns True on success."""
        if self._initialized:
            return True

        # Try FlagEmbedding first (more reliable with newer transformers)
        try:
            logger.info("Loading embedding model via FlagEmbedding: %s on %s", self.model_name, self.device)
            self._init_flagembedding()
            self._initialized = True
            logger.info("Embedding model loaded successfully (backend=%s, device=%s)", self._backend, self.device)
            return True
        except ImportError:
            logger.warning("FlagEmbedding not installed, trying sentence-transformers...")
        except Exception as e:
            logger.warning("FlagEmbedding failed: %s", e)

        # Fallback to sentence-transformers
        try:
            logger.info("Loading embedding model via sentence-transformers: %s on %s", self.model_name, self.device)
            self._init_sentence_transformers()
            self._initialized = True
            logger.info("Embedding model loaded successfully (backend=%s, device=%s)", self._backend, self.device)
            return True
        except ImportError:
            logger.error("Neither FlagEmbedding nor sentence-transformers is installed")
            return False
        except Exception as e:
            logger.error("sentence-transformers also failed: %s", e)

        # CPU fallback
        if self.device == "cuda":
            logger.info("Attempting CPU fallback...")
            old_device = self.device
            old_fp16 = self.use_fp16
            self.device = "cpu"
            self.use_fp16 = False
            try:
                self._init_flagembedding()
                self._initialized = True
                logger.info("Embedding model loaded on CPU (fallback)")
                return True
            except Exception:
                try:
                    self._init_sentence_transformers()
                    self._initialized = True
                    logger.info("Embedding model loaded on CPU (fallback, ST backend)")
                    return True
                except Exception as e2:
                    logger.error("CPU fallback also failed: %s", e2)
            self.device = old_device
            self.use_fp16 = old_fp16

        return False

    def encode(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Encode a list of texts into dense embeddings."""
        if not self._initialized:
            if not self.initialize():
                return None

        try:
            if self._backend == "flagembedding":
                output = self.model.encode(
                    texts,
                    batch_size=16,
                    max_length=8192,
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                )
                embeddings = output["dense_vecs"]
            else:
                # sentence-transformers
                embeddings = self.model.encode(
                    texts,
                    batch_size=16,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )

            if isinstance(embeddings, np.ndarray):
                return embeddings.tolist()
            return embeddings
        except Exception as e:
            logger.error("Embedding encoding failed: %s", e)
            return None

    def encode_single(self, text: str) -> Optional[List[float]]:
        """Encode a single text string."""
        result = self.encode([text])
        if result and len(result) > 0:
            return result[0]
        return None

    def get_status(self) -> dict:
        """Return current status of the embedding service."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "use_fp16": self.use_fp16,
            "initialized": self._initialized,
            "backend": self._backend,
            "cuda_available": config.CUDA_AVAILABLE,
            "dimension": self.dimension,
        }
