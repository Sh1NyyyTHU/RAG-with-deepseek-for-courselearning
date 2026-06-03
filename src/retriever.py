"""
Independent Retriever class.
Retrieves relevant chunks with similarity info and extensible interface.
"""
from typing import List, Dict, Any, Optional

from .embedding_service import EmbeddingService
from .vector_store import VectorStore
from .utils import logger


class Retriever:
    """Retrieve relevant chunks from the vector store.

    Extensible: can add reranker later via a post_process hook.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        top_k: int = 8,
        similarity_threshold: float = 0.0,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        # Extensibility: post-processing pipeline
        self.post_processors: List[callable] = []

    def retrieve(
        self,
        query: str,
        file_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a query.

        Args:
            query: User query string.
            file_filter: Optional list of file names to restrict search.

        Returns:
            List of result dicts with keys: id, text, file_name, page_number,
            chunk_index, similarity, distance.
        """
        # Encode query
        query_embedding = self.embedding_service.encode_single(query)
        if query_embedding is None:
            logger.error("Failed to encode query")
            return []

        # Search vector store
        results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=self.top_k,
            file_filter=file_filter,
        )

        # Apply similarity threshold
        results = [r for r in results if r["similarity"] >= self.similarity_threshold]

        # Apply post-processors (placeholder for reranker)
        for processor in self.post_processors:
            try:
                results = processor(results)
            except Exception as e:
                logger.error("Post-processor failed: %s", e)

        logger.info("Retrieved %d results for query: %.50s...", len(results), query)
        return results

    def add_post_processor(self, processor: callable):
        """Register a post-processing function (e.g., reranker)."""
        self.post_processors.append(processor)
