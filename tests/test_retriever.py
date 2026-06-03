"""Tests for retriever."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock, patch

from src.retriever import Retriever


def make_mock_embedding_service():
    """Create a mock embedding service."""
    svc = MagicMock()
    svc.encode_single.return_value = [0.1] * 1024
    return svc


def make_mock_vector_store():
    """Create a mock vector store."""
    store = MagicMock()
    store.query.return_value = [
        {
            "id": f"id_{i}",
            "text": f"Mock chunk {i}",
            "file_name": "test.pdf",
            "page_number": i + 1,
            "chunk_index": i,
            "file_hash": "hash123",
            "similarity": 0.9 - i * 0.1,
            "distance": 0.2 + i * 0.2,
        }
        for i in range(5)
    ]
    return store


def test_retrieve_basic():
    """Test basic retrieval flow."""
    emb_svc = make_mock_embedding_service()
    vec_store = make_mock_vector_store()
    retriever = Retriever(emb_svc, vec_store, top_k=5)

    results = retriever.retrieve("What is gas discharge?")

    assert len(results) == 5
    assert results[0]["file_name"] == "test.pdf"
    assert results[0]["similarity"] > 0
    emb_svc.encode_single.assert_called_once()
    vec_store.query.assert_called_once()


def test_retrieve_with_file_filter():
    """Test retrieval with file filter."""
    emb_svc = make_mock_embedding_service()
    vec_store = make_mock_vector_store()
    retriever = Retriever(emb_svc, vec_store, top_k=3)

    results = retriever.retrieve(
        "query",
        file_filter=["lecture1.pdf", "lecture2.pdf"],
    )

    vec_store.query.assert_called_once()
    call_kwargs = vec_store.query.call_args.kwargs
    assert call_kwargs["file_filter"] == ["lecture1.pdf", "lecture2.pdf"]


def test_retrieve_embedding_failure():
    """Test graceful handling of embedding failure."""
    emb_svc = make_mock_embedding_service()
    emb_svc.encode_single.return_value = None
    vec_store = make_mock_vector_store()

    retriever = Retriever(emb_svc, vec_store)

    results = retriever.retrieve("query")
    assert results == []
    vec_store.query.assert_not_called()


def test_similarity_threshold():
    """Test that low-similarity results are filtered."""
    emb_svc = make_mock_embedding_service()
    vec_store = make_mock_vector_store()
    retriever = Retriever(emb_svc, vec_store, top_k=5, similarity_threshold=0.8)

    results = retriever.retrieve("query")
    # Only the first two results have similarity >= 0.8
    assert len(results) == 2


def test_post_processor():
    """Test that post-processors are called."""
    emb_svc = make_mock_embedding_service()
    vec_store = make_mock_vector_store()
    retriever = Retriever(emb_svc, vec_store)

    processed = []

    def my_processor(results):
        processed.append(len(results))
        return results

    retriever.add_post_processor(my_processor)
    retriever.retrieve("query")

    assert len(processed) == 1
    assert processed[0] == 5
