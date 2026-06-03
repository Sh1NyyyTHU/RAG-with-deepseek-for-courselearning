"""Tests for vector store."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.vector_store import VectorStore, TextChunk


def make_chunks(file_name="test.pdf", file_hash="abc123", count=3):
    """Helper to create test chunks."""
    chunks = []
    for i in range(count):
        chunks.append(TextChunk(
            document_id=file_hash[:16],
            file_name=file_name,
            file_hash=file_hash,
            page_number=i + 1,
            chunk_index=i,
            text=f"Test chunk {i} content for {file_name} page {i+1}.",
        ))
    return chunks


def make_embeddings(count=3, dim=4):
    """Helper to create dummy embeddings."""
    return [[0.1 * (i + j) for j in range(dim)] for i in range(count)]


def test_add_and_query(tmp_path):
    """Test basic add and query flow."""
    store = VectorStore(persist_dir=tmp_path / "chroma_test")

    chunks = make_chunks()
    embeddings = make_embeddings(len(chunks), dim=1024)

    store.add_chunks(chunks, embeddings)

    # Query
    query_emb = [0.1] * 1024
    results = store.query(query_emb, top_k=2)

    assert len(results) > 0
    assert len(results) <= 2
    for r in results:
        assert "file_name" in r
        assert "page_number" in r
        assert "text" in r
        assert "similarity" in r


def test_delete_by_hash(tmp_path):
    """Test deleting chunks by file hash."""
    store = VectorStore(persist_dir=tmp_path / "chroma_test_del")

    hash1 = "hash_aaa"
    hash2 = "hash_bbb"

    chunks1 = make_chunks(file_hash=hash1)
    chunks2 = make_chunks(file_hash=hash2)

    store.add_chunks(chunks1, make_embeddings(len(chunks1), 1024))
    store.add_chunks(chunks2, make_embeddings(len(chunks2), 1024))

    # Delete hash1
    deleted = store.delete_by_hash(hash1)
    assert deleted > 0

    # hash2 should still be there
    results = store.query([0.1] * 1024, top_k=5)
    for r in results:
        assert r["file_hash"] != hash1


def test_persistence(tmp_path):
    """Test that data persists across store instances."""
    persist_dir = tmp_path / "chroma_persist"

    # First instance
    store1 = VectorStore(persist_dir=persist_dir)
    chunks = make_chunks()
    store1.add_chunks(chunks, make_embeddings(len(chunks), 1024))
    store1.update_file_metadata("abc123", "test.pdf", 3, 3)

    # Second instance should have the data
    store2 = VectorStore(persist_dir=persist_dir)
    assert store2.get_total_chunks() > 0
    assert store2.get_file_count() > 0

    files = store2.get_indexed_files()
    assert len(files) > 0
    assert files[0]["file_name"] == "test.pdf"


def test_get_indexed_files(tmp_path):
    """Test listing indexed files."""
    store = VectorStore(persist_dir=tmp_path / "chroma_list")

    chunks = make_chunks()
    store.add_chunks(chunks, make_embeddings(len(chunks), 1024))
    store.update_file_metadata("abc123", "test.pdf", 3, 3)

    files = store.get_indexed_files()
    assert len(files) == 1
    assert files[0]["file_name"] == "test.pdf"
    assert files[0]["pages"] == 3
    assert files[0]["chunks"] == 3
    assert "indexed_at" in files[0]


def test_clear_all(tmp_path):
    """Test clearing the entire store."""
    store = VectorStore(persist_dir=tmp_path / "chroma_clear")

    chunks = make_chunks()
    store.add_chunks(chunks, make_embeddings(len(chunks), 1024))
    store.update_file_metadata("abc123", "test.pdf", 3, 3)

    assert store.get_total_chunks() > 0

    store.clear_all()

    assert store.get_total_chunks() == 0
    assert store.get_file_count() == 0


def test_file_filter(tmp_path):
    """Test querying with file filter."""
    store = VectorStore(persist_dir=tmp_path / "chroma_filter")

    chunks_a = make_chunks(file_name="file_a.pdf", file_hash="hash_a")
    chunks_b = make_chunks(file_name="file_b.pdf", file_hash="hash_b")

    store.add_chunks(chunks_a, make_embeddings(len(chunks_a), 1024))
    store.add_chunks(chunks_b, make_embeddings(len(chunks_b), 1024))

    results = store.query([0.1] * 1024, top_k=10, file_filter=["file_a.pdf"])
    for r in results:
        assert r["file_name"] == "file_a.pdf"
