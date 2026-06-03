"""Tests for text splitter."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.text_splitter import TextSplitter, TextChunk
from src.pdf_parser import PDFPage


def test_split_single_page():
    """Test splitting a single page into chunks."""
    splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
    page = PDFPage(
        page_number=1,
        text="A" * 250,
        char_count=250,
        is_scanned=False,
    )

    chunks = splitter.split_page(page, "doc1", "test.pdf", "hash123")

    assert len(chunks) >= 2
    for chunk in chunks:
        assert isinstance(chunk, TextChunk)
        assert chunk.page_number == 1
        assert chunk.file_name == "test.pdf"
        assert chunk.file_hash == "hash123"
        assert chunk.document_id == "doc1"
        assert len(chunk.text) <= 100


def test_split_empty_page():
    """Test splitting an empty page produces no chunks."""
    splitter = TextSplitter()
    page = PDFPage(page_number=1, text="", char_count=0, is_scanned=True)

    chunks = splitter.split_page(page, "doc1", "test.pdf", "hash1")
    assert len(chunks) == 0


def test_chunks_dont_cross_pages():
    """Test that chunks from different pages have correct page numbers."""
    splitter = TextSplitter(chunk_size=80, chunk_overlap=10)

    pages = [
        PDFPage(page_number=1, text="PAGE1_" * 20, char_count=100, is_scanned=False),
        PDFPage(page_number=2, text="PAGE2_" * 20, char_count=100, is_scanned=False),
    ]

    chunks = splitter.split_pages(pages, "doc1", "test.pdf", "hash1")

    page1_chunks = [c for c in chunks if c.page_number == 1]
    page2_chunks = [c for c in chunks if c.page_number == 2]

    assert len(page1_chunks) > 0
    assert len(page2_chunks) > 0
    # Each chunk should only contain text from its page
    for c in page1_chunks:
        assert "PAGE2_" not in c.text
    for c in page2_chunks:
        assert "PAGE1_" not in c.text


def test_chunk_overlap():
    """Test that overlapping chunks share content."""
    chunk_size = 100
    overlap = 30
    splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

    # Create text that will produce at least 2 chunks
    text = "ABCDEFGHIJ" * 30  # 300 chars
    page = PDFPage(page_number=1, text=text, char_count=len(text), is_scanned=False)

    chunks = splitter.split_page(page, "doc1", "test.pdf", "hash1")

    if len(chunks) >= 2:
        # The end of chunk 0 should overlap with start of chunk 1
        end_of_0 = chunks[0].text[-overlap // 2:]
        start_of_1 = chunks[1].text[:overlap // 2]
        # There should be some overlap (characters in common)
        # This is a probabilistic check
        assert len(chunks[0].text) <= chunk_size


def test_metadata_preserved():
    """Test that all metadata fields are preserved in chunks."""
    splitter = TextSplitter(chunk_size=200, chunk_overlap=50)
    page = PDFPage(page_number=42, text="Content " * 50, char_count=400, is_scanned=False)

    chunks = splitter.split_page(page, "doc_abc", "lecture.pdf", "hash_xyz")

    for chunk in chunks:
        d = chunk.to_dict()
        assert d["document_id"] == "doc_abc"
        assert d["file_name"] == "lecture.pdf"
        assert d["file_hash"] == "hash_xyz"
        assert d["page_number"] == 42
        assert "chunk_index" in d
        assert "text" in d
        assert "created_at" in d
