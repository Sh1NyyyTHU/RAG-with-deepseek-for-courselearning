"""Tests for PDF parser."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import fitz

from src.pdf_parser import PDFParser, PDFDocument, PDFPage


def create_test_pdf(filepath: Path, pages_text: list):
    """Create a test PDF with given pages of text."""
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=12)
    doc.save(str(filepath))
    doc.close()


def test_parse_simple_pdf(tmp_path):
    """Test parsing a simple text PDF."""
    pdf_path = tmp_path / "test_simple.pdf"
    create_test_pdf(pdf_path, [
        "This is page 1 content. It discusses gas discharge theory.",
        "This is page 2 content. It covers Townsend discharge.",
        "Page 3 has information about Paschen's law.",
    ])

    doc = PDFParser.parse(pdf_path)

    assert isinstance(doc, PDFDocument)
    assert doc.page_count == 3
    assert doc.file_name == "test_simple.pdf"
    assert not doc.is_scanned

    assert len(doc.pages) == 3
    assert doc.pages[0].page_number == 1
    assert "gas discharge" in doc.pages[0].text
    assert doc.pages[1].page_number == 2
    assert "Townsend discharge" in doc.pages[1].text
    assert doc.pages[2].page_number == 3
    assert "Paschen" in doc.pages[2].text


def test_parse_empty_pdf(tmp_path):
    """Test parsing a PDF with empty pages."""
    pdf_path = tmp_path / "test_empty.pdf"
    doc_fitz = fitz.open()
    for _ in range(3):
        doc_fitz.new_page()
    doc_fitz.save(str(pdf_path))
    doc_fitz.close()

    doc = PDFParser.parse(pdf_path)

    assert doc.page_count == 3
    assert doc.total_chars == 0
    # All empty pages should be marked as scanned
    assert doc.is_scanned


def test_scanned_detection(tmp_path):
    """Test scanned PDF detection."""
    pdf_path = tmp_path / "test_mixed.pdf"
    create_test_pdf(pdf_path, [
        "Page 1 has a lot of text content for testing purposes. " * 5,
        "",  # empty page
        "Page 3 has some text.",
    ])

    doc = PDFParser.parse(pdf_path)

    # Only 1 out of 3 pages has low chars -> not scanned overall
    assert doc.page_count == 3
    assert doc.pages[0].char_count > 50
    assert doc.pages[1].char_count == 0
    assert doc.pages[1].is_scanned


def test_file_not_found():
    """Test error on non-existent file."""
    with pytest.raises(FileNotFoundError):
        PDFParser.parse(Path("/nonexistent/file.pdf"))


def test_page_numbers_are_1_based(tmp_path):
    """Test that page numbers start from 1."""
    pdf_path = tmp_path / "test_pages.pdf"
    create_test_pdf(pdf_path, ["Page A", "Page B", "Page C"])

    doc = PDFParser.parse(pdf_path)
    for i, page in enumerate(doc.pages):
        assert page.page_number == i + 1
