"""
PDF parser using PyMuPDF (fitz).
Extracts text page by page, preserving exact page numbers.
"""
from pathlib import Path
from typing import List, Dict, Any
import fitz  # PyMuPDF

from .utils import logger, compute_file_hash, now_iso


class PDFPage:
    """Represents a single page extracted from a PDF."""

    def __init__(
        self,
        page_number: int,
        text: str,
        char_count: int,
        is_scanned: bool = False,
    ):
        self.page_number = page_number
        self.text = text
        self.char_count = char_count
        self.is_scanned = is_scanned

    def __repr__(self):
        return f"PDFPage(page={self.page_number}, chars={self.char_count}, scanned={self.is_scanned})"


class PDFDocument:
    """Represents a parsed PDF document with metadata."""

    def __init__(
        self,
        file_path: Path,
        pages: List[PDFPage],
        file_hash: str,
    ):
        self.file_path = file_path
        self.file_name = file_path.name
        self.pages = pages
        self.file_hash = file_hash
        self.page_count = len(pages)
        self.total_chars = sum(p.char_count for p in pages)
        self.is_scanned = self._detect_scanned()

    def _detect_scanned(self) -> bool:
        """Detect if document is likely a scanned PDF.

        Returns True if more than 50% of pages have very low character count.
        """
        if not self.pages:
            return False
        low_char_pages = sum(1 for p in self.pages if p.char_count < 50)
        return (low_char_pages / len(self.pages)) > 0.5

    def __repr__(self):
        return f"PDFDocument(name={self.file_name}, pages={self.page_count}, scanned={self.is_scanned})"


class PDFParser:
    """Parse PDF files using PyMuPDF, preserving page-level granularity."""

    SCAN_THRESHOLD = 30  # minimum characters to consider a page as having text

    @staticmethod
    def parse(file_path: Path) -> PDFDocument:
        """Parse a single PDF file into a PDFDocument.

        Args:
            file_path: Path to the PDF file.

        Returns:
            PDFDocument with extracted pages and metadata.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file is not a valid PDF.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        file_hash = compute_file_hash(file_path)

        try:
            doc = fitz.open(str(file_path))
        except Exception as e:
            raise ValueError(f"Failed to open PDF: {file_path} — {e}")

        pages = []
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            # Use sort=True to get text in reading order
            text = page.get_text("text", sort=True)
            char_count = len(text.strip())
            is_scanned = char_count < PDFParser.SCAN_THRESHOLD

            pages.append(PDFPage(
                page_number=page_idx + 1,  # 1-based page numbering
                text=text.strip(),
                char_count=char_count,
                is_scanned=is_scanned,
            ))

        doc.close()
        result = PDFDocument(file_path=file_path, pages=pages, file_hash=file_hash)

        logger.info(
            "Parsed PDF: %s — %d pages, %d total chars, scanned=%s",
            result.file_name, result.page_count, result.total_chars, result.is_scanned,
        )
        return result

    @staticmethod
    def get_scan_warnings(doc: PDFDocument) -> List[str]:
        """Generate human-readable warnings for scanned/low-quality pages."""
        warnings = []
        if doc.is_scanned:
            warnings.append(
                f"⚠️ PDF '{doc.file_name}' 可能是扫描件或文字提取失败。"
                f"仅有 {sum(1 for p in doc.pages if not p.is_scanned)}/{doc.page_count} 页提取到足够文字。"
            )

        scanned_pages = [p.page_number for p in doc.pages if p.is_scanned]
        if scanned_pages and not doc.is_scanned:
            if len(scanned_pages) <= 5:
                pages_str = ", ".join(map(str, scanned_pages))
                warnings.append(
                    f"⚠️ PDF '{doc.file_name}' 第 {pages_str} 页文字提取量较低，可能为扫描图片。"
                )
            else:
                warnings.append(
                    f"⚠️ PDF '{doc.file_name}' 有 {len(scanned_pages)} 页文字提取量较低。"
                )

        return warnings
