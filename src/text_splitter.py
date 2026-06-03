"""
Configurable text splitter for PDF pages.
Does NOT split across pages. Preserves page-level metadata.
"""
from typing import List, Dict, Any
from dataclasses import dataclass, field

from .pdf_parser import PDFPage
from .utils import now_iso


@dataclass
class TextChunk:
    """A single chunk of text with page-level metadata."""

    document_id: str
    file_name: str
    file_hash: str
    page_number: int
    chunk_index: int
    text: str
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "file_name": self.file_name,
            "file_hash": self.file_hash,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "created_at": self.created_at,
        }


class TextSplitter:
    """Split text into chunks without crossing page boundaries."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_page(
        self,
        page: PDFPage,
        document_id: str,
        file_name: str,
        file_hash: str,
    ) -> List[TextChunk]:
        """Split a single page into chunks. Simple sliding window approach."""
        text = page.text
        if not text:
            return []

        chunks = []
        start = 0
        chunk_idx = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk_text = text[start:end]

            if chunk_text.strip():
                chunks.append(TextChunk(
                    document_id=document_id,
                    file_name=file_name,
                    file_hash=file_hash,
                    page_number=page.page_number,
                    chunk_index=chunk_idx,
                    text=chunk_text,
                ))
                chunk_idx += 1

            if end >= text_len:
                break
            start = end - self.chunk_overlap

        return chunks

    def split_pages(
        self,
        pages: List[PDFPage],
        document_id: str,
        file_name: str,
        file_hash: str,
    ) -> List[TextChunk]:
        """Split multiple pages into chunks. Each page is processed independently."""
        all_chunks = []
        for page in pages:
            chunks = self.split_page(page, document_id, file_name, file_hash)
            all_chunks.extend(chunks)
        return all_chunks
