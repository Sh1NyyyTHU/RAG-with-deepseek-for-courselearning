"""
Knowledge base orchestration layer.
Coordinates PDF parsing, text splitting, embedding, and vector storage.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import shutil

from .pdf_parser import PDFParser, PDFDocument
from .text_splitter import TextSplitter, TextChunk
from .embedding_service import EmbeddingService
from .vector_store import VectorStore
from .utils import logger, compute_file_hash
import config


class KnowledgeBase:
    """Orchestrates the full indexing and retrieval pipeline."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.pdf_parser = PDFParser()

    def _get_splitter(self, chunk_size: int = None, chunk_overlap: int = None) -> TextSplitter:
        return TextSplitter(
            chunk_size=chunk_size or config.CHUNK_SIZE,
            chunk_overlap=chunk_overlap or config.CHUNK_OVERLAP,
        )

    def index_pdf(
        self,
        file_path: Path,
        chunk_size: int = None,
        chunk_overlap: int = None,
        progress_callback: callable = None,
    ) -> Dict[str, Any]:
        """Index a single PDF file.

        Args:
            file_path: Path to the PDF.
            chunk_size: Override default chunk size.
            chunk_overlap: Override default chunk overlap.
            progress_callback: Optional callable(step, total) for progress.

        Returns:
            Dict with indexing result.
        """
        file_path = Path(file_path)
        file_hash = compute_file_hash(file_path)

        # Check if already indexed
        if self.vector_store.is_file_indexed(file_hash):
            logger.info("File already indexed, skipping: %s", file_path.name)
            return {
                "status": "skipped",
                "file_name": file_path.name,
                "file_hash": file_hash,
                "reason": "already_indexed",
            }

        if progress_callback:
            progress_callback(1, 4)

        # 1. Parse PDF
        try:
            pdf_doc = self.pdf_parser.parse(file_path)
        except Exception as e:
            logger.error("Failed to parse PDF %s: %s", file_path.name, e)
            return {
                "status": "error",
                "file_name": file_path.name,
                "file_hash": file_hash,
                "error": str(e),
            }

        if progress_callback:
            progress_callback(2, 4)

        # 2. Check for scanned pages
        scan_warnings = self.pdf_parser.get_scan_warnings(pdf_doc)

        if progress_callback:
            progress_callback(3, 4)

        # 3. Split into chunks
        splitter = self._get_splitter(chunk_size, chunk_overlap)
        document_id = file_hash[:16]
        chunks = splitter.split_pages(
            pdf_doc.pages,
            document_id=document_id,
            file_name=pdf_doc.file_name,
            file_hash=file_hash,
        )

        if not chunks:
            logger.warning("No text chunks produced for %s", file_path.name)
            return {
                "status": "warning",
                "file_name": file_path.name,
                "file_hash": file_hash,
                "pages": pdf_doc.page_count,
                "chunks": 0,
                "warnings": ["未能提取到可索引的文字内容"],
            }

        # 4. Generate embeddings
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_service.encode(texts)

        if embeddings is None:
            return {
                "status": "error",
                "file_name": file_path.name,
                "file_hash": file_hash,
                "error": "Embedding生成失败",
            }

        # 5. Store in vector DB
        self.vector_store.add_chunks(chunks, embeddings)

        # 6. Update metadata
        self.vector_store.update_file_metadata(
            file_hash=file_hash,
            file_name=pdf_doc.file_name,
            pages=pdf_doc.page_count,
            chunks=len(chunks),
        )

        if progress_callback:
            progress_callback(4, 4)

        logger.info(
            "Indexed PDF: %s — %d pages, %d chunks",
            pdf_doc.file_name, pdf_doc.page_count, len(chunks),
        )

        return {
            "status": "ok",
            "file_name": pdf_doc.file_name,
            "file_hash": file_hash,
            "pages": pdf_doc.page_count,
            "chunks": len(chunks),
            "scanned": pdf_doc.is_scanned,
            "warnings": scan_warnings,
        }

    def delete_pdf(self, file_hash: str) -> int:
        """Delete a PDF's chunks from the vector store."""
        return self.vector_store.delete_by_hash(file_hash)

    def reindex_pdf(
        self,
        file_path: Path,
        chunk_size: int = None,
        chunk_overlap: int = None,
        progress_callback: callable = None,
    ) -> Dict[str, Any]:
        """Reindex a PDF (delete old chunks and re-add)."""
        file_hash = compute_file_hash(file_path)
        self.delete_pdf(file_hash)
        return self.index_pdf(file_path, chunk_size, chunk_overlap, progress_callback)

    def clear(self):
        """Clear all indexed data."""
        self.vector_store.clear_all()

    def get_status(self) -> Dict[str, Any]:
        """Get current knowledge base status."""
        files = self.vector_store.get_indexed_files()
        return {
            "file_count": len(files),
            "total_chunks": self.vector_store.get_total_chunks(),
            "files": files,
            "embedding_device": self.embedding_service.device,
            "embedding_initialized": self.embedding_service._initialized,
        }

    def copy_pdf_to_store(self, source_path: Path) -> Path:
        """Copy a PDF to the managed PDF directory. Returns the new path."""
        dest = config.PDF_DIR / source_path.name
        if dest.exists():
            # If already exists, append a suffix
            stem = source_path.stem
            suffix = source_path.suffix
            counter = 1
            while dest.exists():
                dest = config.PDF_DIR / f"{stem}_{counter}{suffix}"
                counter += 1
        shutil.copy2(source_path, dest)
        return dest
