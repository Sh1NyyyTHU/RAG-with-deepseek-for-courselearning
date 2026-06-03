"""
Batch ingest all PDFs from a folder into the knowledge base.

Usage: python scripts/ingest_folder.py [folder_path]
"""
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging
from src.embedding_service import EmbeddingService
from src.vector_store import VectorStore
from src.knowledge_base import KnowledgeBase
import config


def main():
    logger = setup_logging()
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else config.PDF_DIR

    if not folder.exists():
        logger.error("Folder not found: %s", folder)
        sys.exit(1)

    pdf_files = list(folder.glob("*.pdf")) + list(folder.glob("*.PDF"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", folder)
        sys.exit(0)

    logger.info("Found %d PDF files", len(pdf_files))

    # Initialize services
    logger.info("Initializing embedding service...")
    emb = EmbeddingService()
    if not emb.initialize():
        logger.error("Failed to initialize embedding service")
        sys.exit(1)

    vs = VectorStore()
    kb = KnowledgeBase(emb, vs)

    # Ingest each PDF
    for i, pdf_path in enumerate(pdf_files, 1):
        logger.info("[%d/%d] Processing: %s", i, len(pdf_files), pdf_path.name)
        result = kb.index_pdf(pdf_path)
        if result["status"] == "ok":
            logger.info("  -> %d pages, %d chunks", result["pages"], result["chunks"])
        elif result["status"] == "skipped":
            logger.info("  -> Already indexed, skipped")
        else:
            logger.error("  -> Error: %s", result.get("error", "unknown"))

    # Summary
    status = kb.get_status()
    logger.info("Done. Total files: %d, total chunks: %d", status["file_count"], status["total_chunks"])


if __name__ == "__main__":
    main()
