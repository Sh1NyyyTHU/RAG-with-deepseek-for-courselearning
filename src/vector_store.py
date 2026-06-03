"""
Vector store using ChromaDB PersistentClient.
Supports add, delete, reindex, list, and clear operations.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import shutil

import chromadb
from chromadb.config import Settings as ChromaSettings

from .utils import logger, compute_file_hash, now_iso
from .text_splitter import TextChunk
import config


def chunk_to_document(chunk: TextChunk) -> str:
    """Format chunk text for storage."""
    return chunk.text


class VectorStore:
    """Persistent vector store backed by ChromaDB."""

    COLLECTION_NAME = "courseware_chunks"
    METADATA_FILE = "index_metadata.json"

    def __init__(self, persist_dir: Optional[Path] = None):
        self.persist_dir = Path(persist_dir or config.CHROMA_DIR)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._metadata_path = self.persist_dir / self.METADATA_FILE
        self._index_metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load index metadata from disk."""
        if self._metadata_path.exists():
            with open(self._metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"files": {}}  # file_hash -> {file_name, pages, chunks, indexed_at}

    def _save_metadata(self):
        """Save index metadata to disk."""
        with open(self._metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._index_metadata, f, ensure_ascii=False, indent=2)

    def _make_chunk_id(self, file_hash: str, page_num: int, chunk_idx: int) -> str:
        """Generate a unique chunk ID."""
        return f"{file_hash[:12]}_p{page_num}_c{chunk_idx}"

    def add_chunks(self, chunks: List[TextChunk], embeddings: List[List[float]]) -> int:
        """Add chunks with their embeddings to the store.

        Returns number of chunks added.
        """
        if not chunks or not embeddings:
            return 0

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            chunk_id = self._make_chunk_id(
                chunk.file_hash, chunk.page_number, chunk.chunk_index
            )
            ids.append(chunk_id)
            documents.append(chunk.text)
            metadatas.append({
                "document_id": chunk.document_id,
                "file_name": chunk.file_name,
                "file_hash": chunk.file_hash,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "created_at": chunk.created_at,
            })

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("Added %d chunks to vector store", len(chunks))
        return len(chunks)

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 8,
        file_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Query the vector store for similar chunks.

        Args:
            query_embedding: The query embedding vector.
            top_k: Number of results to return.
            file_filter: Optional list of file names to filter by.

        Returns:
            List of result dicts with similarity, file_name, page_number, text, etc.
        """
        where_filter = None
        if file_filter:
            where_filter = {"file_name": {"$in": file_filter}}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        formatted = []
        if results and results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i] if results["distances"] else 0
                # Convert cosine distance to similarity (cosine distance in [0, 2], similarity = 1 - distance)
                similarity = 1.0 - (distance / 2.0) if distance is not None else 0

                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                formatted.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i] if results["documents"] else "",
                    "file_name": meta.get("file_name", "unknown"),
                    "page_number": meta.get("page_number", 0),
                    "chunk_index": meta.get("chunk_index", 0),
                    "file_hash": meta.get("file_hash", ""),
                    "similarity": round(similarity, 4),
                    "distance": round(distance, 4) if distance else 0,
                })

        return formatted

    def delete_by_hash(self, file_hash: str) -> int:
        """Delete all chunks for a given file hash. Returns number deleted."""
        # Query to find all chunk IDs for this hash
        existing = self.collection.get(
            where={"file_hash": file_hash},
            include=["metadatas"],
        )

        if existing and existing["ids"]:
            self.collection.delete(ids=existing["ids"])
            count = len(existing["ids"])
            logger.info("Deleted %d chunks for hash %s", count, file_hash[:12])

            # Update metadata
            if file_hash in self._index_metadata["files"]:
                del self._index_metadata["files"][file_hash]
                self._save_metadata()
            return count
        return 0

    def clear_all(self):
        """Delete all chunks and reset metadata."""
        # Delete the collection and recreate
        try:
            self.client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass
        self.collection = self.client.create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._index_metadata = {"files": {}}
        self._save_metadata()
        logger.info("Vector store cleared")

    def get_indexed_files(self) -> List[Dict[str, Any]]:
        """Return list of currently indexed files."""
        return [
            {
                "file_name": info.get("file_name", "unknown"),
                "file_hash": fhash,
                "pages": info.get("pages", 0),
                "chunks": info.get("chunks", 0),
                "indexed_at": info.get("indexed_at", ""),
            }
            for fhash, info in self._index_metadata["files"].items()
        ]

    def update_file_metadata(self, file_hash: str, file_name: str, pages: int, chunks: int):
        """Record a file's indexing metadata."""
        self._index_metadata["files"][file_hash] = {
            "file_name": file_name,
            "pages": pages,
            "chunks": chunks,
            "indexed_at": now_iso(),
        }
        self._save_metadata()

    def get_total_chunks(self) -> int:
        """Return total number of chunks in the collection."""
        try:
            return self.collection.count()
        except Exception:
            return 0

    def get_file_count(self) -> int:
        """Return number of distinct indexed files."""
        return len(self._index_metadata["files"])

    def is_file_indexed(self, file_hash: str) -> bool:
        """Check if a file (by hash) is already indexed."""
        return file_hash in self._index_metadata["files"]
