"""
Session history manager with markdown export.
Stores Q&A history in-memory for the session and supports export.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from pathlib import Path
import json

from .utils import logger
import config


class HistoryManager:
    """In-memory session history with export capability."""

    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def add(
        self,
        question: str,
        answer: str,
        mode: str,
        sources: List[Dict[str, Any]],
        retrieved_chunks: List[Dict[str, Any]],
    ):
        """Add a Q&A record to history."""
        self.records.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "answer": answer,
            "mode": mode,
            "sources": [
                {"file_name": s["file_name"], "page_number": s["page_number"]}
                for s in sources
            ],
            "num_chunks": len(retrieved_chunks),
        })

    def get_all(self) -> List[Dict[str, Any]]:
        """Return all history records, most recent first."""
        return list(reversed(self.records))

    def clear(self):
        """Clear history."""
        self.records = []

    def export_markdown(self) -> str:
        """Export history as a Markdown string."""
        lines = [
            "# 课程课件问答记录",
            f"\n导出时间：{datetime.now(timezone.utc).isoformat()}\n",
            f"共 {len(self.records)} 条记录\n",
            "---\n",
        ]

        for i, record in enumerate(self.records, 1):
            lines.append(f"## 记录 {i}")
            lines.append(f"**时间**：{record['timestamp']}")
            lines.append(f"**模式**：{record['mode']}")
            lines.append(f"**问题**：{record['question']}")
            lines.append(f"\n**回答**：\n{record['answer']}\n")

            if record["sources"]:
                lines.append("**引用来源**：")
                for src in record["sources"]:
                    lines.append(f"- {src['file_name']}，第 {src['page_number']} 页")
                lines.append("")

            lines.append("---\n")

        return "\n".join(lines)

    def save_export(self) -> Path:
        """Save history to a markdown file, returns the path."""
        markdown = self.export_markdown()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"qa_history_{timestamp}.md"
        filepath = config.EXPORTS_DIR / filename
        filepath.write_text(markdown, encoding="utf-8")
        logger.info("History exported to %s", filepath)
        return filepath
