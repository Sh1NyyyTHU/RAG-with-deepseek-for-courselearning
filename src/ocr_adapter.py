"""
OCR adapter for scanned PDF pages.
MVP: provides interface and PaddleOCR integration if available.
Core functionality does NOT depend on OCR being installed.
"""
from pathlib import Path
from typing import Optional, List
import io

import fitz  # PyMuPDF

from .utils import logger


class OCRAdapter:
    """Adapter for OCR on scanned PDF pages.

    MVP behavior:
    - Detect if PaddleOCR is available.
    - If available, provide OCR for page images.
    - If unavailable, return None gracefully.
    - Core PDF text extraction works without this.
    """

    def __init__(self):
        self._ocr = None
        self._available = False
        self._init_attempted = False

    @property
    def available(self) -> bool:
        if not self._init_attempted:
            self._initialize()
        return self._available

    def _initialize(self):
        """Try to import and initialize PaddleOCR."""
        self._init_attempted = True
        try:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang="ch",
                use_gpu=False,  # Use CPU for OCR by default
                show_log=False,
            )
            self._available = True
            logger.info("PaddleOCR initialized successfully")
        except ImportError:
            logger.info("PaddleOCR not installed — OCR features disabled")
            self._available = False
        except Exception as e:
            logger.warning("PaddleOCR initialization failed: %s", e)
            self._available = False

    def extract_page_text(self, pdf_path: Path, page_num: int) -> Optional[str]:
        """Extract text from a specific PDF page using OCR.

        Args:
            pdf_path: Path to the PDF file.
            page_num: 1-based page number.

        Returns:
            Extracted text string, or None if OCR unavailable/failed.
        """
        if not self.available:
            return None

        try:
            doc = fitz.open(str(pdf_path))
            if page_num < 1 or page_num > len(doc):
                doc.close()
                return None

            page = doc[page_num - 1]
            # Render page to image at 300 DPI
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")
            doc.close()

            # OCR the image
            result = self._ocr.ocr(img_bytes, cls=True)
            if result and result[0]:
                lines = [line[1][0] for line in result[0]]
                return "\n".join(lines)
            return ""
        except Exception as e:
            logger.error("OCR failed for %s page %d: %s", pdf_path.name, page_num, e)
            return None

    def extract_image_text(self, image_bytes: bytes) -> Optional[str]:
        """Extract text from an image (PNG/JPG/JPEG) using OCR.

        Args:
            image_bytes: Raw image bytes.

        Returns:
            Extracted text string, or None if OCR unavailable/failed.
        """
        if not self.available:
            return None

        try:
            result = self._ocr.ocr(image_bytes, cls=True)
            if result and result[0]:
                lines = [line[1][0] for line in result[0]]
                return "\n".join(lines)
            return ""
        except Exception as e:
            logger.error("OCR failed for image: %s", e)
            return None
