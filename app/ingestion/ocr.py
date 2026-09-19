"""
Smart OCR module for RAG ingestion pipeline.

This module provides intelligent OCR capabilities that selectively process pages
without selectable text. It uses PaddleOCR for scanned/image-only pages while
skipping unnecessary OCR on pages with extractable text to optimize performance.

Key features:
- Smart detection of selectable vs. scanned content
- Adaptive zoom with confidence-based retry mechanism
- Lazy-loading of OCR engine for memory efficiency
- Comprehensive logging and latency monitoring
"""

from pathlib import Path
from typing import Any, Optional
import logging

from app.config import settings
from app.config.constants import LogEvent
from app.core.exceptions import OCRError
from app.monitoring import get_logger, measure_latency


logger = get_logger("food_safety_rag.ingestion.ocr")

# =============================================================================
# CONSTANTS
# =============================================================================

MIN_ALPHANUMERIC_THRESHOLD = 20
DEFAULT_ZOOM_LEVEL = 1.0
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_RETRY_ZOOM = 2.0

# =============================================================================
# SMART OCR CLASS
# =============================================================================

class SmartOCR:
    """
    Intelligent OCR processor that applies OCR selectively.

    Only processes pages without selectable text (scanned pages, image-only pages).
    Uses adaptive zoom rendering: starts at initial zoom, retries at higher zoom
    if confidence is below threshold.

    Attributes:
        lang: Language code for OCR (default: 'en').
        use_gpu: Whether to use GPU acceleration (default: False).
        zoom_level: Initial zoom factor for page rendering.
        confidence_threshold: Minimum confidence to accept OCR results (0.0-1.0).
        retry_zoom: Zoom factor for retry when confidence is low.
        enabled: Whether OCR processing is enabled via settings.
    """

    def __init__(
        self,
        lang: str = "en",
        use_gpu: bool = False,
        zoom_level: Optional[float] = None,
        confidence_threshold: Optional[float] = None,
        retry_zoom: Optional[float] = None,
    ) -> None:
        """
        Initialize Smart OCR processor.

        Args:
            lang: Language code for OCR (default: 'en').
            use_gpu: Enable GPU acceleration (default: False).
            zoom_level: Initial zoom factor (default from settings or 1.0).
            confidence_threshold: Min confidence threshold (default from settings or 0.7).
            retry_zoom: Zoom factor for retry (default from settings or 2.0).

        Raises:
            ValueError: If confidence_threshold is out of range or zoom levels are invalid.
        """
        self.lang = lang
        self.use_gpu = use_gpu
        self.zoom_level = zoom_level or getattr(settings, "OCR_ZOOM_LEVEL", DEFAULT_ZOOM_LEVEL)
        self.confidence_threshold = confidence_threshold or getattr(
            settings, "OCR_CONFIDENCE_THRESHOLD", DEFAULT_CONFIDENCE_THRESHOLD
        )
        self.retry_zoom = retry_zoom or getattr(settings, "OCR_RETRY_ZOOM", DEFAULT_RETRY_ZOOM)
        self.enabled = getattr(settings, "OCR_ENABLED", True)

        # Validate configuration
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be between 0.0 and 1.0, "
                f"got {self.confidence_threshold}"
            )

        if self.zoom_level <= 0 or self.retry_zoom <= 0:
            raise ValueError(
                f"zoom levels must be positive, "
                f"got zoom_level={self.zoom_level}, retry_zoom={self.retry_zoom}"
            )

        if not self.enabled:
            logger.log_event(
                event=LogEvent.SYSTEM_STARTUP,
                message="OCR is disabled via settings.OCR_ENABLED=False",
                level=20,  # INFO
            )

        self._ocr_engine: Optional[Any] = None

    # =============================================================================
    # OCR ENGINE MANAGEMENT
    # =============================================================================

    def _get_ocr_engine(self) -> Any:
        """
        Lazy-load PaddleOCR engine on first use.

        Returns:
            The initialized PaddleOCR engine instance.

        Raises:
            OCRError: If OCR is disabled, PaddleOCR not installed, or initialization fails.
        """
        if not self.enabled:
            raise OCRError(
                message="OCR is disabled via settings.OCR_ENABLED=False",
                ocr_engine="PaddleOCR (disabled)",
                details={"enabled": False},
            )

        if self._ocr_engine is not None:
            return self._ocr_engine

        try:
            from paddleocr import PaddleOCR

            self._ocr_engine = PaddleOCR(
                use_angle_cls=True,
                lang=self.lang,
                use_gpu=self.use_gpu,
                show_log=False,
            )

            logger.log_event(
                event=LogEvent.OCR_START,
                message=f"PaddleOCR engine initialized (lang={self.lang}, gpu={self.use_gpu})",
                details={"lang": self.lang, "use_gpu": self.use_gpu},
            )
            return self._ocr_engine

        except ImportError as exc:
            raise OCRError(
                message="PaddleOCR not installed. Install with: pip install paddleocr",
                ocr_engine="PaddleOCR",
                details={"lang": self.lang, "use_gpu": self.use_gpu},
            ) from exc
        except Exception as exc:
            raise OCRError(
                message=f"Failed to initialize PaddleOCR: {str(exc)}",
                ocr_engine="PaddleOCR",
                details={"lang": self.lang, "use_gpu": self.use_gpu},
                original_exception=exc,
            ) from exc

    # =============================================================================
    # PAGE ANALYSIS
    # =============================================================================

    def _has_selectable_text(self, page_text: str) -> bool:
        """
        Determine if page contains sufficient selectable text.

        A page is considered to have selectable text if it contains at least
        MIN_ALPHANUMERIC_THRESHOLD (20) alphanumeric characters after stripping.

        Args:
            page_text: Text extracted from the page.

        Returns:
            True if page has sufficient selectable text, False otherwise.
        """
        if not page_text:
            return False

        stripped = page_text.strip()
        if not stripped:
            return False

        alphanumeric_count = sum(1 for c in stripped if c.isalnum())
        return alphanumeric_count >= MIN_ALPHANUMERIC_THRESHOLD

    # =============================================================================
    # IMAGE RENDERING
    # =============================================================================

    def _render_page_to_image(
        self,
        pdf_path: Path,
        page_number: int,
        zoom: float = 1.0
    ) -> Optional[Any]:
        """
        Render PDF page to image for OCR processing.

        Args:
            pdf_path: Path to PDF file.
            page_number: Page number (1-based indexing).
            zoom: Zoom factor for rendering (default: 1.0).

        Returns:
            PIL Image object, or None if rendering fails.

        Raises:
            OCRError: If OCR disabled, page out of range, or rendering fails.
        """
        if not self.enabled:
            raise OCRError(
                message="OCR is disabled, cannot render pages",
                ocr_engine="PyMuPDF",
                details={"enabled": False},
            )

        try:
            import fitz  # PyMuPDF

            doc = fitz.open(str(pdf_path))

            try:
                if not (1 <= page_number <= len(doc)):
                    raise OCRError(
                        message=f"Page {page_number} out of range (1-{len(doc)})",
                        document_path=str(pdf_path),
                        page_number=page_number,
                        ocr_engine="PyMuPDF",
                    )

                page = doc[page_number - 1]  # Convert to 0-based indexing

                # Render with zoom factor
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)

                # Convert to PIL Image
                from PIL import Image
                import io

                image_data = pix.tobytes("ppm")
                image = Image.open(io.BytesIO(image_data))

                return image

            finally:
                doc.close()

        except ImportError as exc:
            raise OCRError(
                message="PyMuPDF not installed. Install with: pip install pymupdf",
                ocr_engine="PyMuPDF",
                details={"page": page_number},
            ) from exc
        except OCRError:
            raise
        except Exception as exc:
            raise OCRError(
                message=f"Failed to render page {page_number}: {str(exc)}",
                document_path=str(pdf_path),
                page_number=page_number,
                ocr_engine="PyMuPDF",
                original_exception=exc,
            ) from exc

    # =============================================================================
    # OCR EXECUTION
    # =============================================================================

    def _run_ocr_on_image(self, image: Any) -> tuple[str, float]:
        """
        Run PaddleOCR on image and extract text with confidence score.

        Args:
            image: PIL Image object.

        Returns:
            Tuple of (extracted_text, average_confidence).

        Raises:
            OCRError: If OCR processing fails.
        """
        if not self.enabled:
            raise OCRError(
                message="OCR is disabled, cannot process images",
                ocr_engine="PaddleOCR",
                details={"enabled": False},
            )

        try:
            import numpy as np
            
            # Convert PIL Image to numpy array (PaddleOCR expects np.ndarray)
            if hasattr(image, 'convert'):
                # Convert to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                img_array = np.array(image)
            else:
                # If already a numpy array, use it directly
                img_array = image

            ocr_engine = self._get_ocr_engine()
            results = ocr_engine.ocr(img_array, cls=True)

            if not results or not results[0]:
                return "", 0.0

            # Extract text and calculate average confidence
            text_lines = []
            confidences = []

            for line in results[0]:
                text = line[1][0] if line[1] else ""
                confidence = float(line[1][1]) if len(line[1]) > 1 else 0.0

                if text:
                    text_lines.append(text)
                    confidences.append(confidence)

            extracted_text = "\n".join(text_lines)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            return extracted_text, avg_confidence

        except OCRError:
            raise
        except Exception as exc:
            raise OCRError(
                message=f"OCR processing failed: {str(exc)}",
                ocr_engine="PaddleOCR",
                original_exception=exc,
            ) from exc

    # =============================================================================
    # PAGE PROCESSING
    # =============================================================================

    def process_page(
        self,
        page_text: str,
        page_number: int,
        pdf_path: Optional[Path] = None,
    ) -> dict[str, Any]:
        """
        Process single page with smart OCR decision logic.

        Args:
            page_text: Text extracted from page by document parser.
            page_number: Page number (1-based).
            pdf_path: Path to PDF file (required for OCR).

        Returns:
            Dictionary with keys:
                - text: Extracted or OCR'd text
                - ocr_applied: Whether OCR was applied
                - ocr_confidence: OCR confidence (None if not applied)
                - page_number: Page number
                - zoom_used: Zoom factor used (None if OCR not applied)
                - retried: Whether retry with higher zoom occurred
        """
        if not self.enabled:
            logger.log_ingestion(
                event=LogEvent.OCR_SKIPPED,
                document_id="in_progress",
                document_name="page_ocr",
                message=f"OCR disabled, returning raw text for page {page_number}",
                details={"page_number": page_number, "reason": "ocr_disabled"},
            )
            return {
                "text": page_text,
                "ocr_applied": False,
                "ocr_confidence": None,
                "page_number": page_number,
                "zoom_used": None,
                "retried": False,
            }

        # Check if page has sufficient selectable text
        if self._has_selectable_text(page_text):
            logger.log_ingestion(
                event=LogEvent.OCR_SKIPPED,
                document_id="in_progress",
                document_name="page_ocr",
                message=f"Page {page_number} has selectable text, skipping OCR",
                details={
                    "page_number": page_number,
                    "text_length": len(page_text),
                    "alphanumeric_count": sum(1 for c in page_text if c.isalnum()),
                },
            )
            return {
                "text": page_text,
                "ocr_applied": False,
                "ocr_confidence": None,
                "page_number": page_number,
                "zoom_used": None,
                "retried": False,
            }

        # Page needs OCR - verify PDF path provided
        if pdf_path is None:
            logger.log_ingestion(
                event=LogEvent.OCR_SKIPPED,
                document_id="in_progress",
                document_name="page_ocr",
                message=f"Page {page_number} needs OCR but no PDF path provided, skipping",
                details={"page_number": page_number, "reason": "no_pdf_path"},
            )
            return {
                "text": page_text,
                "ocr_applied": False,
                "ocr_confidence": None,
                "page_number": page_number,
                "zoom_used": None,
                "retried": False,
            }

        # Run OCR with adaptive zoom retry
        logger.log_ingestion(
            event=LogEvent.OCR_START,
            document_id="in_progress",
            document_name="page_ocr",
            message=f"Running OCR on page {page_number} (zoom={self.zoom_level})",
            details={"page_number": page_number, "pdf_path": str(pdf_path), "zoom": self.zoom_level},
        )

        try:
            # First attempt at initial zoom
            image = self._render_page_to_image(pdf_path, page_number, zoom=self.zoom_level)
            if image is None:
                raise OCRError(
                    message=f"Failed to render page {page_number}",
                    document_path=str(pdf_path),
                    page_number=page_number,
                    ocr_engine="PyMuPDF",
                )

            ocr_text, confidence = self._run_ocr_on_image(image)
            zoom_used = self.zoom_level
            retried = False

            # Retry with higher zoom if confidence is below threshold
            if confidence < self.confidence_threshold and self.retry_zoom > self.zoom_level:
                logger.log_ingestion(
                    event=LogEvent.OCR_START,
                    document_id="in_progress",
                    document_name="page_ocr",
                    message=f"Page {page_number}: confidence {confidence:.2f} < {self.confidence_threshold}, "
                            f"retrying with zoom={self.retry_zoom}",
                    details={
                        "page_number": page_number,
                        "confidence": round(confidence, 3),
                        "threshold": self.confidence_threshold,
                        "retry_zoom": self.retry_zoom,
                    },
                )

                retry_image = self._render_page_to_image(pdf_path, page_number, zoom=self.retry_zoom)
                if retry_image is not None:
                    retry_text, retry_confidence = self._run_ocr_on_image(retry_image)
                    if retry_confidence > confidence:
                        ocr_text = retry_text
                        confidence = retry_confidence
                        zoom_used = self.retry_zoom
                        retried = True
                        logger.log_ingestion(
                            event=LogEvent.OCR_COMPLETE,
                            document_id="in_progress",
                            document_name="page_ocr",
                            message=f"Page {page_number}: improved confidence to {confidence:.2f}",
                            details={
                                "page_number": page_number,
                                "confidence": round(confidence, 3),
                                "zoom_used": zoom_used,
                            },
                        )

            logger.log_ingestion(
                event=LogEvent.OCR_COMPLETE,
                document_id="in_progress",
                document_name="page_ocr",
                message=f"OCR complete for page {page_number}",
                details={
                    "page_number": page_number,
                    "ocr_text_length": len(ocr_text),
                    "confidence": round(confidence, 3),
                    "zoom_used": zoom_used,
                    "retried": retried,
                },
            )

            return {
                "text": ocr_text,
                "ocr_applied": True,
                "ocr_confidence": confidence,
                "page_number": page_number,
                "zoom_used": zoom_used,
                "retried": retried,
            }

        except OCRError:
            raise
        except Exception as exc:
            raise OCRError(
                message=f"Unexpected error during OCR on page {page_number}: {str(exc)}",
                document_path=str(pdf_path),
                page_number=page_number,
                ocr_engine="PaddleOCR",
                original_exception=exc,
            ) from exc

    # =============================================================================
    # DOCUMENT PROCESSING
    # =============================================================================

    def process_document(
        self,
        pages: list[dict[str, Any]],
        pdf_path: Optional[Path] = None,
    ) -> list[dict[str, Any]]:
        """
        Process all pages of a document with smart OCR.

        Args:
            pages: List of page dictionaries with 'page_number' and 'text' keys.
            pdf_path: Path to PDF file (required for OCR fallback).

        Returns:
            List of processed page results with OCR metadata.
        """
        if not self.enabled:
            logger.log_ingestion(
                event=LogEvent.OCR_SKIPPED,
                document_id="in_progress",
                document_name="document_ocr",
                message="OCR disabled, returning all pages with raw text",
                details={"pages": len(pages), "reason": "ocr_disabled"},
            )
            return [
                {
                    "text": page.get("text", ""),
                    "ocr_applied": False,
                    "ocr_confidence": None,
                    "page_number": page.get("page_number", 0),
                    "zoom_used": None,
                    "retried": False,
                }
                for page in pages
            ]

        results: list[dict[str, Any]] = []
        ocr_count = 0
        retry_count = 0
        skipped_count = 0

        for page in pages:
            page_number = page.get("page_number", 0)
            page_text = page.get("text", "")

            # Use measure_latency as context manager manually
            with measure_latency("ocr_processing") as latency:
                result = self.process_page(
                    page_text=page_text,
                    page_number=page_number,
                    pdf_path=pdf_path,
                )
                latency.stop(
                    page_number=page_number,
                    ocr_applied=result["ocr_applied"],
                    confidence=result.get("ocr_confidence"),
                    zoom_used=result.get("zoom_used"),
                    retried=result.get("retried", False),
                )

            results.append(result)

            # Update counters
            if result["ocr_applied"]:
                ocr_count += 1
                if result.get("retried", False):
                    retry_count += 1
            else:
                skipped_count += 1

        logger.log_ingestion(
            event=LogEvent.OCR_COMPLETE,
            document_id="in_progress",
            document_name="document_ocr",
            message=f"Document processing complete",
            details={
                "total_pages": len(pages),
                "ocr_pages": ocr_count,
                "retry_pages": retry_count,
                "skipped_pages": skipped_count,
            },
        )

        return results