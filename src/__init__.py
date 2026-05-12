"""Vietnamese handwriting OCR pipeline.

Phase 1 modules:
- :class:`src.detector.TextDetector` (PaddleOCR DBNet wrapper)
- :class:`src.recognizer.TextRecognizer` (VietOCR wrapper)
- :class:`src.pipeline.HandwritingOCRPipeline` (end-to-end)
"""

__version__ = "0.1.0"
