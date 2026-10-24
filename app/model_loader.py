"""Singleton model loader for the AI service.

The InsightFace model is expensive to initialize. Keep one engine instance per
Python process and reuse it for every registration and verification request.
"""

from threading import Lock
from app.face_engine import FaceEngine, InsightFaceBuffaloEngine

_face_engine: FaceEngine | None = None
_face_engine_lock = Lock()


def get_face_engine() -> FaceEngine:
    """Return the process-wide face engine, loading it only on first use."""
    global _face_engine
    if _face_engine is None:
        with _face_engine_lock:
            if _face_engine is None:
                _face_engine = InsightFaceBuffaloEngine()
    return _face_engine


def is_face_engine_loaded() -> bool:
    """Whether this process already holds the model, for readiness checks."""
    return _face_engine is not None
