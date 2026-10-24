import cv2
import warnings
import numpy as np
from io import StringIO
from app.config import settings
from abc import ABC, abstractmethod
from typing import Iterable, Iterator
from insightface.app import FaceAnalysis
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout


class FaceEngine(ABC):
    """Contract every face model adapter must implement."""

    @abstractmethod
    def image_to_embeddings(self, image_bytes: bytes) -> list[np.ndarray]:
        """Return one normalized embedding per detected face."""

    def warm_up(self) -> None:
        """Run one throwaway inference so the first real request is not slow."""


class InsightFaceBuffaloEngine(FaceEngine):
    """InsightFace Buffalo_L adapter used by the AI service.
    The rest of the system depends only on the FaceEngine contract, so Django,
    React, and API payloads remain unchanged if this adapter is replaced later.
    """

    def __init__(self) -> None:
        """Load and prepare the model, hiding InsightFace's startup console noise."""
        with _suppress_insightface_console_output(redirect_console=True):
            self.app = FaceAnalysis(
                name=settings.insightface_model_name,
                providers=settings.insightface_providers,
                allowed_modules=settings.insightface_allowed_modules,
            )
            self.app.prepare(
                ctx_id=settings.insightface_ctx_id,
                det_size=settings.detection_size,
                det_thresh=settings.detection_threshold,
            )

    def image_to_embeddings(self, image_bytes: bytes) -> list[np.ndarray]:
        """Decode the image and return one normalized float32 embedding per face."""
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Unable to decode image bytes.")

        with _suppress_insightface_console_output():
            faces = self.app.get(image)

        return [face.normed_embedding.astype(np.float32, copy=False) for face in faces]

    def warm_up(self) -> None:
        """Run detection and recognition once on blank input to initialise ONNX sessions."""
        width, height = settings.detection_size
        blank = np.zeros((height, width, 3), dtype=np.uint8)

        with _suppress_insightface_console_output():
            self.app.get(blank)

            recognition_model = self.app.models.get("recognition")
            if recognition_model is not None:
                recognition_model.get_feat(np.zeros((112, 112, 3), dtype=np.uint8))


def stack_embeddings(embeddings: Iterable[np.ndarray]) -> np.ndarray:
    """Pack embeddings into one contiguous ``(count, dimensions)`` float32 matrix."""
    matrix = np.stack(
        [np.asarray(embedding, dtype=np.float32).reshape(-1) for embedding in embeddings]
    )
    return np.ascontiguousarray(matrix, dtype=np.float32)


def match_embeddings(
    queries: np.ndarray,
    known_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the best-matching row index and cosine score for every query row.
    One BLAS matrix multiply replaces a Python loop over every query/known pair,
    which is where verification time went once a class roster grew past a few
    dozen students.
    """
    if queries.size == 0 or known_matrix.size == 0:
        empty_indices = np.empty(0, dtype=np.intp)
        return empty_indices, np.empty(0, dtype=np.float32)

    scores = queries @ known_matrix.T
    best_indices = scores.argmax(axis=1)
    best_scores = scores[np.arange(scores.shape[0]), best_indices]
    return best_indices, best_scores


@contextmanager
def _suppress_insightface_console_output(
    redirect_console: bool = False,
) -> Iterator[None]:
    """Suppress FutureWarnings without redirecting stdout during parallel inference.
    stdout/stderr redirection is process-global and not thread-safe, so it is only
    used for model startup logs before parallel verification workers run.
    """
    output_context = redirect_stdout(StringIO()) if redirect_console else nullcontext()
    error_context = redirect_stderr(StringIO()) if redirect_console else nullcontext()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        with output_context, error_context:
            yield
