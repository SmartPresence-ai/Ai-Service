import hmac
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.gridfs_storage import delete_embedding
from app.model_loader import get_face_engine, is_face_engine_loaded
from app.recognition import (
    register_student_face,
    verify_attendance_images,
)

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the face model once before serving requests.

    The timings are logged because a request sent before startup finishes waits
    for it, which is indistinguishable from a slow request on the caller's side.
    """
    started_at = time.perf_counter()
    face_engine = await run_in_threadpool(get_face_engine)
    loaded_at = time.perf_counter()

    await run_in_threadpool(face_engine.warm_up)

    logger.info(
        "face model ready in %.2fs (load %.2fs, warm-up %.2fs) - requests before "
        "this point were queued, not slow",
        time.perf_counter() - started_at,
        loaded_at - started_at,
        time.perf_counter() - loaded_at,
    )
    yield


app = FastAPI(
    title="AI Attendance Face Service",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_api_key(
    x_ai_service_key: str | None = Header(default=None),
) -> None:
    """Reject requests without the shared secret in the X-AI-Service-Key header.

    The check is disabled while AI_SERVICE_API_KEY is empty, so existing
    deployments keep working until a key is configured on both sides.
    """
    expected = settings.ai_service_api_key
    if not expected:
        return
    if not x_ai_service_key or not hmac.compare_digest(
        x_ai_service_key.encode("utf-8"),
        expected.encode("utf-8"),
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


protected = [Depends(require_api_key)]


@app.get("/health")
def health() -> dict[str, object]:
    """Liveness probe; left unauthenticated so load balancers can call it."""
    return {"status": "running", "model_loaded": is_face_engine_loaded()}


@app.post("/register-face", dependencies=protected)
async def register_face(
    school_id: str = Form(...),
    student_id: str = Form(...),
    image: UploadFile = File(...),
) -> dict[str, str]:
    """Register or replace a student's face embedding from one photo."""
    try:
        image_bytes = await image.read()

        if not image_bytes:
            raise HTTPException(
                status_code=422,
                detail="Image is empty.",
            )
        return await run_in_threadpool(
            register_student_face,
            face_engine=get_face_engine(),
            school_id=school_id.strip(),
            student_id=student_id.strip(),
            image_bytes=image_bytes,
        )

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Unhandled error in /register-face")
        raise HTTPException(
            status_code=500,
            detail="Internal server error.",
        ) from exc


@app.post("/verify-attendance", dependencies=protected)
async def verify_attendance(
    school_id: str = Form(...),
    images: list[UploadFile] = File(...),
    student_ids: str | None = Form(default=None),
) -> dict[str, object]:
    """Mark students Present or Absent from one or more class photos.

    student_ids is an optional comma-separated roster; blank or duplicate
    entries are dropped, and an empty roster means the whole school.
    """
    try:
        image_bytes_list = [
            await uploaded_image.read()
            for uploaded_image in images
        ]

        if not image_bytes_list:
            raise HTTPException(
                status_code=422,
                detail="At least one attendance image is required.",
            )

        parsed_student_ids: list[str] | None = None

        if student_ids and student_ids.strip():
            parsed_student_ids = list(
                dict.fromkeys(
                    student_id.strip()
                    for student_id in student_ids.split(",")
                    if student_id.strip()
                )
            )

            if not parsed_student_ids:
                parsed_student_ids = None

        return await run_in_threadpool(
            verify_attendance_images,
            face_engine=get_face_engine(),
            school_id=school_id.strip(),
            student_ids=parsed_student_ids,
            image_bytes_list=image_bytes_list,
        )

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Unhandled error in /verify-attendance")
        raise HTTPException(
            status_code=500,
            detail="Internal server error.",
        ) from exc


@app.delete("/delete-face", dependencies=protected)
async def delete_face(
    school_id: str = Form(...),
    student_id: str = Form(...),
) -> dict[str, str]:
    """Delete a student's stored face embedding."""
    try:
        deleted = await run_in_threadpool(
            delete_embedding,
            school_id=school_id.strip(),
            student_id=student_id.strip(),
        )

        if not deleted:
            raise HTTPException(
                status_code=404,
                detail="Face embedding not found.",
            )

        return {
            "message": "Face embedding deleted successfully."
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Unhandled error in /delete-face")
        raise HTTPException(
            status_code=500,
            detail="Internal server error.",
        ) from exc
