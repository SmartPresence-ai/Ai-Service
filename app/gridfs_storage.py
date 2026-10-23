import io
import numpy as np
from app.database import bucket, files_collection


def save_embedding(
    school_id: str,
    student_id: str,
    embedding: np.ndarray,
) -> None:
    """Store a student's embedding in GridFS as a .npy file."""
    buffer = io.BytesIO()
    np.save(buffer, embedding)
    buffer.seek(0)

    bucket.upload_from_stream(
        f"{school_id}_{student_id}.npy",
        buffer,
        metadata={
            "school_id": school_id,
            "student_id": student_id,
        },
    )


def delete_embedding(
    school_id: str,
    student_id: str,
) -> bool:
    """Delete every embedding stored for a student; return False if none existed."""
    files = list(
        files_collection.find(
            {
                "metadata.school_id": school_id,
                "metadata.student_id": student_id,
            }
        )
    )

    if not files:
        return False

    for file in files:
        bucket.delete(file["_id"])

    return True


def load_embeddings(
    school_id: str,
    student_ids: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Return embeddings for a school keyed by student_id, optionally limited to student_ids."""
    query = {
        "metadata.school_id": school_id,
    }

    if student_ids:
        query["metadata.student_id"] = {
            "$in": student_ids,
        }

    embeddings: dict[str, np.ndarray] = {}

    files = files_collection.find(query)

    for file in files:
        stream = io.BytesIO()

        bucket.download_to_stream(
            file["_id"],
            stream,
        )

        stream.seek(0)

        embeddings[file["metadata"]["student_id"]] = np.load(
            stream,
            allow_pickle=False,
        )

    return embeddings
