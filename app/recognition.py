import numpy as np
from concurrent.futures import ThreadPoolExecutor
from app.config import settings
from app.face_engine import FaceEngine, match_embeddings, stack_embeddings
from app.gridfs_storage import save_embedding, delete_embedding, load_embeddings


def register_student_face(
    face_engine: FaceEngine,
    school_id: str,
    student_id: str,
    image_bytes: bytes,
) -> dict[str, str]:
    """Store the embedding of the single face in the image, replacing any old one."""
    embeddings = face_engine.image_to_embeddings(image_bytes)

    if len(embeddings) != 1:
        raise ValueError(
            "Registration image must contain exactly one detectable face."
        )

    delete_embedding(
        school_id=school_id,
        student_id=student_id,
    )

    save_embedding(
        school_id=school_id,
        student_id=student_id,
        embedding=embeddings[0],
    )

    return {
        "status": "success",
        "school_id": school_id,
        "student_id": student_id,
        "storage": "gridfs",
    }


def verify_attendance_images(
    face_engine: FaceEngine,
    image_bytes_list: list[bytes],
    school_id: str,
    student_ids: list[str] | None = None,
) -> dict[str, object]:
    """Match every face in the images against the school's registered students.

    Each face counts for its best-scoring student when the cosine score reaches
    SIMILARITY_THRESHOLD; every known student is reported Present or Absent.
    """
    if not image_bytes_list:
        raise ValueError("At least one attendance image is required.")

    known_embeddings = load_embeddings(
        school_id=school_id,
        student_ids=student_ids,
    )

    if not known_embeddings:
        raise ValueError("No registered student embeddings were found.")

    known_ids = sorted(known_embeddings)
    known_matrix = stack_embeddings(
        known_embeddings[known_id] for known_id in known_ids
    )

    image_results = _extract_embeddings_in_parallel(
        face_engine,
        image_bytes_list,
    )

    face_origins: list[tuple[int, int]] = [
        (image_index, face_index)
        for image_index, detected_embeddings in enumerate(image_results)
        for face_index, _ in enumerate(detected_embeddings)
    ]

    present_ids: set[str] = set()
    matches: list[dict[str, object]] = []

    if face_origins:
        query_matrix = stack_embeddings(
            embedding
            for detected_embeddings in image_results
            for embedding in detected_embeddings
        )

        best_indices, best_scores = match_embeddings(query_matrix, known_matrix)

        for (image_index, face_index), best_index, best_score in zip(
            face_origins,
            best_indices,
            best_scores,
        ):
            if best_score < settings.similarity_threshold:
                continue

            best_student_id = known_ids[int(best_index)]
            present_ids.add(best_student_id)

            matches.append(
                {
                    "image_index": image_index,
                    "face_index": face_index,
                    "student_id": best_student_id,
                    "similarity": round(float(best_score), 4),
                }
            )

    students = [
        {
            "student_id": student_id,
            "status": "Present" if student_id in present_ids else "Absent",
        }
        for student_id in known_ids
    ]

    return {
        "status": "success",
        "school_id": school_id,
        "image_count": len(image_bytes_list),
        "detected_faces": len(face_origins),
        "images": [
            {
                "image_index": image_index,
                "detected_faces": len(embeddings),
            }
            for image_index, embeddings in enumerate(image_results)
        ],
        "matches": matches,
        "students": students,
    }


def _extract_embeddings_in_parallel(
    face_engine: FaceEngine,
    image_bytes_list: list[bytes],
) -> list[list[np.ndarray]]:
    """Extract embeddings for each image, using up to VERIFICATION_MAX_WORKERS threads."""
    max_workers = min(
        len(image_bytes_list),
        settings.verification_max_workers,
    )

    if max_workers <= 1:
        return [
            face_engine.image_to_embeddings(image_bytes)
            for image_bytes in image_bytes_list
        ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(
            executor.map(
                face_engine.image_to_embeddings,
                image_bytes_list,
            )
        )
