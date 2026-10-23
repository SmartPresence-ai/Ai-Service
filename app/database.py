"""MongoDB client and the GridFS bucket that stores face embeddings.

The unique index on (school_id, student_id) guarantees one embedding per
student per school.
"""

from pymongo import ASCENDING, MongoClient
from gridfs import GridFSBucket

from app.config import settings

client = MongoClient(settings.mongodb_uri)
db = client[settings.mongodb_database]

bucket = GridFSBucket(
    db,
    bucket_name=settings.gridfs_bucket,
)

files_collection = db[f"{settings.gridfs_bucket}.files"]

files_collection.create_index(
    [
        ("metadata.school_id", ASCENDING),
        ("metadata.student_id", ASCENDING),
    ],
    unique=True,
)
