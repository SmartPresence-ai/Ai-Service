# AI Service

Independent FastAPI service for face detection, embedding generation, embedding storage, and attendance prediction. Embeddings are stored in MongoDB GridFS, keyed by `school_id` + `student_id`.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

The first run downloads the InsightFace `buffalo_l` model pack, so startup is slow once. The model is loaded and warmed up during app startup; requests sent before that finishes are queued, not slow.

## Configuration

All settings are required and read from `.env` (see `.env.example`):

| Variable | Purpose |
| --- | --- |
| `MONGODB_URI`, `MONGODB_DATABASE`, `GRIDFS_BUCKET` | Embedding storage |
| `SIMILARITY_THRESHOLD` | Cosine score above which a face counts as a match (default `0.45`) |
| `VERIFICATION_MAX_WORKERS` | How many attendance photos are processed in parallel |
| `DEVICE` | `cpu`, or any other value to request CUDA execution providers |
| `INSIGHTFACE_MODEL_NAME`, `INSIGHTFACE_ALLOWED_MODULES` | Model pack and enabled modules |
| `DETECTION_WIDTH`, `DETECTION_HEIGHT`, `DETECTION_THRESHOLD` | Detector input size and confidence floor |
| `AI_SERVICE_API_KEY` | Shared secret; when set, every endpoint except `/health` needs the `X-AI-Service-Key` header (the Django backend sends it) |
| `CORS_ALLOWED_ORIGINS` | JSON list of browser origins allowed to call the service (default `["*"]`) |

`requirements.txt` pins `onnxruntime`, the CPU-only build. Setting `DEVICE` to a non-CPU value has no effect with that wheel — ONNX Runtime drops the unavailable `CUDAExecutionProvider` and silently runs on CPU. Real GPU inference needs `onnxruntime-gpu` plus a matching CUDA 12 / cuDNN 9 runtime.

## Docker

```bash
docker build -f DockerFile -t smartpresence-ai .
docker run --env-file .env -p 8001:8001 smartpresence-ai
```

The image downloads the face model at build time, so containers start without a download. It listens on `$PORT` (default 8001); `WEB_CONCURRENCY` sets the worker count (each worker loads its own model, so keep it at 1-2). See `.env.production.example`.

## APIs

- `GET /health` — returns service status and whether the model is loaded.
- `POST /register-face` — form fields `school_id`, `student_id` and file `image`. The image must contain exactly one detectable face. Replaces any existing embedding for that student.
- `POST /verify-attendance` — form field `school_id`, one or more `images` files, and optional `student_ids` (comma-separated) to restrict the roster compared against. Photos are processed in parallel; returns per-image face counts, matches, and each student's `Present`/`Absent` status.
- `DELETE /delete-face` — form fields `school_id`, `student_id`. Returns 404 if no embedding exists.

The model adapter is isolated in `app/face_engine.py` so InsightFace Buffalo_L can later be replaced by another implementation without changing Django or React.

## Test frontend

`ai_check_fe/` is a small Vite-served static frontend for exercising the three endpoints by hand. Set `VITE_AI_SERVICE_URL` to the service URL, then:

```bash
npm install
npm run dev
```
