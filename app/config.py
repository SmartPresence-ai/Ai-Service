from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Service settings read from the environment or a local .env file.

    AI_SERVICE_API_KEY and CORS_ALLOWED_ORIGINS are optional; every other
    setting is required so a misconfigured deployment fails at startup.
    """

    mongodb_uri: str
    mongodb_database: str
    gridfs_bucket: str
    similarity_threshold: float
    verification_max_workers: int
    device: str
    insightface_model_name: str
    detection_width: int
    detection_height: int
    detection_threshold: float
    insightface_allowed_modules: list[str]
    ai_service_api_key: str = ""
    cors_allowed_origins: list[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @computed_field
    @property
    def insightface_ctx_id(self) -> int:
        return -1 if self.device.lower() == "cpu" else 0

    @computed_field
    @property
    def insightface_providers(self) -> list[str]:
        """ONNX Runtime providers, with CPU kept as the fallback for CUDA."""
        if self.device.lower() == "cpu":
            return ["CPUExecutionProvider"]
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    @computed_field
    @property
    def detection_size(self) -> tuple[int, int]:
        return (self.detection_width, self.detection_height)


settings = Settings()
