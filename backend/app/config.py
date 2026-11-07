"""Application settings, read from environment variables and an optional ``backend/.env`` file.

Every tunable the system uses lives here so that tests, the API, workers and the evaluation runner agree.
"""

from __future__ import annotations

import os
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent

INSECURE_DEFAULT_KEYS = frozenset(
    {"change-me", "dev-local-key-change-me", "changeme", "test", "replace-with-a-long-random-key-0000"}
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # repo-root .env (shared with docker compose), then backend/.env; real environment variables win over both
        env_file=(BACKEND_DIR.parent / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- runtime ---------------------------------------------------------------------------------------------
    app_env: Literal["development", "test", "production"] = "development"
    app_name: str = "DocIntel"
    log_level: str = "INFO"
    log_json: bool = False

    # --- storage, database, queue ----------------------------------------------------------------------------
    database_url: str = f"sqlite:///{(BACKEND_DIR / 'data' / 'app.db').as_posix()}"
    db_pool_size: int = 10
    db_max_overflow: int = 20
    redis_url: str | None = None
    queue_backend: Literal["inline", "local", "celery"] = "local"
    local_queue_workers: int = 2
    storage_dir: Path = BACKEND_DIR / "data" / "storage"

    # --- upload and parser limits ----------------------------------------------------------------------------
    max_upload_mb: int = 200
    max_files_per_request: int = 50
    max_pdf_pages: int = 2000
    max_image_pixels: int = 120_000_000
    max_sheet_rows: int = 200_000
    max_sheet_cols: int = 500
    max_xlsx_uncompressed_mb: int = 500
    max_xlsx_compression_ratio: int = 200

    # --- OCR -------------------------------------------------------------------------------------------------
    ocr_engine: Literal["rapidocr", "tesseract"] = "rapidocr"
    ocr_dpi: int = 200
    ocr_workers: int = 0  # 0 = OCR inline in the calling process; N = process pool of N workers
    ocr_min_text_chars: int = 25  # a PDF page with fewer alphanumeric characters is OCR'd
    ocr_low_confidence: float = 0.75  # mean OCR confidence below this raises OCR_LOW_QUALITY
    ocr_page_timeout_s: float = 120.0  # one page/frame of OCR taking longer than this is treated as an OCR outage

    # --- extraction ------------------------------------------------------------------------------------------
    extraction_mode: Literal["rules", "llm", "hybrid"] = "hybrid"
    default_dollar_currency: str = "USD"  # what a bare "$" means when nothing else says
    date_order: Literal["auto", "DMY", "MDY"] = "auto"

    # --- LLM providers ---------------------------------------------------------------------------------------
    llm_provider: Literal["auto", "anthropic", "gemini", "local", "none"] = "auto"
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-opus-5-5"
    anthropic_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3-flash-preview"
    llm_base_url: str | None = None  # OpenAI-compatible endpoint, e.g. llama.cpp at http://localhost:8080/v1
    llm_model: str = "qwen3.5-9b"
    llm_api_key: SecretStr | None = None
    llm_timeout_s: float = 180.0
    llm_max_input_chars: int = 60_000  # upper bound; the effective budget is also capped by the model's context
    # Context window (tokens) per request for the local/OpenAI-compatible model. None = ask the server (llama.cpp
    # /props n_ctx / total_slots) and fall back to llm_max_input_chars when the server does not say.
    llm_context_tokens: int | None = None
    # Extra JSON merged into OpenAI-compatible request bodies,
    # e.g. '{"chat_template_kwargs":{"enable_thinking":false}}'.
    # None = send llama.cpp's thinking switch only when the server identifies as llama.cpp.
    llm_extra_body: str | None = None
    llm_concurrency: int = 2
    llm_max_output_tokens: int = 4096

    # --- embeddings and search -------------------------------------------------------------------------------
    embedding_provider: Literal["fastembed", "hashing"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    embedding_cache_dir: Path | None = None
    chunk_max_chars: int = 1200
    chunk_overlap: int = 150

    # --- validation and review -------------------------------------------------------------------------------
    amount_tolerance: Decimal = Decimal("0.02")
    review_confidence_threshold: float = 0.6
    max_reasonable_amount: Decimal = Decimal("1000000000")
    tax_rate_max_percent: Decimal = Decimal("30")

    # --- jobs ------------------------------------------------------------------------------------------------
    job_max_attempts: int = 3
    job_stale_after_s: int = 300
    job_max_runtime_s: int = 4 * 3600  # an attempt running longer stops heartbeating, so the sweeper reclaims/fails it

    # --- security --------------------------------------------------------------------------------------------
    api_keys: str = ""  # comma-separated; empty disables auth (refused when app_env=production)
    cors_origins: str = "http://localhost:3000"
    rate_limit_enabled: bool = True
    rate_limit_upload: str = "60/minute"
    rate_limit_search: str = "300/minute"
    rate_limit_default: str = "1200/minute"

    # --- derived ---------------------------------------------------------------------------------------------
    @property
    def api_key_list(self) -> list[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    def production_problems(self) -> list[str]:
        """Configuration that is unacceptable in production (empty list = OK)."""
        problems: list[str] = []
        if self.app_env != "production":
            return problems
        keys = self.api_key_list
        if not keys:
            problems.append("API_KEYS must be set in production")
        if any(k.lower() in INSECURE_DEFAULT_KEYS or len(k) < 24 for k in keys):
            problems.append("API_KEYS contains a default or short (< 24 chars) key")
        if self.is_sqlite:
            problems.append("DATABASE_URL must point to PostgreSQL in production")
        if "*" in self.cors_origin_list:
            problems.append("CORS_ORIGINS must not be '*' in production")
        return problems


@lru_cache
def get_settings() -> Settings:
    # Tests set DOCINTEL_IGNORE_DOTENV=1 so a developer's .env (API keys, Postgres URL) never leaks into them.
    if os.environ.get("DOCINTEL_IGNORE_DOTENV") == "1":
        return Settings(_env_file=None)  # type: ignore[call-arg]
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
