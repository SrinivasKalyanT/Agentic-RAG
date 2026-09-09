from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central place for the app's configuration flags.

    Values are read from environment variables / `.env` (matching the
    variable names already used elsewhere in the app, e.g. `GROQ_API_KEY`),
    so no existing `.env` values need to change.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Which ingestion strategy turns an uploaded PDF into indexable chunks.
    # "docling_caption": Docling for parsing (tables as structured Markdown)
    #   + a vision LLM to caption extracted images.
    # "raganything": not implemented yet.
    ingestion_strategy: Literal["docling_caption", "raganything"] = "docling_caption"

    groq_api_key: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
