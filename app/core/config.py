from functools import lru_cache
from typing import Literal

from pydantic_settings import (
    BaseSettings,
    JsonConfigSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class Settings(BaseSettings):
    """Central place for the app's configuration flags.

    Values are read from `config.json` (if present), then environment
    variables / `.env` (matching the variable names already used elsewhere
    in the app, e.g. `GROQ_API_KEY`), so no existing `.env` values need to
    change. A real env var always wins over `config.json` over `.env`, so
    e.g. a deploy-time secret can still override a checked-in config.json.
    """

    model_config = SettingsConfigDict(
        env_file=".env", json_file="config.json", extra="ignore"
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            JsonConfigSettingsSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )

    # Which ingestion strategy turns an uploaded PDF into indexable chunks.
    # "docling_caption": Docling for parsing (tables as structured Markdown)
    #   + a vision LLM to caption extracted images.
    # "raganything": not implemented yet.
    ingestion_strategy: Literal["docling_caption", "raganything"] = "docling_caption"

    groq_api_key: str

    # Hybrid (dense + sparse) retrieval — see app/retrieval/faiss_store.py.
    # Fuses FAISS similarity search with a BM25 retriever over the same
    # knowledge base via weighted Reciprocal Rank Fusion. Disable to fall
    # back to the old dense-only behavior; weights need not sum to 1 (only
    # their ratio matters to the fusion).
    hybrid_search_enabled: bool = True
    hybrid_dense_weight: float = 0.5
    hybrid_sparse_weight: float = 0.5


@lru_cache
def get_settings() -> Settings:
    return Settings()
