from app.core.config import get_settings
from app.ingestion.strategies.base import IngestionStrategy


def get_strategy(name: str | None = None) -> IngestionStrategy:
    """Return the ingestion strategy selected by config (or `name` if given)."""
    strategy_name = name or get_settings().ingestion_strategy

    if strategy_name == "docling_caption":
        from app.ingestion.strategies.docling_caption import DoclingCaptionStrategy

        return DoclingCaptionStrategy()

    if strategy_name == "raganything":
        raise NotImplementedError(
            "The 'raganything' ingestion strategy is not implemented yet. "
            "Set INGESTION_STRATEGY=docling_caption in .env for now."
        )

    raise ValueError(f"Unknown ingestion strategy: {strategy_name!r}")
