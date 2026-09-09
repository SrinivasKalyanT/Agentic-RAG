from abc import ABC, abstractmethod
from pathlib import Path

from langchain_core.documents import Document


class IngestionStrategy(ABC):
    """Common interface every ingestion strategy must implement.

    A strategy turns a single PDF file into a list of ready-to-index
    LangChain ``Document`` chunks. Whatever it does internally - which
    parser it uses, how it handles tables/images, how it chunks - the rest
    of the pipeline never needs to know which strategy produced the chunks.
    """

    @abstractmethod
    def load_and_chunk(self, pdf_path: Path) -> list[Document]:
        """Parse `pdf_path` and return ready-to-index chunks."""
        raise NotImplementedError
