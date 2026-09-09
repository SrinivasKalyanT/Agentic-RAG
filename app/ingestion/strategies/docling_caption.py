"""Docling-based ingestion strategy with local image captioning.

Docling parses the PDF with real layout understanding: tables come out as
structured tables (exported as clean Markdown instead of being flattened
into run-together text), and figures/pictures are captioned by a small
vision-language model so their content becomes searchable text instead of
being silently dropped. Chunking is done with docling-core's
`HybridChunker`, which splits along document structure (sections, tables,
pictures) instead of blindly by character count, so a table or a caption
never gets cut in half across two chunks.

Captioning runs locally (SmolVLM, via Docling's default
`picture_description_options`, downloaded from HuggingFace on first use) -
no external API key required, unlike the Groq LLM used elsewhere in this
app, which has no vision-capable model available.
"""

import re
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_core.documents import Document
from langchain_docling.loader import DoclingLoader, ExportType

from app.ingestion.strategies.base import IngestionStrategy

_PICTURE_LABEL = "picture"
_TABLE_LABEL = "table"

# Small local vision-language models (e.g. the default SmolVLM preset) can
# echo their own chat-template stop sequence as literal text instead of it
# being stripped as a real special token - and generation can stop mid-token,
# so what lands in the caption is often a truncated fragment like
# "...a graph.<end_of_utteranc" rather than the full "<end_of_utterance>".
# Match it structurally (an unterminated/terminated "<...>"-shaped fragment
# at the very end of the text) rather than by exact spelling, since the
# fragment's ending is unpredictable.
_TRAILING_TAG_ARTIFACT_RE = re.compile(r"<\|?/?[A-Za-z_]{0,32}\|?>?\s*$")


def _clean_caption_artifacts(text: str) -> str:
    return _TRAILING_TAG_ARTIFACT_RE.sub("", text).rstrip()


def _build_converter() -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    pipeline_options.do_picture_description = True
    # picture_description_options is left at Docling's default (the
    # "smolvlm" preset), which runs locally via HuggingFace Transformers.

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def _content_type(chunk_metadata: dict) -> str:
    """Classify a chunk as "table", "image" or "text" from Docling's metadata."""
    doc_items = chunk_metadata.get("dl_meta", {}).get("doc_items", [])
    labels = {item.get("label") for item in doc_items}
    if _TABLE_LABEL in labels:
        return "table"
    if _PICTURE_LABEL in labels:
        return "image"
    return "text"


def _page_number(chunk_metadata: dict) -> int:
    """Read the (1-indexed) source page from Docling's per-item provenance.

    The rest of the app (`app/retrieval/faiss_store.py`) reads
    `doc.metadata["page"]` directly - it was populated by the previous
    PyPDFLoader-based loader, so this strategy has to keep setting it too.
    """
    doc_items = chunk_metadata.get("dl_meta", {}).get("doc_items", [])
    for item in doc_items:
        for prov in item.get("prov", []):
            page_no = prov.get("page_no")
            if page_no is not None:
                return page_no
    return 1


class DoclingCaptionStrategy(IngestionStrategy):
    """Parses PDFs with Docling and captions images with a vision LLM."""

    def load_and_chunk(self, pdf_path: Path) -> list[Document]:
        loader = DoclingLoader(
            file_path=str(pdf_path),
            converter=_build_converter(),
            export_type=ExportType.DOC_CHUNKS,
        )
        chunks = loader.load()
        for chunk in chunks:
            chunk.page_content = _clean_caption_artifacts(chunk.page_content)
            chunk.metadata["content_type"] = _content_type(chunk.metadata)
            chunk.metadata["page"] = _page_number(chunk.metadata)
        return chunks
