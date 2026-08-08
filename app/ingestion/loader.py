from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


def load_single_pdf(pdf_path: Path):

    loader = PyPDFLoader(str(pdf_path))

    return loader.load()
