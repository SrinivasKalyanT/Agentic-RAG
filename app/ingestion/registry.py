import hashlib
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

REGISTRY_ROOT = Path("data/metadata")


class DocumentRegistry:

    def __init__(self, knowledge_base: str):
        self.knowledge_base = knowledge_base

        self.registry_dir = REGISTRY_ROOT / knowledge_base
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        self.registry_file = self.registry_dir / "documents.json"

        if not self.registry_file.exists():
            self.save({})

    def load(self):

        with open(self.registry_file, "r") as f:
            return json.load(f)

    def save(self, data):

        with open(self.registry_file, "w") as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def calculate_hash(file_path: Path):

        sha256 = hashlib.sha256()

        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)

        return sha256.hexdigest()

    @staticmethod
    def generate_document_id():

        return f"doc_{uuid4().hex[:12]}"

    def get_changed_documents(self, upload_path: Path):

        registry = self.load()

        new_files = []
        modified_files = []

        pdf_files = list(upload_path.glob("*.pdf"))

        for pdf_file in pdf_files:

            current_hash = self.calculate_hash(pdf_file)

            if pdf_file.name not in registry:

                new_files.append((pdf_file, current_hash))

            elif registry[pdf_file.name]["hash"] != current_hash:

                modified_files.append((pdf_file, current_hash))

        return new_files, modified_files

    def register_document(self, filename: str, file_hash: str, document_id: str):

        registry = self.load()

        registry[filename] = {
            "document_id": document_id,
            "hash": file_hash,
            "indexed_at": datetime.utcnow().isoformat(),
        }

        self.save(registry)

    def get_document_id(self, filename: str):

        registry = self.load()

        if filename not in registry:
            return None

        return registry[filename]["document_id"]
