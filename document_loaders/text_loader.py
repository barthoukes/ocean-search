#!/usr/bin/env python3
"""
Loader for plain text and markup-like text files (TXT, MD, RST, etc.).

This loader reads text files using UTF-8 encoding and returns a Document
with metadata. It uses BERT for embedding (indicated in metadata).
"""

import os
from typing import Optional, Set
from langchain_core.documents import Document
from .base import DocumentLoader


class TextFileLoader(DocumentLoader):
    """Loader for natural language text files (uses BERT)."""

    def _get_supported_extensions(self) -> Set[str]:
        return {'.txt', '.md', '.rst', '.text'}

    def load_document(self, file_path: str) -> Optional[Document]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            metadata = {
                "filename": os.path.basename(file_path),
                "filepath": file_path,
                "type": "text",
                "size": os.path.getsize(file_path),
                "extension": os.path.splitext(file_path)[1].lower(),
                "match_source": "file_content",
                "embedding_type": "bert"
            }
            return Document(page_content=content, metadata=metadata)
        except Exception as e:
            print(f"Error loading text file {file_path}: {e}")
            return None


if __name__ == "__main__":
    # Self-test: create a temporary file and load it
    import tempfile

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test content for the text loader.")
        temp_path = f.name

    try:
        loader = TextFileLoader()
        doc = loader.load_document(temp_path)
        if doc:
            print("TextFileLoader test PASSED")
            print(f"  Page content: {doc.page_content}")
            print(f"  Metadata: {doc.metadata}")
        else:
            print("TextFileLoader test FAILED")
    finally:
        os.unlink(temp_path)

