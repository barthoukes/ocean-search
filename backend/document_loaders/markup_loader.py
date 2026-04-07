#!/usr/bin/env python3
"""
Loader for markup and structured files (HTML, CSS, JSON, XML, YAML, etc.).

This loader reads the file as plain text and stores the raw content.
It is intended for files that contain structured data or markup,
and will be processed with Ollama (as indicated in metadata).
"""

import os
from typing import Optional, Set
from langchain_core.documents import Document
from .base import DocumentLoader


class MarkupFileLoader(DocumentLoader):
    """Loader for markup files (HTML, XML, JSON, etc.) - uses Ollama."""

    def _get_supported_extensions(self) -> Set[str]:
        return {'.html', '.css', '.json', '.xml', '.yaml', '.yml'}

    def load_document(self, file_path: str) -> Optional[Document]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            metadata = {
                "filename": os.path.basename(file_path),
                "filepath": file_path,
                "type": "markup",
                "size": os.path.getsize(file_path),
                "extension": os.path.splitext(file_path)[1].lower(),
                "match_source": "file_content",
                "embedding_type": "ollama"
            }

            return Document(page_content=content, metadata=metadata)
        except Exception as e:
            print(f"Error loading markup file {file_path}: {e}")
            return None


if __name__ == "__main__":
    # Self-test: create a temporary JSON file and load it
    import tempfile
    import json

    # Sample JSON content
    sample_data = {
        "name": "test",
        "type": "markup",
        "value": 42
    }
    json_str = json.dumps(sample_data, indent=2)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        f.write(json_str)
        temp_path = f.name

    try:
        loader = MarkupFileLoader()

        # Test can_handle
        print("MarkupFileLoader self-test:")
        print(f"  can_handle('test.json'): {loader.can_handle('test.json')}")
        print(f"  can_handle('test.html'): {loader.can_handle('test.html')}")
        print(f"  can_handle('test.txt'): {loader.can_handle('test.txt')}")

        # Load the temporary file
        doc = loader.load_document(temp_path)
        if doc:
            print("  Load on temporary JSON file SUCCESS")
            print(f"    Content length: {len(doc.page_content)} characters")
            print(f"    Metadata: {doc.metadata}")
            # Optionally verify content matches
            if doc.page_content.strip() == json_str.strip():
                print("    Content matches expected JSON")
            else:
                print("    WARNING: content does not match")
        else:
            print("  Load on temporary JSON file FAILED")
    finally:
        os.unlink(temp_path)

    # Test with non-existent file (should return None)
    doc_none = MarkupFileLoader().load_document("non_existent.json")
    if doc_none is None:
        print("  Load on non-existent file returned None (as expected)")
    else:
        print("  WARNING: load on non-existent file returned a Document – unexpected")

