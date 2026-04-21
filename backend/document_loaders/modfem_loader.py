#!/usr/bin/env python3
"""
Loader for Modfem files (.modfem).

This loader extracts basic metadata (dimensions, format, EXIF) from modfem
"""

import os
from typing import Optional, Set
from langchain_core.documents import Document
from .base import DocumentLoader

class ModfemLoader(DocumentLoader):
    def _get_supported_extensions(self) -> Set[str]:
        return {'.modfem'}
    
    def load_document(self, file_path: str) -> Optional[Document]:
        # Extract only filesystem metadata
        stat = os.stat(file_path)
        metadata = {
            "filename": os.path.basename(file_path),
            "filepath": file_path,
            "type": "femap_model",
            "extension": ".modfem",
            "size": stat.st_size,
            "modified_time": stat.st_mtime,
            "match_source": "metadata_only",
            "embedding_type": "ollama",
            "format": "binary",
            "searchable": 0  # Not text-searchable
        }
        content = f"Femap 3D model file: {os.path.basename(file_path)}\n"
        content += f"Size: {stat.st_size} bytes\n"
        content += "Note: Binary format - only metadata is searchable"
        return Document(page_content=content, metadata=metadata)