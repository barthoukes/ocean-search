#!/usr/bin/env python3
"""
Base class for all document loaders.

This module defines the abstract DocumentLoader class that all concrete
loaders must inherit from. It provides common functionality like extension
handling and document ID generation.
"""

import os
import hashlib
from abc import ABC, abstractmethod
from typing import Optional, Set, List
from langchain_core.documents import Document


class DocumentLoader(ABC):
    """Abstract base class for loading and processing documents."""

    def __init__(self, extensions: Optional[Set[str]] = None):
        self.extensions = extensions if extensions else set()
        self.supported_extensions = self._get_supported_extensions()

    @abstractmethod
    def _get_supported_extensions(self) -> Set[str]:
        pass

    @abstractmethod
    def load_document(self, file_path: str) -> Optional[Document]:
        pass

    def can_handle(self, file_path: str) -> bool:
        ext = os.path.splitext(file_path)[1].lower()
        if self.extensions and ext not in self.extensions:
            return False
        return ext in self.supported_extensions

    def get_document_id(self, file_path: str) -> str:
        return hashlib.md5(file_path.encode()).hexdigest()


if __name__ == "__main__":
    # Simple self-test
    class DummyLoader(DocumentLoader):
        def _get_supported_extensions(self) -> Set[str]:
            return {'.dummy'}
        def load_document(self, file_path: str) -> Optional[Document]:
            return Document(page_content="dummy", metadata={})

    loader = DummyLoader()
    print("Base class test:")
    print(f"  can_handle('test.dummy'): {loader.can_handle('test.dummy')}")
    print(f"  can_handle('test.txt'): {loader.can_handle('test.txt')}")
    print(f"  document ID: {loader.get_document_id('test.txt')}")

