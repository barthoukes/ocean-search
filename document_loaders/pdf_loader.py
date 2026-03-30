#!/usr/bin/env python3
"""
Loader for PDF documents.

This loader extracts text from PDF files using PyPDF2 or pypdf. If no text
can be extracted, it returns a placeholder message and marks the metadata
accordingly. It requires the 'pypdf' or 'PyPDF2' library to be installed.
"""

import os
from typing import Optional, Set
from langchain_core.documents import Document
from .base import DocumentLoader


class PDFLoader(DocumentLoader):
    """Loader for PDF files."""

    def _get_supported_extensions(self) -> Set[str]:
        return {'.pdf'}

    def load_document(self, file_path: str) -> Optional[Document]:
        try:
            # Try to import pypdf first (preferred), then PyPDF2
            try:
                from pypdf import PdfReader
            except ImportError:
                try:
                    from PyPDF2 import PdfReader
                except ImportError:
                    print("Please install PyPDF2 or pypdf: pip install pypdf")
                    return None

            reader = PdfReader(file_path)
            content = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    content += page_text + "\n"

            if not content.strip():
                content = f"[PDF document with {len(reader.pages)} pages - no extractable text content]"
                match_source = "metadata_only"
            else:
                match_source = "file_content"

            metadata = {
                "filename": os.path.basename(file_path),
                "filepath": file_path,
                "type": "pdf",
                "pages": len(reader.pages),
                "size": os.path.getsize(file_path),
                "extension": ".pdf",
                "match_source": match_source,
                "has_text_content": bool(content.strip() and
                                         content.strip() != f"[PDF document with {len(reader.pages)} pages - no extractable text content]"),
                "embedding_type": "ollama"
            }

            return Document(page_content=content, metadata=metadata)
        except Exception as e:
            print(f"Error loading PDF {file_path}: {e}")
            return None


if __name__ == "__main__":
    # Self-test: check if required library is installed and try to load a PDF
    try:
        from pypdf import PdfReader
        pdf_available = True
    except ImportError:
        try:
            from PyPDF2 import PdfReader
            pdf_available = True
        except ImportError:
            pdf_available = False

    if not pdf_available:
        print("PDFLoader self-test skipped: pypdf/PyPDF2 not installed")
        print("Install with: pip install pypdf")
    else:
        print("PDFLoader self-test:")
        loader = PDFLoader()
        # Test can_handle
        print(f"  can_handle('test.pdf'): {loader.can_handle('test.pdf')}")
        print(f"  can_handle('test.txt'): {loader.can_handle('test.txt')}")

        # Try to load a real PDF – for a complete test, provide a path to a real PDF file
        # Here we simulate by trying to load a non-existent file; should return None.
        test_file = "test_does_not_exist.pdf"
        doc = loader.load_document(test_file)
        if doc is None:
            print("  Load on non-existent file returned None (as expected)")
        else:
            print("  WARNING: load on non-existent file returned a Document – unexpected")

        print("\n  For a full test, run with a real PDF file, e.g.:")
        print("    loader = PDFLoader()")
        print("    doc = loader.load_document('example.pdf')")

