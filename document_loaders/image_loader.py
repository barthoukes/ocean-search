#!/usr/bin/env python3
"""
Loader for image files (JPG, PNG, GIF, etc.).

This loader extracts basic metadata (dimensions, format, EXIF) from images
using Pillow. It does not perform OCR; content is limited to metadata.
Requires the Pillow library.
"""

import os
from typing import Optional, Set
from langchain_core.documents import Document
from .base import DocumentLoader


class ImageLoader(DocumentLoader):
    """Loader for image files (using metadata and OCR)."""

    def _get_supported_extensions(self) -> Set[str]:
        return {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}

    def load_document(self, file_path: str) -> Optional[Document]:
        try:
            from PIL import Image
        except ImportError:
            print("Please install Pillow: pip install Pillow")
            return None

        try:
            img = Image.open(file_path)

            # Extract metadata
            content = f"Image file: {os.path.basename(file_path)}\n"
            content += f"Dimensions: {img.width}x{img.height}\n"
            content += f"Format: {img.format}\n"
            content += f"Mode: {img.mode}\n"

            # Try to extract EXIF data
            exif_data = img._getexif()
            if exif_data:
                content += "\nEXIF Data:\n"
                for tag_id, value in exif_data.items():
                    content += f"  {tag_id}: {value}\n"

            metadata = {
                "filename": os.path.basename(file_path),
                "filepath": file_path,
                "type": "image",
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "size": os.path.getsize(file_path),
                "extension": os.path.splitext(file_path)[1].lower(),
                "match_source": "metadata_only",
                "embedding_type": "ollama"
            }

            return Document(page_content=content, metadata=metadata)
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            return None


if __name__ == "__main__":
    # Self-test: check if Pillow is installed
    try:
        from PIL import Image
        pillow_available = True
    except ImportError:
        pillow_available = False

    if not pillow_available:
        print("ImageLoader self-test skipped: Pillow not installed")
        print("Install with: pip install Pillow")
    else:
        print("ImageLoader self-test:")
        loader = ImageLoader()

        # Test can_handle
        print(f"  can_handle('test.jpg'): {loader.can_handle('test.jpg')}")
        print(f"  can_handle('test.png'): {loader.can_handle('test.png')}")
        print(f"  can_handle('test.txt'): {loader.can_handle('test.txt')}")

        # Create a tiny dummy image and load it
        import tempfile
        from io import BytesIO

        # Create a 1x1 black PNG in memory
        img = Image.new('RGB', (1, 1), color='black')
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img.save(tmp.name, format='PNG')
            tmp_path = tmp.name

        try:
            doc = loader.load_document(tmp_path)
            if doc:
                print("  Load on dummy image SUCCESS")
                print(f"    Content snippet: {doc.page_content[:100]}")
                print(f"    Metadata: {doc.metadata}")
            else:
                print("  Load on dummy image FAILED")
        finally:
            os.unlink(tmp_path)

        # Test with non-existent file (should return None)
        doc_none = loader.load_document("non_existent.png")
        if doc_none is None:
            print("  Load on non-existent file returned None (as expected)")
        else:
            print("  WARNING: load on non-existent file returned a Document – unexpected")

