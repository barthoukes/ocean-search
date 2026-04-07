"""
Document processor that manages loaders and vector database operations.
"""

import os
from typing import List, Optional, Tuple, Dict, Any
from langchain_chroma import Chroma
from langchain_core.documents import Document

from embedders import SmartEmbedder
from document_loaders import (
    TextFileLoader, MarkupFileLoader, PDFLoader, 
    ImageLoader, CodeLoader, DocumentLoader
)
from query_matcher import QueryMatcher


class EnhancedDocument(Document):
    """Extended Document class with additional helper methods."""
    
    def get_match_type_display(self) -> str:
        """Return a user-friendly display of the match source."""
        match_source = self.metadata.get('match_source', 'unknown')
        
        if match_source == 'file_content':
            return "✓ Match from file content"
        elif match_source == 'metadata_only':
            return "ℹ Match from metadata only"
        else:
            return "? Unknown match source"
    
    def get_file_type_display(self) -> str:
        """Return a formatted file type display."""
        file_type = self.metadata.get('type', 'unknown')
        extension = self.metadata.get('extension', '')
        
        type_icons = {
            'text': '📄',
            'markup': '📝',
            'pdf': '📑',
            'image': '🖼️',
            'code': '💻',
            'unknown': '❓'
        }
        
        icon = type_icons.get(file_type, '📁')
        embedder = self.metadata.get('embedding_type', 'ollama')
        embedder_icon = "🧠" if embedder == "bert" else "🔧"
        
        if file_type == 'code':
            language = self.metadata.get('language', '')
            return f"{icon} {file_type.upper()} ({language}) {embedder_icon}"
        elif file_type == 'image':
            dimensions = f"{self.metadata.get('width', '?')}x{self.metadata.get('height', '?')}"
            return f"{icon} {file_type.upper()} [{dimensions}] {embedder_icon}"
        elif file_type == 'pdf':
            pages = self.metadata.get('pages', '?')
            return f"{icon} {file_type.upper()} [{pages} pages] {embedder_icon}"
        else:
            return f"{icon} {file_type.upper()}{extension} {embedder_icon}"
    
    def is_empty(self) -> bool:
        """Check if the document has meaningful content."""
        # Check if content is empty or just whitespace
        if not self.page_content or not self.page_content.strip():
            return True
        
        # Check for placeholder content from empty PDFs/images
        empty_indicators = [
            "no extractable text content",
            "Image file:",
            "EXIF Data:"
        ]
        
        content_lower = self.page_content.lower()
        for indicator in empty_indicators:
            if indicator.lower() in content_lower:
                # If it's only metadata and no real content
                return True
        
        # Check if content is very short (less than 10 chars of real text)
        if len(self.page_content.strip()) < 10:
            return True
        
        return False


def test_enhanced_document():
    pass


if __name__ == "__main__":
    # Run test when script is executed directly
    test_enhanced_document()
