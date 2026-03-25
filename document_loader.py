"""
Document loaders for various file types.
"""

import os
import hashlib
from abc import ABC, abstractmethod
from typing import Optional, Set, List
from langchain_core.documents import Document


class DocumentLoader(ABC):
    """Abstract base class for loading and processing documents."""
    
    def __init__(self, extensions: Optional[Set[str]] = None):
        """
        Initialize the document loader.
        
        Args:
            extensions: Set of file extensions to process (e.g., {'.txt', '.pdf'})
        """
        self.extensions = extensions if extensions else set()
        self.supported_extensions = self._get_supported_extensions()
    
    @abstractmethod
    def _get_supported_extensions(self) -> Set[str]:
        """Return the set of file extensions this loader supports."""
        pass
    
    @abstractmethod
    def load_document(self, file_path: str) -> Optional[Document]:
        """Load a single document from a file path."""
        pass
    
    def can_handle(self, file_path: str) -> bool:
        """Check if this loader can handle the given file."""
        ext = os.path.splitext(file_path)[1].lower()
        if self.extensions and ext not in self.extensions:
            return False
        return ext in self.supported_extensions
    
    def get_document_id(self, file_path: str) -> str:
        """Generate a unique ID for the document."""
        return hashlib.md5(file_path.encode()).hexdigest()


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


class PDFLoader(DocumentLoader):
    """Loader for PDF files."""
    
    def _get_supported_extensions(self) -> Set[str]:
        return {'.pdf'}
    
    def load_document(self, file_path: str) -> Optional[Document]:
        try:
            # Try to import PyPDF2 or pypdf
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
                "has_text_content": bool(content.strip() and content.strip() != f"[PDF document with {len(reader.pages)} pages - no extractable text content]"),
                "embedding_type": "ollama"
            }
            
            return Document(page_content=content, metadata=metadata)
        except Exception as e:
            print(f"Error loading PDF {file_path}: {e}")
            return None


class ImageLoader(DocumentLoader):
    """Loader for image files (using metadata and OCR)."""
    
    def _get_supported_extensions(self) -> Set[str]:
        return {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    
    def load_document(self, file_path: str) -> Optional[Document]:
        try:
            from PIL import Image
            
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
        except ImportError:
            print("Please install Pillow: pip install Pillow")
            return None
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            return None


class CodeLoader(DocumentLoader):
    """Loader for code files - uses Ollama."""
    
    def _get_supported_extensions(self) -> Set[str]:
        return {'.py', '.cpp', '.c', '.h', '.hpp', '.js', '.ts', '.java', 
                '.go', '.rs', '.rb', '.php', '.sh', '.bash', '.cc', '.c', '.kt', '.lua' }
    
    def load_document(self, file_path: str) -> Optional[Document]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Count lines of code
            lines = content.split('\n')
            code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#') and not l.strip().startswith('//')]
            comment_lines = [l for l in lines if l.strip().startswith('#') or l.strip().startswith('//')]
            
            metadata = {
                "filename": os.path.basename(file_path),
                "filepath": file_path,
                "type": "code",
                "language": os.path.splitext(file_path)[1][1:],
                "lines_total": len(lines),
                "lines_code": len(code_lines),
                "lines_comments": len(comment_lines),
                "size": os.path.getsize(file_path),
                "extension": os.path.splitext(file_path)[1].lower(),
                "match_source": "file_content",
                "embedding_type": "ollama"
            }
            
            return Document(page_content=content, metadata=metadata)
        except Exception as e:
            print(f"Error loading code file {file_path}: {e}")
            return None

