#!/usr/bin/env python3
"""
file_search.py
Interactive tool to fill and query a vector database of files.
Commands:
  fill <path>   - add files from <path> to the database
  <query>       - search for files matching the query
  q             - quit
"""

import os
import argparse
import sys
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from pathlib import Path
import hashlib
import json
import numpy as np

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Optional BERT import - will gracefully fallback if not available
try:
    from sentence_transformers import SentenceTransformer
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("⚠️ sentence-transformers not installed. BERT support disabled.")
    print("   To enable BERT for text files: pip install sentence-transformers")


class BERTEmbedder:
    """BERT-based embedder specifically for natural language text files."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        """
        Initialize BERT embedder.
        
        Args:
            model_name: BERT model to use (default: all-MiniLM-L6-v2)
            device: 'cpu' or 'cuda' for GPU acceleration
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.is_available = False
        
        if BERT_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Load the BERT model with error handling."""
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.is_available = True
            print(f"✅ Loaded BERT model: {self.model_name} for natural language text files")
        except Exception as e:
            print(f"⚠️ Failed to load BERT model: {e}")
            print("   Text files will use Ollama embeddings instead")
            self.model = None
    
    def embed_documents(self, texts: List[str]) -> Optional[List[List[float]]]:
        """
        Generate embeddings for multiple documents.
        
        Args:
            texts: List of document texts
            
        Returns:
            List of embedding vectors or None if failed
        """
        if not self.is_available or self.model is None:
            return None
        
        try:
            # Truncate very long texts to avoid memory issues (BERT has 512 token limit)
            truncated_texts = [text[:100000] for text in texts]  # Roughly 25k tokens
            embeddings = self.model.encode(truncated_texts, convert_to_numpy=True)
            return [emb.tolist() for emb in embeddings]
            
        except Exception as e:
            print(f"Error generating BERT embeddings: {e}")
            return None
    
    def embed_query(self, query: str) -> Optional[List[float]]:
        """
        Embed a query string.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector or None if failed
        """
        if not self.is_available or self.model is None:
            return None
        
        try:
            embedding = self.model.encode([query], convert_to_numpy=True)[0]
            return embedding.tolist()
        except Exception as e:
            print(f"Error generating BERT query embedding: {e}")
            return None


class SmartEmbedder:
    """
    Smart embedder that uses BERT only for natural language text files,
    and Ollama for code files and everything else.
    """
    
    # File types that should use BERT (natural language text)
    BERT_FILE_TYPES = {'text'}  # Only text files (.txt, .md, .rst, etc.)
    
    def __init__(self, ollama_embeddings, use_bert: bool = True):
        """
        Initialize smart embedder.
        
        Args:
            ollama_embeddings: OllamaEmbeddings instance
            use_bert: Whether to use BERT for text files
        """
        self.ollama = ollama_embeddings
        self.bert = BERTEmbedder() if use_bert else None
        
        # Track whether BERT is available and should be used
        self.use_bert = use_bert and self.bert and self.bert.is_available
        
        if self.use_bert:
            print("🎯 Smart Embedder: Using BERT for natural language text files, Ollama for code and other files")
        else:
            print("📝 Smart Embedder: Using Ollama for all file types")
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query. For queries, we use the primary embedder (Ollama)
        to ensure consistent vector space across all document types.
        
        Args:
            text: Query text
            
        Returns:
            Embedding vector
        """
        # Always use Ollama for queries to maintain consistent vector space
        # This ensures that text files (embedded with BERT) and other files
        # (embedded with Ollama) can be compared in the same space
        return self.ollama.embed_query(text)
    
    def embed_documents(self, texts: List[str], file_types: List[str] = None) -> List[List[float]]:
        """
        Embed documents, using BERT only for natural language text files,
        and Ollama for code files and everything else.
        
        Args:
            texts: List of document texts
            file_types: List of file types corresponding to each text
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # If no file types provided or BERT not available, use Ollama for all
        if file_types is None or not self.use_bert:
            return self.ollama.embed_documents(texts)
        
        # Split documents by type
        bert_indices = []
        bert_texts = []
        ollama_indices = []
        ollama_texts = []
        
        for i, (text, ftype) in enumerate(zip(texts, file_types)):
            # Use BERT ONLY for natural language text files (not code)
            if ftype in self.BERT_FILE_TYPES:
                bert_indices.append(i)
                bert_texts.append(text)
            else:
                # Code, PDF, images, etc. use Ollama
                ollama_indices.append(i)
                ollama_texts.append(text)
        
        # Initialize result array
        results = [None] * len(texts)
        
        # Process BERT texts (only natural language text files)
        if bert_texts:
            bert_embeddings = self.bert.embed_documents(bert_texts)
            if bert_embeddings:
                for idx, emb in zip(bert_indices, bert_embeddings):
                    results[idx] = emb
            else:
                # Fallback to Ollama if BERT fails
                fallback_embeddings = self.ollama.embed_documents(bert_texts)
                for idx, emb in zip(bert_indices, fallback_embeddings):
                    results[idx] = emb
        
        # Process Ollama texts (code, PDFs, images, etc.)
        if ollama_texts:
            ollama_embeddings = self.ollama.embed_documents(ollama_texts)
            for idx, emb in zip(ollama_indices, ollama_embeddings):
                results[idx] = emb
        
        return results


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
        """
        Load a single document from a file path.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Document object or None if loading fails
        """
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
        # Only natural language text files, no code files
        return {'.txt', '.md', '.rst', '.text'}
    
    def load_document(self, file_path: str) -> Optional[Document]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create document with content and metadata
            metadata = {
                "filename": os.path.basename(file_path),
                "filepath": file_path,
                "type": "text",  # This will trigger BERT embedding
                "size": os.path.getsize(file_path),
                "extension": os.path.splitext(file_path)[1].lower(),
                "match_source": "file_content",
                "embedding_type": "bert" if BERT_AVAILABLE else "ollama"
            }
            
            return Document(
                page_content=content,
                metadata=metadata
            )
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
                "embedding_type": "ollama"  # Markup files use Ollama
            }
            
            return Document(
                page_content=content,
                metadata=metadata
            )
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
                "embedding_type": "ollama"  # PDFs use Ollama
            }
            
            return Document(
                page_content=content,
                metadata=metadata
            )
        except Exception as e:
            print(f"Error loading PDF {file_path}: {e}")
            return None


class ImageLoader(DocumentLoader):
    """Loader for image files (using metadata and OCR)."""
    
    def _get_supported_extensions(self) -> Set[str]:
        return {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    
    def load_document(self, file_path: str) -> Optional[Document]:
        try:
            # Try to use PIL for image metadata
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
                "embedding_type": "ollama"  # Images use Ollama
            }
            
            return Document(
                page_content=content,
                metadata=metadata
            )
        except ImportError:
            print("Please install Pillow: pip install Pillow")
            return None
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            return None


class CodeLoader(DocumentLoader):
    """Loader for code files - uses Ollama."""
    
    def _get_supported_extensions(self) -> Set[str]:
        return {'.py', '.cpp', '.c', '.h', '.hpp', '.js', '.ts', '.java', '.go', '.rs', '.rb', '.php', '.sh', '.bash'}
    
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
                "type": "code",  # This will NOT trigger BERT
                "language": os.path.splitext(file_path)[1][1:],
                "lines_total": len(lines),
                "lines_code": len(code_lines),
                "lines_comments": len(comment_lines),
                "size": os.path.getsize(file_path),
                "extension": os.path.splitext(file_path)[1].lower(),
                "match_source": "file_content",
                "embedding_type": "ollama"  # Code files use Ollama
            }
            
            return Document(
                page_content=content,
                metadata=metadata
            )
        except Exception as e:
            print(f"Error loading code file {file_path}: {e}")
            return None


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


class DocumentProcessor:
    """
    Main processor that manages document loaders and vector database operations.
    """
    
    def __init__(self, db_path: str, embed_model: str, extensions: Optional[List[str]] = None, use_bert: bool = True):
        """
        Initialize the document processor.
        
        Args:
            db_path: Path to Chroma DB
            embed_model: Ollama embedding model name (fallback)
            extensions: List of file extensions to filter (optional)
            use_bert: Whether to use BERT for text files
        """
        # Create Ollama embeddings
        self.ollama_embeddings = OllamaEmbeddings(model=embed_model)
        
        # Create smart embedder that chooses based on file type
        self.smart_embedder = SmartEmbedder(self.ollama_embeddings, use_bert)
        
        # Initialize vector store with smart embedder
        self.vector_store = Chroma(
            collection_name="document_collection",
            embedding_function=self.smart_embedder,
            persist_directory=db_path
        )
        
        # Initialize available loaders
        self.loaders = self._init_loaders(extensions)
        
        # Store file types for embedding
        self.pending_docs = []  # Store (document, file_type) pairs before embedding
        
    def _init_loaders(self, extensions: Optional[List[str]] = None) -> List[DocumentLoader]:
        """Initialize all available document loaders."""
        ext_set = set(extensions) if extensions else None
        
        # Create loader instances
        loaders = [
            TextFileLoader(ext_set),      # Natural language text files (uses BERT)
            MarkupFileLoader(ext_set),    # HTML, XML, JSON, etc. (uses Ollama)
            PDFLoader(ext_set),           # PDF files (uses Ollama)
            ImageLoader(ext_set),         # Image files (uses Ollama)
            CodeLoader(ext_set)           # Code files (uses Ollama)
        ]
        
        # Filter loaders that have support for requested extensions
        if ext_set:
            loaders = [l for l in loaders if l.supported_extensions & ext_set]
        
        return loaders
    
    def process_file(self, file_path: str) -> Optional[EnhancedDocument]:
        """
        Process a single file using the appropriate loader.
        
        Args:
            file_path: Path to the file
            
        Returns:
            EnhancedDocument object or None if no loader can handle it
        """
        for loader in self.loaders:
            if loader.can_handle(file_path):
                doc = loader.load_document(file_path)
                if doc:
                    # Convert to EnhancedDocument
                    return EnhancedDocument(
                        page_content=doc.page_content,
                        metadata=doc.metadata
                    )
        return None
    
    def add_files_from_directory(self, directory: str) -> None:
        """
        Walk through directory and add all supported files to the vector store.
        
        Args:
            directory: Directory to scan for files
        """
        documents = []
        file_types = []  # Track file types for smart embedding
        count = 0
        failed = 0
        content_matches = 0
        metadata_matches = 0
        bert_embedded = 0
        ollama_embedded = 0
        
        for root, _, files in os.walk(directory):
            for filename in files:
                file_path = os.path.join(root, filename)
                
                doc = self.process_file(file_path)
                if doc:
                    documents.append(doc)
                    file_types.append(doc.metadata.get('type', 'unknown'))
                    count += 1
                    
                    # Track match types
                    if doc.metadata.get('match_source') == 'file_content':
                        content_matches += 1
                    elif doc.metadata.get('match_source') == 'metadata_only':
                        metadata_matches += 1
                    
                    # Track embedding types
                    if doc.metadata.get('embedding_type') == 'bert':
                        bert_embedded += 1
                    else:
                        ollama_embedded += 1
                    
                    # Add in batches to avoid memory issues
                    if len(documents) >= 100:
                        # Add to vector store
                        self.vector_store.add_documents(documents)
                        documents = []
                        file_types = []
                else:
                    failed += 1
        
        if documents:
            # Add remaining documents
            self.vector_store.add_documents(documents)
        
        print(f"\n📊 Indexing Summary:")
        print(f"   ✅ Added {count} files to the database")
        print(f"   📝 {content_matches} files with searchable content")
        print(f"   🏷️  {metadata_matches} files indexed by metadata only")
        if bert_embedded > 0:
            print(f"   🧠 {bert_embedded} natural language text files embedded with BERT")
        if ollama_embedded > 0:
            print(f"   🔧 {ollama_embedded} files embedded with Ollama (code, PDFs, images, markup)")
        if failed > 0:
            print(f"   ⚠️  Skipped {failed} unsupported files")
    
    def search(self, query: str, k: int = 5) -> List[EnhancedDocument]:
        """
        Search for documents matching the query.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of EnhancedDocument objects
        """
        results = self.vector_store.similarity_search(query, k=k)
        # Convert to EnhancedDocument if needed
        return [EnhancedDocument(page_content=doc.page_content, metadata=doc.metadata) 
                if not isinstance(doc, EnhancedDocument) else doc 
                for doc in results]


def main():
    parser = argparse.ArgumentParser(description="Interactive document search with Ollama embeddings and optional BERT for text.")
    parser.add_argument("--db_path", default="./chroma_db", help="Path to Chroma DB")
    parser.add_argument("--embed_model", default="Qwen3-Embedding:8B", help="Ollama embedding model (fallback)")
    parser.add_argument("--extensions", nargs="+", default=None,
                        help="File extensions to include when filling (e.g., .txt .pdf .py)")
    parser.add_argument("--k", type=int, default=5, help="Number of results to return per query")
    parser.add_argument("--verbose", action="store_true", help="Show full content preview")
    parser.add_argument("--no-bert", action="store_true", help="Disable BERT for text files (use Ollama only)")
    args = parser.parse_args()
    
    # Create document processor with BERT enabled by default
    processor = DocumentProcessor(
        args.db_path, 
        args.embed_model, 
        args.extensions,
        use_bert=not args.no_bert
    )
    
    print("🔍 Interactive Document Search with Smart Embedding")
    print("=" * 60)
    print("Commands:")
    print("  fill <path>   - add documents from <path> to the database")
    print("  <query>       - search for documents matching the query")
    print("  q             - quit")
    print("\nSupported extensions: " + 
          ", ".join(sorted(set().union(*[l.supported_extensions for l in processor.loaders]))))
    
    if not args.no_bert and BERT_AVAILABLE:
        print("\n🧠 BERT enabled: Natural language text files (.txt, .md, .rst) use BERT embeddings")
        print("   Code files, PDFs, images, markup files use Ollama embeddings")
    else:
        print("\n🔧 Using Ollama for all file types")
    
    print("\n💡 Tip: Results show match source and embedding type (🧠=BERT for text, 🔧=Ollama for others)")
    print()
    
    while True:
        user_input = input("\n> ").strip()
        if not user_input:
            continue
        if user_input.lower() == 'q':
            print("Exiting.")
            break
        
        # Check for fill command
        if user_input.lower().startswith("fill "):
            path = user_input[5:].strip()
            if not os.path.isdir(path):
                print(f"Error: '{path}' is not a valid directory.")
                continue
            processor.add_files_from_directory(path)
        else:
            # Treat as query
            results = processor.search(user_input, k=args.k)
            print(f"\n📋 Top {len(results)} results for: \"{user_input}\"")
            print("-" * 60)
            
            for i, doc in enumerate(results, 1):
                print(f"\n{i}. {doc.metadata.get('filename', 'N/A')}")
                print(f"   📂 Path: {doc.metadata.get('filepath', 'N/A')}")
                print(f"   🏷️  Type: {doc.get_file_type_display()}")
                print(f"   🔍 Match: {doc.get_match_type_display()}")
                
                # Show embedding type
                embedder = doc.metadata.get('embedding_type', 'ollama')
                if embedder == 'bert':
                    print(f"   🧠 Embedded with: BERT (semantic search for natural language)")
                else:
                    print(f"   🔧 Embedded with: Ollama")
                
                # Additional metadata based on file type
                if doc.metadata.get('type') == 'code':
                    print(f"   📊 Stats: {doc.metadata.get('lines_code', 0)} lines of code, "
                          f"{doc.metadata.get('lines_comments', 0)} comments")
                elif doc.metadata.get('type') == 'pdf':
                    if doc.metadata.get('has_text_content', False):
                        print(f"   📄 Contains {doc.metadata.get('pages', 0)} pages with extractable text")
                    else:
                        print(f"   ⚠️  PDF has {doc.metadata.get('pages', 0)} pages but no extractable text")
                elif doc.metadata.get('type') == 'image':
                    print(f"   📐 Dimensions: {doc.metadata.get('width', '?')}x{doc.metadata.get('height', '?')}")
                
                # Show content preview
                if args.verbose or doc.metadata.get('match_source') == 'metadata_only':
                    content_preview = doc.page_content[:200].replace('\n', ' ')
                    print(f"   📝 Preview: {content_preview}...")
                else:
                    # Show a snippet from the content
                    content_preview = doc.page_content[:150].replace('\n', ' ')
                    if content_preview:
                        print(f"   📝 Content snippet: {content_preview}...")
                
                # Add separator between results
                if i < len(results):
                    print("   " + "-" * 50)


if __name__ == "__main__":
    main()
