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


class DocumentProcessor:
    """Main processor that manages document loaders and vector database operations."""
    
    def __init__(self, db_path: str, embed_model: str, extensions: Optional[List[str]] = None, 
                 use_bert: bool = True, filter_empty: bool = True):
        """
        Initialize the document processor.
        
        Args:
            db_path: Path to Chroma DB
            embed_model: Ollama embedding model name (fallback)
            extensions: List of file extensions to filter (optional)
            use_bert: Whether to use BERT for text files
            filter_empty: Whether to filter out empty files from results
        """
        from langchain_ollama import OllamaEmbeddings
        
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
        self.query_matcher = QueryMatcher()
        self.filter_empty = filter_empty
        
        # Track empty files
        self.empty_files_count = 0
    
    def _init_loaders(self, extensions: Optional[List[str]] = None) -> List[DocumentLoader]:
        """Initialize all available document loaders."""
        ext_set = set(extensions) if extensions else None
        
        # Create loader instances
        loaders = [
            TextFileLoader(ext_set),
            MarkupFileLoader(ext_set),
            PDFLoader(ext_set),
            ImageLoader(ext_set),
            CodeLoader(ext_set)
        ]
        
        # Filter loaders that have support for requested extensions
        if ext_set:
            loaders = [l for l in loaders if l.supported_extensions & ext_set]
        
        return loaders
    
    def process_file(self, file_path: str) -> Optional[EnhancedDocument]:
        """Process a single file using the appropriate loader."""
        for loader in self.loaders:
            if loader.can_handle(file_path):
                doc = loader.load_document(file_path)
                if doc:
                    enhanced_doc = EnhancedDocument(
                        page_content=doc.page_content,
                        metadata=doc.metadata
                    )
                    
                    # Skip empty files during indexing (optional)
                    if enhanced_doc.is_empty():
                        self.empty_files_count += 1
                        # Optionally skip adding empty files to database
                        return None
                    
                    return enhanced_doc
        return None
    
    def add_files_from_directory(self, directory: str) -> None:
        """Walk through directory and add all supported files to the vector store."""
        documents = []
        count = 0
        failed = 0
        empty_skipped = 0
        content_matches = 0
        metadata_matches = 0
        bert_embedded = 0
        ollama_embedded = 0
        self.empty_files_count = 0  # Reset counter
        
        for root, _, files in os.walk(directory):
            for filename in files:
                file_path = os.path.join(root, filename)
                
                doc = self.process_file(file_path)
                if doc:
                    documents.append(doc)
                    count += 1
                    
                    if doc.metadata.get('match_source') == 'file_content':
                        content_matches += 1
                    elif doc.metadata.get('match_source') == 'metadata_only':
                        metadata_matches += 1
                    
                    if doc.metadata.get('embedding_type') == 'bert':
                        bert_embedded += 1
                    else:
                        ollama_embedded += 1
                    
                    if len(documents) >= 100:
                        self.vector_store.add_documents(documents)
                        documents = []
                else:
                    failed += 1
        
        if documents:
            self.vector_store.add_documents(documents)
        
        print(f"\n📊 Indexing Summary:")
        print(f"   ✅ Added {count} files to the database")
        
        if self.empty_files_count > 0:
            print(f"   ⚠️  Skipped {self.empty_files_count} empty files (no meaningful content)")
        
        print(f"   📝 {content_matches} files with searchable content")
        print(f"   🏷️  {metadata_matches} files indexed by metadata only")
        
        if bert_embedded > 0:
            print(f"   🧠 {bert_embedded} natural language text files embedded with BERT")
        if ollama_embedded > 0:
            print(f"   🔧 {ollama_embedded} files embedded with Ollama")
        if failed > 0:
            print(f"   ⚠️  Skipped {failed} unsupported files")
    
    def search(self, query: str, k: int = 5, min_content_length: int = 10) -> List[Tuple[EnhancedDocument, List[Dict[str, Any]]]]:
        """
        Search for documents and find matching snippets.
        
        Args:
            query: Search query
            k: Number of results to return
            min_content_length: Minimum content length to consider (filters out very short/empty files)
        
        Returns:
            List of tuples (document, matching_snippets)
        """
        # Get more results than needed to filter out empty ones
        fetch_k = k * 3 if self.filter_empty else k
        results = self.vector_store.similarity_search(query, k=fetch_k)
        
        enhanced_results = []
        
        for doc in results:
            enhanced_doc = EnhancedDocument(
                page_content=doc.page_content,
                metadata=doc.metadata
            )
            
            # Skip empty or very short files if filtering is enabled
            if self.filter_empty:
                if enhanced_doc.is_empty():
                    continue
                
                # Additional check: if content is too short, skip it
                if len(enhanced_doc.page_content.strip()) < min_content_length:
                    continue
            
            # Find matching snippets
            if enhanced_doc.metadata.get('match_source') == 'file_content':
                snippets = self.query_matcher.find_matching_snippets(
                    doc.page_content, 
                    query
                )
            else:
                # For metadata-only matches, show filename/path matches
                filename = enhanced_doc.metadata.get('filename', '')
                filepath = enhanced_doc.metadata.get('filepath', '')
                combined = f"{filename} {filepath}"
                snippets = self.query_matcher.find_matching_snippets(combined, query)
            
            enhanced_results.append((enhanced_doc, snippets))
            
            # Stop when we have enough results
            if len(enhanced_results) >= k:
                break
        
        return enhanced_results


