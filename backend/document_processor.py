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
from enhanced_document import EnhancedDocument


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
   
        self.use_summaries = False      
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
        
        # Test embedding functionality
        self._test_embeddings(embed_model)
    
    
    def _test_embeddings(self, embed_model: str) -> None:
        """Test if embeddings are working properly."""
        try:
            print("\n🔍 Testing embedding functionality...")
            test_query = "test document search"
            test_embedding = self.smart_embedder.embed_query(test_query)
            
            if not test_embedding:
                print("   ❌ ERROR: Embedding returned None")
                print("   Check if Ollama is running: 'ollama serve'")
                print(f"   Verify model is available: 'ollama pull {embed_model}'")
            elif all(v == 0 for v in test_embedding):
                print("   ❌ ERROR: Embedding is all zeros!")
                print("   The embedding model is not working correctly.")
                print(f"   Try: ollama pull {embed_model}")
                print("   Also check: ollama list")
            else:
                # Calculate variance to see if embeddings are meaningful
                variance = sum(v*v for v in test_embedding) / len(test_embedding)
                print(f"   ✅ Embedding model verified")
                print(f"   📊 Embedding dimension: {len(test_embedding)}")
                print(f"   📊 Vector variance: {variance:.6f} (should be > 0)")
                
                if variance < 0.0001:
                    print("   ⚠️  WARNING: Very low variance - embeddings may be poor quality")
                else:
                    print("   ✅ Embedding quality looks good")
                    
        except Exception as e:
            print(f"\n   ❌ ERROR: Failed to create test embedding: {e}")
            print("   Make sure Ollama is running: 'ollama serve'")
            print(f"   Install the model: 'ollama pull {embed_model}'")
    
    
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
    
    
    def search(self, query: str, k: int = 10, min_content_length: int = 10, score_threshold: float = 0.1) -> List[Tuple[EnhancedDocument, List[Dict[str, Any]]]]:
        """
        Search for documents and find matching snippets.
        
        Args:
            query: Search query
            k: Number of results to return
            min_content_length: Minimum content length to consider (filters out very short/empty files)
            score_threshold: Minimum relevance score (0-1). Documents below this are filtered out.
                           Start with 0.3 and adjust based on your results.
        
        Returns:
            List of tuples (document, matching_snippets)
        """
        # Get more results than needed to filter out low scores and empty ones
        fetch_k = k * 3 if self.filter_empty else k
        
        # CRITICAL FIX: Use similarity_search_with_score to get relevance scores
        results_with_scores = self.vector_store.similarity_search_with_score(query, k=fetch_k)
        
        # Debug: Show scores to understand what's happening
        if results_with_scores:
            print(f"\n📊 Relevance scores for '{query}':")
            for doc, score in results_with_scores[:k]:  # Show top k scores
                filename = doc.metadata.get('filename', 'N/A')[:40]
                print(f"   Score {score:.4f}: {filename}")
        
        enhanced_results = []
        
        for doc, score in results_with_scores:
            # CRITICAL FIX: Skip documents with relevance score below threshold
            if score < score_threshold:
                continue
            
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
            
            # Find matching snippets for display
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
        
        # Provide feedback if no results due to threshold
        if not enhanced_results and results_with_scores:
            max_score = max(score for _, score in results_with_scores) if results_with_scores else 0
            print(f"\n⚠️  Found {len(results_with_scores)} potential results but none above threshold ({score_threshold})")
            print(f"   Highest relevance score was: {max_score:.4f}")
            print(f"   Try lowering the score_threshold or check your embedding model")
            print(f"   Example: Use 'score_threshold=0.1' for more results")
        
        return enhanced_results
    
    
    def debug_search(self, query: str, k: int = 10) -> None:
        """
        Debug method to show detailed search information.
        
        Args:
            query: Search query
            k: Number of results to show
        """
        print(f"\n🔍 DEBUG SEARCH for: '{query}'")
        print("=" * 60)
        
        # Get raw results with scores
        results_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
        
        if not results_with_scores:
            print("No results found!")
            return
        
        print(f"\nFound {len(results_with_scores)} results:\n")
        
        for i, (doc, score) in enumerate(results_with_scores, 1):
            print(f"{i}. Score: {score:.6f}")
            print(f"   Filename: {doc.metadata.get('filename', 'N/A')}")
            print(f"   Filepath: {doc.metadata.get('filepath', 'N/A')}")
            print(f"   Type: {doc.metadata.get('type', 'unknown')}")
            print(f"   Content preview: {doc.page_content[:150]}...")
            print(f"   Content length: {len(doc.page_content)} chars")
            print("-" * 60)

    def _get_file_stats(self, file_path: str) -> Dict[str, Any]:
        """
        Extract comprehensive file statistics.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file metadata
        """
        try:
            stat = os.stat(file_path)
            path_obj = Path(file_path)
            
            stats = {
                'size_bytes': stat.st_size,
                'created_time': stat.st_ctime,
                'modified_time': stat.st_mtime,
                'accessed_time': stat.st_atime,
                'file_extension': path_obj.suffix.lower(),
                'file_name': path_obj.name,
                'file_name_stem': path_obj.stem,
                'parent_dir': str(path_obj.parent),
                'is_symlink': os.path.islink(file_path),
                'permissions': oct(stat.st_mode)[-3:],
            }
            
            # Add human-readable dates
            stats['created_date'] = datetime.fromtimestamp(stat.st_ctime).isoformat()
            stats['modified_date'] = datetime.fromtimestamp(stat.st_mtime).isoformat()
            
            # Check if file is binary (simple heuristic)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    f.read(1024)
                stats['is_text'] = True
            except (UnicodeDecodeError, IOError):
                stats['is_text'] = False
            
            # Add file age in days
            stats['age_days'] = (time.time() - stat.st_ctime) / 86400
            
            return stats
            
        except Exception as e:
            print(f"⚠️  Could not read file stats for {file_path}: {e}")
            return {}
    
    
    def process_file(self, file_path: str) -> Optional[EnhancedDocument]:
        """Process a single file using the appropriate loader."""
        for loader in self.loaders:
            if loader.can_handle(file_path):
                doc = loader.load_document(file_path)
                if doc:
                    # Get file statistics
                    file_stats = self._get_file_stats(file_path)
                    
                    # Generate summary if enabled
                    summary = None
                    if self.use_summaries and doc.page_content:
                        print(f"   📝 Generating summary for {os.path.basename(file_path)}...")
                        summary = self.summarizer.generate_summary(doc)
                    
                    # Enhance metadata with file stats
                    enhanced_metadata = doc.metadata.copy()
                    enhanced_metadata.update({
                        'size_human': self._format_size(file_stats.get('size_bytes', 0)),
                        'age_days': round(file_stats.get('age_days', 0), 1),
                        'is_recent': file_stats.get('age_days', 999) < 30,
                    })
                    
                    enhanced_doc = EnhancedDocument(
                        page_content=doc.page_content,
                        metadata=enhanced_metadata,
                        summary=summary,
                        file_stats=file_stats
                    )
                    
                    if enhanced_doc.is_empty():
                        self.empty_files_count += 1
                        return None
                    
                    return enhanced_doc
        return None
    
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size for human readability."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"


# Test function to verify the processor works correctly
def test_processor():
    """Test the DocumentProcessor with a simple example."""
    import tempfile
    import shutil
    
    print("\n" + "=" * 60)
    print("TESTING DOCUMENT PROCESSOR")
    print("=" * 60)
    
    # Create temporary directory for testing
    test_dir = tempfile.mkdtemp()
    db_path = tempfile.mkdtemp()
    
    try:
        # Create test files
        test_file = os.path.join(test_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("This is a test document about Python programming and machine learning.")
        
        # Initialize processor
        processor = DocumentProcessor(
            db_path=db_path,
            #embed_model="Qwen3-Embedding:8B",
            embed_model="nomic-embed-text:latest",
            use_bert=False  # Use Ollama for testing
        )
        
        # Add test file
        print("\n📁 Adding test file to database...")
        processor.add_files_from_directory(test_dir)
        
        # Test search
        print("\n🔍 Testing search functionality...")
        results = processor.search("python programming", k=3, score_threshold=0.1)
        
        if results:
            print(f"\n✅ Search successful! Found {len(results)} results")
            for doc, snippets in results:
                print(f"   - {doc.metadata.get('filename')} (score-based match)")
        else:
            print("\n⚠️  No results found. Trying debug mode...")
            processor.debug_search("python programming")
        
        # Test embedding quality
        print("\n📊 Testing embedding quality...")
        test_embedding = processor.smart_embedder.embed_query("test query")
        print(f"   Embedding dimension: {len(test_embedding)}")
        print(f"   Non-zero values: {sum(1 for v in test_embedding if v != 0)}/{len(test_embedding)}")
        
    finally:
        # Cleanup
        shutil.rmtree(test_dir, ignore_errors=True)
        shutil.rmtree(db_path, ignore_errors=True)
        print("\n🧹 Test cleanup complete")


if __name__ == "__main__":
    # Run test when script is executed directly
    test_processor()
