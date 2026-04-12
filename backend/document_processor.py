"""
Document processor that manages loaders and vector database operations.
"""

import os
import time
from datetime import datetime
from pathlib import Path
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
    
    
    def search(self, query: str, k: int = 10, offset: int = 0, 
               min_content_length: int = 10, score_threshold: float = 0.1,
               max_total: int = 1000) -> List[Tuple[EnhancedDocument, List[Dict[str, Any]]]]:
        """
        Search for documents and find matching snippets with pagination support.
        """
        fetch_k = min(max_total, offset + k + 100)
        results_with_scores = self.vector_store.similarity_search_with_score(query, k=fetch_k)
    
        # Single loop - store results with their scores
        all_filtered_results = []  # Each element: (enhanced_doc, snippets, score)
        
        for doc, score in results_with_scores:
            if score < score_threshold:
                continue
            
            enhanced_doc = EnhancedDocument(
                page_content=doc.page_content,
                metadata=doc.metadata
            )
            
            if self.filter_empty:
                if enhanced_doc.is_empty() or len(enhanced_doc.page_content.strip()) < min_content_length:
                    continue
        
            # Find matching snippets
            if enhanced_doc.metadata.get('match_source') == 'file_content':
                snippets = self.query_matcher.find_matching_snippets(doc.page_content, query)
            else:
                filename = doc.metadata.get('filename', '')
                filepath = doc.metadata.get('filepath', '')
                snippets = self.query_matcher.find_matching_snippets(f"{filename} {filepath}", query)
            
            all_filtered_results.append((enhanced_doc, snippets, score))
            
            if len(all_filtered_results) >= max_total:
                break
    
        total_count = len(all_filtered_results)
        paginated_results = all_filtered_results[offset:offset + k]
        
        # Display scores for current page
        if paginated_results:
            print(f"\n📊 Relevance scores for '{query}':")
            for i, (doc, snippets, score) in enumerate(paginated_results, offset + 1):
                filename = doc.metadata.get('filename', 'N/A')[:40]
                print(f"   [{i}] Score {score:.4f}: {filename}")
            
            print(f"\n📄 Showing results {offset + 1}-{min(offset + k, total_count)} of {total_count} total")
            
            # Return without scores (backward compatible)
            return [(doc, snippets) for doc, snippets, _ in paginated_results]
        elif results_with_scores:
            max_score = max(score for _, score in results_with_scores)
            print(f"\n⚠️  Found {len(results_with_scores)} results but none above threshold ({score_threshold})")
            print(f"   Highest score: {max_score:.4f}")
        
        return []
    
    
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
        """Process a single file using the appropriate loader with comprehensive statistics."""
        for loader in self.loaders:
            if loader.can_handle(file_path):
                doc = loader.load_document(file_path)
                if doc:
                    # Get comprehensive file statistics
                    file_stats = self._get_enhanced_file_stats(file_path, doc)
                    
                    # Generate summary if enabled
                    summary = None
                    if self.use_summaries and doc.page_content:
                        print(f"   📝 Generating summary for {os.path.basename(file_path)}...")
                        summary = self.summarizer.generate_summary(doc)
                    
                    # Enhance metadata with aggregated stats
                    enhanced_metadata = doc.metadata.copy()
                    enhanced_metadata.update({
                        # Size information
                        'size_human': self._format_size(file_stats.get('size_bytes', 0)),
                        'size_bytes': file_stats.get('size_bytes', 0),
                        'size_category': file_stats.get('size_category', 'unknown'),
                        
                        # Time information
                        'age_days': round(file_stats.get('age_days', 0), 1),
                        'age_hours': round(file_stats.get('age_hours', 0), 1),
                        'is_recent': file_stats.get('is_recent', False),
                        'is_old': file_stats.get('is_old', False),
                        'created_date': file_stats.get('created_date', ''),
                        'modified_date': file_stats.get('modified_date', ''),
                        
                        # Content statistics
                        'word_count': file_stats.get('word_count', 0),
                        'char_count': file_stats.get('char_count', 0),
                        'line_count': file_stats.get('line_count', 0),
                        'has_content': file_stats.get('has_content', False),
                        'content_density': file_stats.get('content_density', 0.0),
                        
                        # File properties
                        'file_extension': file_stats.get('file_extension', ''),
                        'is_binary': not file_stats.get('is_text', True),
                        'is_text': file_stats.get('is_text', True),
                        'is_code_file': file_stats.get('is_code_file', False),
                        'is_document': file_stats.get('is_document', False),
                        'is_image': file_stats.get('is_image', False),
                        
                        # Complexity metrics (for code/text)
                        'complexity_score': file_stats.get('complexity_score', 0.0),
                        'unique_terms': file_stats.get('unique_terms', 0),
                        
                        # Directory information
                        'depth': file_stats.get('depth', 0),
                        'parent_dir': file_stats.get('parent_dir', ''),
                        
                        # Permissions and security
                        'permissions': file_stats.get('permissions', '000'),
                        'is_readable': file_stats.get('is_readable', True),
                        'is_writable': file_stats.get('is_writable', True),
                        
                        # Processing metadata
                        'processing_time_ms': file_stats.get('processing_time_ms', 0),
                        'embedding_time_ms': file_stats.get('embedding_time_ms', 0),
                    })
                    
                    enhanced_doc = EnhancedDocument(
                        page_content=doc.page_content,
                        metadata=enhanced_metadata,
                        summary=summary,
                        file_stats=file_stats  # Store full stats separately
                    )
                    
                    if enhanced_doc.is_empty():
                        self.empty_files_count += 1
                        return None
                    
                    # Print detailed stats for debugging (optional)
                    if self.verbose:
                        self._print_file_stats(file_path, file_stats)
                    
                    return enhanced_doc
        return None    

    def _get_enhanced_file_stats(self, file_path: str, doc: Any = None) -> Dict[str, Any]:
        """
        Extract comprehensive file statistics with content analysis.
        
        Args:
            file_path: Path to the file
            doc: Loaded document (optional, for content analysis)
            
        Returns:
            Dictionary with comprehensive file metadata
        """
        stats = {}
        start_time = time.time()
        
        try:
            # Basic file system stats
            stat = os.stat(file_path)
            path_obj = Path(file_path)
            
            # File system information
            stats.update({
                'size_bytes': stat.st_size,
                'size_blocks': stat.st_blocks,
                'created_time': stat.st_ctime,
                'modified_time': stat.st_mtime,
                'accessed_time': stat.st_atime,
                'inode': stat.st_ino,
                'device_id': stat.st_dev,
                'hard_links': stat.st_nlink,
                'user_id': stat.st_uid,
                'group_id': stat.st_gid,
            })
            
            # Path analysis
            stats.update({
                'file_extension': path_obj.suffix.lower(),
                'file_name': path_obj.name,
                'file_name_stem': path_obj.stem,
                'parent_dir': str(path_obj.parent),
                'depth': len(path_obj.parents) - 1,  # Depth from root
                'is_hidden': path_obj.name.startswith('.'),
                'is_symlink': os.path.islink(file_path),
                'absolute_path': str(path_obj.absolute()),
            })
            
            # Human-readable dates
            stats['created_date'] = datetime.fromtimestamp(stat.st_ctime).isoformat()
            stats['modified_date'] = datetime.fromtimestamp(stat.st_mtime).isoformat()
            stats['created_date_human'] = datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
            stats['modified_date_human'] = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            
            # Time-based calculations
            current_time = time.time()
            stats['age_days'] = (current_time - stat.st_ctime) / 86400
            stats['age_hours'] = (current_time - stat.st_ctime) / 3600
            stats['age_minutes'] = (current_time - stat.st_ctime) / 60
            stats['is_recent'] = stats['age_days'] < 30
            stats['is_old'] = stats['age_days'] > 365
            stats['days_since_modified'] = (current_time - stat.st_mtime) / 86400
            
            # Size categorization
            if stats['size_bytes'] < 1024:
                stats['size_category'] = 'tiny'
            elif stats['size_bytes'] < 1024 * 10:
                stats['size_category'] = 'very_small'
            elif stats['size_bytes'] < 1024 * 100:
                stats['size_category'] = 'small'
            elif stats['size_bytes'] < 1024 * 1024:
                stats['size_category'] = 'medium'
            elif stats['size_bytes'] < 1024 * 1024 * 10:
                stats['size_category'] = 'large'
            else:
                stats['size_category'] = 'very_large'
            
            # Permissions
            stats['permissions'] = oct(stat.st_mode)[-3:]
            stats['permissions_readable'] = self._parse_permissions(stat.st_mode)
            stats['is_readable'] = os.access(file_path, os.R_OK)
            stats['is_writable'] = os.access(file_path, os.W_OK)
            stats['is_executable'] = os.access(file_path, os.X_OK)
            
            # File type classification
            stats['is_text'] = self._is_text_file(file_path)
            stats['is_binary'] = not stats['is_text']
            stats['is_code_file'] = self._is_code_file(file_path)
            stats['is_document'] = self._is_document_file(file_path)
            stats['is_image'] = self._is_image_file(file_path)
            stats['is_archive'] = self._is_archive_file(file_path)
            
            # Content analysis (if document provided)
            if doc and hasattr(doc, 'page_content') and doc.page_content:
                content = doc.page_content
                
                # Basic content statistics
                stats['word_count'] = len(content.split())
                stats['char_count'] = len(content)
                stats['line_count'] = content.count('\n') + 1
                stats['paragraph_count'] = content.count('\n\n') + 1
                stats['has_content'] = stats['word_count'] > 0
                
                # Content density (words per KB)
                stats['content_density'] = (stats['word_count'] / max(1, stats['size_bytes'])) * 1024
                
                # Unique terms analysis
                words = content.lower().split()
                unique_words = set(words)
                stats['unique_terms'] = len(unique_words)
                stats['lexical_diversity'] = len(unique_words) / max(1, len(words))
                
                # Sentence and paragraph analysis
                stats['sentence_count'] = content.count('.') + content.count('!') + content.count('?')
                stats['avg_word_length'] = sum(len(w) for w in words) / max(1, len(words))
                stats['avg_sentence_length'] = len(words) / max(1, stats['sentence_count'])
                
                # Complexity scoring (for code/text)
                stats['complexity_score'] = self._calculate_complexity(content, stats['is_code_file'])
                
                # Language detection (basic)
                stats['likely_language'] = self._detect_language(content)
                
                # Special metrics for code files
                if stats['is_code_file']:
                    stats.update(self._analyze_code_file(content))
                
                # Special metrics for documents
                if stats['is_document']:
                    stats.update(self._analyze_document(content))
            
            # Performance metrics
            stats['processing_time_ms'] = (time.time() - start_time) * 1000
            
            # Add checksums for file integrity (optional)
            stats['file_hash_md5'] = self._compute_file_hash(file_path, 'md5')
            stats['file_hash_sha256'] = self._compute_file_hash(file_path, 'sha256')
            
        except Exception as e:
            print(f"⚠️  Could not read file stats for {file_path}: {e}")
            stats['error'] = str(e)
        
        return stats    


    def _is_text_file(self, file_path: str) -> bool:
        """Determine if file is text-based."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read(1024)
            return True
        except (UnicodeDecodeError, IOError):
            return False


    def _is_code_file(self, file_path: str) -> bool:
        """Check if file is a source code file."""
        code_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.h', 
            '.hpp', '.cs', '.go', '.rs', '.php', '.rb', '.swift', '.kt', '.kts',
            '.scala', '.sql', '.sh', '.bash', '.zsh', '.pl', '.pm', '.lua', '.r'
        }
        ext = Path(file_path).suffix.lower()
        return ext in code_extensions


    def _is_document_file(self, file_path: str) -> bool:
        """Check if file is a document."""
        doc_extensions = {'.txt', '.md', '.rst', '.pdf', '.doc', '.docx', '.odt', '.rtf'}
        ext = Path(file_path).suffix.lower()
        return ext in doc_extensions


    def _format_size(self, size_bytes: int) -> str:
        """Format file size for human readability."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"


    def add_file(self, file_path: str) -> bool:
        """
        Add a single file to the vector database.
    
        Args:
            file_path: Path to the file to add
    
        Returns:
            True if successful, False otherwise
        """
        try:
            # Process the file
            doc = self.process_file(file_path)
        
            if doc:
                # Add to vector store
                self.vector_store.add_documents([doc])
            
                # Track statistics
                if doc.metadata.get('embedding_type') == 'bert':
                    print(f"   🧠 BERT embedded")
                else:
                    print(f"   🔧 Ollama embedded")
            
                return True
            else:
                # Empty file or unsupported
                return False
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False


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
