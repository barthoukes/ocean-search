#!/usr/bin/env python3
"""
file_search.py
Interactive tool to fill and query a vector database of files.
Commands:
  fill <path> --exclude <path names>  - add documents from <path> to the database
  clear         - delete all documents from the database
  <query>       - search for files matching the query
  n, next       - next page of results
  p, prev       - previous page of results
  q             - quit
"""

import os
import argparse
import sys
import shutil
import time
from typing import List, Set, Tuple, Dict, Any
from langchain_chroma import Chroma
# Check if BERT is available
try:
    from embedders import BERT_AVAILABLE
except ImportError:
    BERT_AVAILABLE = False

from document_processor import DocumentProcessor
from query_matcher import QueryMatcher
from datetime import datetime


class SearchState:
    """Store pagination state for the current search"""
    def __init__(self):
        self.query: str = ""
        self.current_page: int = 1
        self.page_size: int = 10
        self.total_results: int = 0
        self.total_pages: int = 0
        self.results: List[Tuple] = []
        self.pagination_info: Dict[str, Any] = {}
        

def clear_database(db_path: str, processor: DocumentProcessor) -> bool:
    """
    Clear the database by deleting all documents and resetting the collection.
    
    Args:
        db_path: Path to the database directory
        processor: DocumentProcessor instance to reset
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Confirm with user
        print(f"\n⚠️  WARNING: This will delete ALL documents from the database at:")
        print(f"   {os.path.abspath(db_path)}")
        response = input("   Are you sure? (yes/no): ").strip().lower()
        
        if response != 'yes':
            print("   Database clear cancelled.")
            return False
        
        print("\n🗑️  Clearing database...")
        
        # Try to close any open connections
        try:
            # For Chroma, we need to delete the collection first
            if hasattr(processor.vector_store, '_collection'):
                try:
                    # Try to delete the collection
                    processor.vector_store._collection.delete()
                    print("   ✓ Deleted document collection")
                except Exception as e:
                    print(f"   ⚠️  Could not delete collection: {e}")
            
            # Try to close any persistent connections
            if hasattr(processor.vector_store, '_client'):
                try:
                    processor.vector_store._client.clear_system_cache()
                except:
                    pass
            
            # Give time for connections to close
            time.sleep(0.5)
            
        except Exception as e:
            print(f"   ⚠️  Note: {e}")
        
        # Delete the database directory
        if os.path.exists(db_path):
            try:
                # Try to remove directory with retries
                for attempt in range(3):
                    try:
                        shutil.rmtree(db_path)
                        print(f"   ✓ Deleted database directory: {db_path}")
                        break
                    except PermissionError:
                        if attempt < 2:
                            print(f"   ⏳ Waiting for file locks to clear...")
                            time.sleep(1)
                        else:
                            raise
            except Exception as e:
                print(f"   ⚠️  Could not delete directory: {e}")
                print(f"   Trying alternative method...")
                # Try to remove files individually
                try:
                    for root, dirs, files in os.walk(db_path, topdown=False):
                        for name in files:
                            try:
                                os.remove(os.path.join(root, name))
                            except:
                                pass
                        for name in dirs:
                            try:
                                os.rmdir(os.path.join(root, name))
                            except:
                                pass
                    os.rmdir(db_path)
                    print(f"   ✓ Deleted database directory (alternative method)")
                except Exception as e2:
                    print(f"   ⚠️  Could not delete directory: {e2}")
        
        # Create new processor with fresh database
        print("   🔄 Reinitializing database...")
        
        # Create a new vector store with the same embedding function
        new_vector_store = Chroma(
            collection_name="document_collection",
            embedding_function=processor.smart_embedder,
            persist_directory=db_path
        )
        
        # Replace the old vector store
        processor.vector_store = new_vector_store
        
        print("\n✅ Database cleared successfully!")
        print("   You can now add new documents with 'fill <path>'")
        return True
        
    except Exception as e:
        print(f"\n❌ Error clearing database: {e}")
        return False


def cmd_fill(path_str, processor):
    """
    Handle fill command: add documents from directory to database
    
    Args:
        path_str: The path string from user input (may include exclusions after --)
        processor: DocumentProcessor instance
    """
    # Parse path and exclusions
    parts = path_str.strip().split()
    
    # First part is the path
    path = parts[0] if parts else None
    
    # Check for -- or --exclude flag for exclusions
    exclude_dirs = []
    if len(parts) > 1:
        if '--' in parts:
            exclude_idx = parts.index('--')
            exclude_dirs = parts[exclude_idx + 1:]
        elif '--exclude' in parts:
            exclude_idx = parts.index('--exclude')
            exclude_dirs = parts[exclude_idx + 1:]
    
    # Validate path
    if not path:
        print("❌ Error: No path specified")
        print("   Usage: fill <path> [--exclude dir1 dir2 ...]")
        return
    
    if not os.path.isdir(path):
        print(f"❌ Error: '{path}' is not a valid directory")
        return
    
    # Default exclusions
    default_excludes = {
        '__pycache__', 'node_modules', '.git', 'venv', '.venv', 'env',
        'dist', 'build', '.angular', '.cache', '.idea', '.vscode',
        'python3.12', 'python3.11', 'python3.10', 'python3.9',
        'lib/python3', 'local/lib/python', '__pycache__'
    }
    
    # Add user exclusions
    all_excludes = default_excludes.union(set(exclude_dirs))
    
    print(f"📂 Scanning: {path}")
    print(f"🚫 Excluding: {', '.join(sorted(all_excludes))}")
    
    # Walk through directory
    files_processed = 0
    files_skipped = 0
    dirs_skipped = 0
    
    for root, dirs, files in os.walk(path):
        # Filter out excluded directories (modify dirs in-place)
        original_dir_count = len(dirs)
        dirs[:] = [d for d in dirs if d not in all_excludes]
        dirs_skipped += (original_dir_count - len(dirs))
        
        for file in files:
            filepath = os.path.join(root, file)
            print(filepath)
 
            # Skip excluded file patterns
            if any(file.endswith(ext) for ext in ['.pyc', '.bak', '.pyo', '.so', '.dll', '.exe']):
                files_skipped += 1
                continue
            
            # Process the file
            try:
                # Call your existing method
                print(f"  📄 {filepath}")
                if processor.add_file(filepath):
                    files_processed += 1
            except Exception as e:
                print(f"  ⚠️  Error processing {file}: {e}")
                files_skipped += 1
    
    # Print summary
    print(f"\n✅ Fill completed!")
    print(f"   📁 Files processed: {files_processed}")
    if files_skipped > 0:
        print(f"   ⏭️  Files skipped: {files_skipped}")
    if dirs_skipped > 0:
        print(f"   📁 Directories skipped: {dirs_skipped}")


def display_search_results(results, query, args, pagination_info=None):
    """
    Display search results with pagination information.
    
    Args:
        results: List of (doc, snippets) tuples
        query: Search query string
        args: Command line arguments
        pagination_info: Dictionary with pagination metadata
    """
    if not results:
        print("   No results found.")
        return
    
    if pagination_info:
        print(f"\n📋 Page {pagination_info['current_page']}/{pagination_info['total_pages']} - {pagination_info['total_results']} total results")
    else:
        print(f"\n📋 Top {len(results)} results for: \"{query}\"")
    
    print("=" * 60)
    
    for i, (doc, snippets) in enumerate(results, 1):
        # Calculate global result number if paginated
        if pagination_info:
            global_num = pagination_info['start_index'] + i - 1
            print(f"\n{global_num}. {doc.metadata.get('filename', 'N/A')}")
        else:
            print(f"\n{i}. {doc.metadata.get('filename', 'N/A')}")
        
        print(f"   📂 Path: {doc.metadata.get('filepath', 'N/A')}")
        print(f"   🏷️  Type: {doc.get_file_type_display()}")
        
        # Display file size information
        size_bytes = doc.metadata.get('size_bytes', 0)
        size_human = doc.metadata.get('size_human', 'N/A')
        size_category = doc.metadata.get('size_category', 'unknown')
        
        # Choose icon based on size category
        size_icons = {
            'tiny': '🔹', 'very_small': '📄', 'small': '📄',
            'medium': '📘', 'large': '📚', 'very_large': '📦'
        }
        size_icon = size_icons.get(size_category, '📄')
        
        print(f"   💾 Size: {size_human} ({size_bytes:,} bytes) {size_icon} [{size_category}]")
        
        # Display date information
        created_date = doc.metadata.get('created_date_human', doc.metadata.get('created_date', 'N/A'))
        modified_date = doc.metadata.get('modified_date_human', doc.metadata.get('modified_date', 'N/A'))
        age_days = doc.metadata.get('age_days', 0)
        is_recent = doc.metadata.get('is_recent', False)
        is_old = doc.metadata.get('is_old', False)
        
        # Choose age indicator
        if is_recent:
            age_icon = "🆕"
        elif is_old:
            age_icon = "📅"
        else:
            age_icon = "📆"
        
        print(f"   🕐 Created: {created_date}")
        print(f"   🕐 Modified: {modified_date}")
        print(f"   ⏰ Age: {age_days:.1f} days old {age_icon}")
        
        # Display content statistics if available
        word_count = doc.metadata.get('word_count', 0)
        if word_count > 0:
            char_count = doc.metadata.get('char_count', 0)
            line_count = doc.metadata.get('line_count', 0)
            print(f"   📝 Content: {word_count:,} words, {char_count:,} chars, {line_count:,} lines")
        
        # Display complexity for code files
        complexity = doc.metadata.get('complexity_score', 0)
        if complexity > 0:
            complexity_stars = "⭐" * min(5, int(complexity / 20))
            print(f"   🔧 Complexity: {complexity:.1f} {complexity_stars}")
        
        print(f"   🔍 Match: {doc.get_match_type_display()}")
        
        # Show embedding type
        embedder = doc.metadata.get('embedding_type', 'ollama')
        if embedder == 'bert':
            print(f"   🧠 Embedded with: BERT (semantic search)")
        else:
            print(f"   🔧 Embedded with: Ollama")
        
        # Additional metadata based on file type
        if doc.metadata.get('type') == 'code':
            print(f"   📊 Code Stats: {doc.metadata.get('lines_code', 0)} lines of code, "
                  f"{doc.metadata.get('lines_comments', 0)} comments")
            print(f"   🎯 Lexical Diversity: {doc.metadata.get('lexical_diversity', 0):.2f}")
        
        elif doc.metadata.get('type') == 'pdf':
            if doc.metadata.get('has_text_content', False):
                print(f"   📄 Contains {doc.metadata.get('pages', 0)} pages with extractable text")
            else:
                print(f"   ⚠️  PDF has {doc.metadata.get('pages', 0)} pages but no extractable text")
        elif doc.metadata.get('type') == 'image':
            print(f"   📐 Dimensions: {doc.metadata.get('width', '?')}x{doc.metadata.get('height', '?')}")
        
        # Show matching snippets with highlighting
        if snippets:
            print(f"\n   🎯 Matching Text Snippets:")
            for j, snippet_info in enumerate(snippets[:3], 1):
                snippet = snippet_info['snippet']
                matched_terms = snippet_info['matched_terms']
                match_type = snippet_info.get('match_type', 'keywords')
                
                # Highlight the snippet
                if not args.no_color:
                    highlighted = QueryMatcher.highlight_text(snippet, query)
                else:
                    highlighted = snippet
                
                # Clean up newlines
                highlighted = highlighted.replace('\n', ' ')
                
                # Show match info
                match_icon = "🔍" if match_type == 'exact_phrase' else "📌"
                terms_display = ", ".join(matched_terms[:5])
                print(f"      {match_icon} [{j}] {highlighted}")
                print(f"          Matched: {terms_display}")
                print()
        else:
            # If no snippets found, show a preview
            if args.verbose:
                preview = doc.page_content[:200].replace('\n', ' ')
            else:
                preview = doc.page_content[:150].replace('\n', ' ')
            if preview:
                print(f"\n   📝 Content Preview: {preview}...")
        
        # Add separator
        if i < len(results):
            print("   " + "-" * 50)
    
    # Show pagination controls
    if pagination_info:
        print("\n" + "=" * 60)
        print(f"Showing {pagination_info['start_index']}-{pagination_info['end_index']} of {pagination_info['total_results']} results")
        
        controls = []
        if pagination_info['has_prev']:
            controls.append("← Previous page (p, prev, pageup)")
        if pagination_info['has_next']:
            controls.append("→ Next page (n, next, pagedown)")
        
        if controls:
            print("   " + "  |  ".join(controls))
        print("   Type 'q' to quit, or enter a new search query")

        
def perform_search(processor, query, page, page_size, args):
    """
    Perform a search with pagination.
    
    Returns:
        Tuple of (results, pagination_info)
    """
    offset = (page - 1) * page_size
    
    # Use the search method with offset
    results = processor.search(
        query=query,
        k=page_size,
        offset=offset,
        min_content_length=10,
        score_threshold=0.1,
        max_total=1000
    )
    
    # Get total count by doing a separate search without offset
    all_results = processor.search(
        query=query,
        k=1000,
        offset=0,
        min_content_length=10,
        score_threshold=0.1,
        max_total=1000
    )
    total_count = len(all_results)
    total_pages = (total_count + page_size - 1) // page_size if total_count > 0 else 1
    
    pagination_info = {
        'current_page': page,
        'page_size': page_size,
        'total_results': total_count,
        'total_pages': total_pages,
        'has_next': page < total_pages,
        'has_prev': page > 1,
        'start_index': offset + 1 if results else 0,
        'end_index': min(offset + page_size, total_count) if results else 0
    }
    
    return results, pagination_info


def show_database_stats(processor: DocumentProcessor, db_path):
    """Display detailed database statistics including file sizes and dates."""
    print("\n" + "=" * 60)
    print("📊 DATABASE STATISTICS")
    print("=" * 60)
    
    try:
        # Get all documents (you might need to implement a method to get all docs)
        # For now, we'll do a broad search
        all_results = processor.search(
            query="",
            k=1000,
            offset=0,
            score_threshold=0.0,  # Very low threshold to get everything
            max_total=10000
        )
        
        if not all_results:
            print("No documents in database.")
            return
        
        total_files = len(all_results)
        total_size_bytes = 0
        oldest_file = None
        newest_file = None
        oldest_date = float('inf')
        newest_date = 0
        
        size_categories = {}
        file_types = {}
        
        for doc, _ in all_results:
            # Size statistics
            size_bytes = doc.metadata.get('size_bytes', 0)
            total_size_bytes += size_bytes
            
            size_cat = doc.metadata.get('size_category', 'unknown')
            size_categories[size_cat] = size_categories.get(size_cat, 0) + 1
            
            # File type statistics
            file_type = doc.metadata.get('type', 'unknown')
            file_types[file_type] = file_types.get(file_type, 0) + 1
            
            # Date statistics
            created_time = doc.metadata.get('created_time', 0)
            if created_time and created_time < oldest_date:
                oldest_date = created_time
                oldest_file = doc.metadata.get('filename', 'N/A')
            if created_time and created_time > newest_date:
                newest_date = created_time
                newest_file = doc.metadata.get('filename', 'N/A')
        
        # Display results
        print(f"\n📁 Total files: {total_files}")
        print(f"💾 Total size: {format_size(total_size_bytes)}")
        print(f"📊 Average size: {format_size(total_size_bytes / total_files)}")
        
        print(f"\n📏 Size Distribution:")
        for category in ['tiny', 'very_small', 'small', 'medium', 'large', 'very_large']:
            count = size_categories.get(category, 0)
            if count > 0:
                bar = "█" * min(40, count)
                print(f"   {category:12} : {count:4} files {bar}")
        
        print(f"\n📄 File Type Distribution:")
        for ftype, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_files) * 100
            bar = "█" * min(40, int(percentage))
            print(f"   {ftype:12} : {count:4} files ({percentage:.1f}%) {bar}")
        
        if oldest_file:
            oldest_date_str = datetime.fromtimestamp(oldest_date).strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n🕐 Oldest file: {oldest_file} ({oldest_date_str})")
        
        if newest_file:
            newest_date_str = datetime.fromtimestamp(newest_date).strftime('%Y-%m-%d %H:%M:%S')
            print(f"🕐 Newest file: {newest_file} ({newest_date_str})")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"❌ Error getting statistics: {e}")


def format_size(size_bytes):
    """Format file size for human readability."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def main():
    parser = argparse.ArgumentParser(description="Interactive document search with Ollama embeddings and optional BERT for text.")
    parser.add_argument("--db_path", default="./chroma_db", help="Path to Chroma DB")
    #parser.add_argument("--embed_model", default="nomic-embed-text-v2-moe", help="Ollama embedding model (fallback)")
    parser.add_argument("--embed_model", default="embeddinggemma:300m", help="Ollama embedding model (fallback)")
    parser.add_argument("--extensions", nargs="+", default=None,
                        help="File extensions to include when filling (e.g., .txt .pdf .py)")
    parser.add_argument("--k", type=int, default=10, help="Number of results to return per page")
    parser.add_argument("--verbose", action="store_true", help="Show full content preview")
    parser.add_argument("--no-bert", action="store_true", help="Disable BERT for text files (use Ollama only)")
    parser.add_argument("--no-color", action="store_true", help="Disable colored highlighting")
    parser.add_argument("--force", action="store_true", help="Skip confirmation when clearing database")
    parser.add_argument("--stats", action="store_true", help="Show database statistics and exit")
    args = parser.parse_args()
    
    # Disable colors if requested
    if args.no_color:
        os.environ['NO_COLOR'] = '1'
    
    # Create document processor
    print("Create document processor")
    processor = DocumentProcessor(
        args.db_path, 
        args.embed_model, 
        args.extensions,
        use_bert=not args.no_bert
    )
    # Check if the embedding model is available in Ollama
    print("\n🔍 Checking embedding model availability...")
    if not check_and_load_embedding_model(args.embed_model, processor):
        print("\n❌ Failed to find a suitable embedding model. Exiting.")
        print("   Please install at least one embedding model:")
        print("   ollama pull nomic-embed-text")
        sys.exit(1)
    
    # Check if stats flag is set (do this before showing interactive UI)
    if args.stats:
        show_database_stats(processor, args.db_path)
        return
    
    # Get all supported extensions
    all_extensions: Set[str] = set()
    for loader in processor.loaders:
        all_extensions.update(loader.supported_extensions)
    
    print("🔍 Interactive Document Search with Smart Embedding")
    print("=" * 60)
    print("Commands:")
    print("  fill <path> --exclude <paths>  - add documents from <path> to database")
    print("  clear         - delete ALL documents from the database")
    print("  stats         - show database statistics")
    print("  <query>       - search for documents matching the query")
    print("  n, next       - next page of results")
    print("  p, prev       - previous page of results")
    print("  q             - quit")
    print("\nSupported extensions: " + ", ".join(sorted(all_extensions)))
    
    if not args.no_bert and BERT_AVAILABLE:
        print("\n🧠 BERT enabled: Natural language text files (.txt, .md, .rst) use BERT embeddings")
        print("   Code files, PDFs, images, markup files use Ollama embeddings")
    else:
        print("\n🔧 Using Ollama for all file types")
    
    # Show current embedding model
    print(f"\n📌 Current Ollama embedding model: {processor.embedding_model}")
    
    # Show database stats if it exists
    if os.path.exists(args.db_path):
        print(f"\n📁 Database exists at: {os.path.abspath(args.db_path)}")
        print("   Use 'clear' to delete all documents and start fresh")
        print("   Use 'stats' to see document count")
    else:
        print(f"\n📁 New database will be created at: {os.path.abspath(args.db_path)}")
    
    print("\n💡 Tip: Results show highlighted matching text to verify relevance")
    print("   🟢 Green = exact phrase match, 🟡 Yellow = keyword match")
    print("   Use 'n' for next page, 'p' for previous page")
    print()
    
    # Initialize search state
    search_state = SearchState()
    
    while True:
        try:
            user_input = input("\n> ").strip()
            if not user_input:
                continue
            
            # Quit command
            if user_input.lower() == 'q':
                print("Exiting.")
                break
            
            # Next page commands
            if user_input.lower() in ['n', 'next', 'pagedown']:
                if search_state.query and search_state.pagination_info.get('has_next', False):
                    search_state.current_page += 1
                    results, pagination_info = perform_search(
                        processor, search_state.query, 
                        search_state.current_page, args.k, args
                    )
                    search_state.results = results
                    search_state.pagination_info = pagination_info
                    display_search_results(results, search_state.query, args, pagination_info)
                else:
                    print("   No next page available. Perform a search first.")
                continue
            
            # Previous page commands
            if user_input.lower() in ['p', 'prev', 'pageup']:
                if search_state.query and search_state.pagination_info.get('has_prev', False):
                    search_state.current_page -= 1
                    results, pagination_info = perform_search(
                        processor, search_state.query, 
                        search_state.current_page, args.k, args
                    )
                    search_state.results = results
                    search_state.pagination_info = pagination_info
                    display_search_results(results, search_state.query, args, pagination_info)
                else:
                    print("   No previous page available.")
                continue
            
            # Check for fill command
            if user_input.lower().startswith('fill '):
                cmd_fill(user_input[5:], processor)
                continue
 
            # Check for stats command
            elif user_input.lower() == 'stats':
                show_database_stats(processor, args.db_path)
                continue
            
            # Check for clear command
            elif user_input.lower() == 'clear':
                if args.force:
                    # Skip confirmation if force flag is set
                    print("\n🗑️  Clearing database (force mode)...")
                    try:
                        # Try to close connections
                        if hasattr(processor.vector_store, '_client'):
                            try:
                                processor.vector_store._client.clear_system_cache()
                            except:
                                pass
                        
                        # Give time for connections to close
                        time.sleep(0.5)
                        
                        # Delete the collection
                        if hasattr(processor.vector_store, '_collection'):
                            try:
                                processor.vector_store._collection.delete()
                            except:
                                pass
                        
                        # Delete the database directory
                        if os.path.exists(args.db_path):
                            shutil.rmtree(args.db_path, ignore_errors=True)
                        
                        # Recreate processor with fresh database
                        processor.vector_store = Chroma(
                            collection_name="document_collection",
                            embedding_function=processor.smart_embedder,
                            persist_directory=args.db_path
                        )
                        print("✅ Database cleared successfully!")
                    except Exception as e:
                        print(f"❌ Error clearing database: {e}")
                else:
                    clear_database(args.db_path, processor)
            
            else:
                # Treat as query
                search_state.query = user_input
                search_state.current_page = 1
                
                results, pagination_info = perform_search(
                    processor, user_input, 1, args.k, args
                )
                
                search_state.results = results
                search_state.pagination_info = pagination_info
                
                display_search_results(results, user_input, args, pagination_info)
                        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Exiting.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("   Please try again or use 'q' to quit.")


if __name__ == "__main__":
    main()
    
