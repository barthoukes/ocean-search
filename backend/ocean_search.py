#!/usr/bin/env python3
"""
file_search.py
Interactive tool to fill and query a vector database of files.
Commands:
  fill <path>   - add documents from <path> to the database
  clear         - delete all documents from the database
  <query>       - search for files matching the query
  q             - quit
"""

import os
import argparse
import sys
import shutil
import time
from typing import List, Set
from langchain_chroma import Chroma
# Check if BERT is available
try:
    from embedders import BERT_AVAILABLE
except ImportError:
    BERT_AVAILABLE = False

from document_processor import DocumentProcessor
from query_matcher import QueryMatcher


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


def main():
    parser = argparse.ArgumentParser(description="Interactive document search with Ollama embeddings and optional BERT for text.")
    parser.add_argument("--db_path", default="./chroma_db", help="Path to Chroma DB")
    parser.add_argument("--embed_model", default="nomic-embed-text-v2-moe", help="Ollama embedding model (fallback)")
    parser.add_argument("--extensions", nargs="+", default=None,
                        help="File extensions to include when filling (e.g., .txt .pdf .py)")
    parser.add_argument("--k", type=int, default=10, help="Number of results to return per query")
    parser.add_argument("--verbose", action="store_true", help="Show full content preview")
    parser.add_argument("--no-bert", action="store_true", help="Disable BERT for text files (use Ollama only)")
    parser.add_argument("--no-color", action="store_true", help="Disable colored highlighting")
    parser.add_argument("--force", action="store_true", help="Skip confirmation when clearing database")
    args = parser.parse_args()
    
    # Disable colors if requested
    if args.no_color:
        os.environ['NO_COLOR'] = '1'
    
    # Create document processor
    processor = DocumentProcessor(
        args.db_path, 
        args.embed_model, 
        args.extensions,
        use_bert=not args.no_bert
    )
    
    # Get all supported extensions
    all_extensions: Set[str] = set()
    for loader in processor.loaders:
        all_extensions.update(loader.supported_extensions)
    
    print("🔍 Interactive Document Search with Smart Embedding")
    print("=" * 60)
    print("Commands:")
    print("  fill <path>   - add documents from <path> to the database")
    print("  clear         - delete ALL documents from the database")
    print("  stats         - show database statistics")
    print("  <query>       - search for documents matching the query")
    print("  q             - quit")
    print("\nSupported extensions: " + ", ".join(sorted(all_extensions)))
    
    if not args.no_bert and BERT_AVAILABLE:
        print("\n🧠 BERT enabled: Natural language text files (.txt, .md, .rst) use BERT embeddings")
        print("   Code files, PDFs, images, markup files use Ollama embeddings")
    else:
        print("\n🔧 Using Ollama for all file types")
    
    # Show database stats if it exists
    if os.path.exists(args.db_path):
        print(f"\n📁 Database exists at: {os.path.abspath(args.db_path)}")
        print("   Use 'clear' to delete all documents and start fresh")
        print("   Use 'stats' to see document count")
    else:
        print(f"\n📁 New database will be created at: {os.path.abspath(args.db_path)}")
    
    print("\n💡 Tip: Results show highlighted matching text to verify relevance")
    print("   🟢 Green = exact phrase match, 🟡 Yellow = keyword match")
    print()
    
    while True:
        try:
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
            
            # Check for stats command
            elif user_input.lower() == 'stats':
                try:
                    # Try to get collection count
                    if hasattr(processor.vector_store, '_collection'):
                        try:
                            count = processor.vector_store._collection.count()
                            print(f"\n📊 Database Statistics:")
                            print(f"   📁 Path: {os.path.abspath(args.db_path)}")
                            print(f"   📄 Documents: {count}")
                            print(f"   🧠 Embedder: {'BERT + Ollama' if not args.no_bert else 'Ollama only'}")
                        except Exception as e:
                            print(f"   Could not get count: {e}")
                    else:
                        print("\n📊 Database exists but unable to get statistics")
                except Exception as e:
                    print(f"\n❌ Error getting stats: {e}")
            
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
                results = processor.search(user_input, k=args.k)
                
                if not results:
                    print("   No results found.")
                    continue
                
                print(f"\n📋 Top {len(results)} results for: \"{user_input}\"")
                print("=" * 60)
                
                for i, (doc, snippets) in enumerate(results, 1):
                    print(f"\n{i}. {doc.metadata.get('filename', 'N/A')}")
                    print(f"   📂 Path: {doc.metadata.get('filepath', 'N/A')}")
                    print(f"   🏷️  Type: {doc.get_file_type_display()}")
                    print(f"   🔍 Match: {doc.get_match_type_display()}")
                    
                    # Show embedding type
                    embedder = doc.metadata.get('embedding_type', 'ollama')
                    if embedder == 'bert':
                        print(f"   🧠 Embedded with: BERT (semantic search)")
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
                    
                    # Show matching snippets with highlighting
                    if snippets:
                        print(f"\n   🎯 Matching Text Snippets:")
                        for j, snippet_info in enumerate(snippets[:3], 1):
                            snippet = snippet_info['snippet']
                            matched_terms = snippet_info['matched_terms']
                            match_type = snippet_info.get('match_type', 'keywords')
                            
                            # Highlight the snippet
                            if not args.no_color:
                                highlighted = QueryMatcher.highlight_text(snippet, user_input)
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
                        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Exiting.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("   Please try again or use 'q' to quit.")


if __name__ == "__main__":
    main()
    
