#!/usr/bin/env python3
"""
Loader for JSON files.

This loader extracts data from JSON files and converts them to a searchable
text format. It can handle nested structures, arrays, and large files.
Supports both pretty-printed and minified JSON.
"""

import os
import json
from typing import Optional, Set, Dict, Any, List, Union
from langchain_core.documents import Document
from .base import DocumentLoader


class JSONLoader(DocumentLoader):
    """Loader for JSON files - supports nested structures and large files."""

    def _get_supported_extensions(self) -> Set[str]:
        return {'.json', '.jsonl', '.geojson'}

    def load_document(self, file_path: str) -> Optional[Document]:
        """Load and parse JSON file."""
        try:
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext == '.jsonl':
                return self._load_jsonl(file_path)
            else:
                return self._load_json(file_path)
                
        except Exception as e:
            print(f"Error loading JSON file {file_path}: {e}")
            return None

    def _load_json(self, file_path: str) -> Optional[Document]:
        """Load standard JSON file."""
        try:
            file_size = os.path.getsize(file_path)
            
            # For large files (>10MB), use streaming parser
            if file_size > 10 * 1024 * 1024:
                return self._load_large_json(file_path)
            
            # For smaller files, load entirely
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert JSON to searchable text
            content, metadata = self._parse_json_data(data, file_path)
            
            return Document(page_content=content, metadata=metadata)
            
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in {file_path}: {e}")
            return None
        except Exception as e:
            print(f"Error loading JSON file {file_path}: {e}")
            return None

    def _load_large_json(self, file_path: str) -> Optional[Document]:
        """Stream large JSON files to avoid memory issues."""
        try:
            import ijson
            
            content_parts = []
            metadata = {
                "filename": os.path.basename(file_path),
                "filepath": file_path,
                "type": "json",
                "extension": ".json",
                "size": os.path.getsize(file_path),
                "match_source": "file_content",
                "embedding_type": "ollama",
                "is_large_file": 1,
                "parsing_method": "streaming"
            }
            
            # Count top-level items
            item_count = 0
            with open(file_path, 'rb') as f:
                parser = ijson.parse(f)
                for prefix, event, value in parser:
                    if event == 'map_key' and prefix == '':
                        item_count += 1
            
            metadata["top_level_items"] = item_count
            content_parts.append(f"JSON file with {item_count} top-level items\n")
            
            # Extract key information
            with open(file_path, 'rb') as f:
                # Get first few items as sample
                items = ijson.items(f, 'item')
                sample_count = 0
                for item in items:
                    if sample_count < 10:  # Sample first 10 items
                        item_str = json.dumps(item, indent=2, ensure_ascii=False)
                        content_parts.append(f"\nItem {sample_count + 1}:\n{item_str[:500]}")
                        sample_count += 1
                    else:
                        content_parts.append(f"\n... and {item_count - sample_count} more items")
                        break
            
            content = "\n".join(content_parts)
            return Document(page_content=content, metadata=metadata)
            
        except ImportError:
            # Fallback to regular loading if ijson not available
            print(f"  ⚠️  ijson not installed for streaming large JSON. Falling back to regular load.")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            content, metadata = self._parse_json_data(data, file_path)
            metadata["is_large_file"] = 1
            metadata["parsing_method"] = "full_load"
            return Document(page_content=content, metadata=metadata)
        except Exception as e:
            print(f"Error streaming large JSON {file_path}: {e}")
            return None

    def _load_jsonl(self, file_path: str) -> Optional[Document]:
        """Load JSONL (JSON Lines) format - one JSON object per line."""
        try:
            content_parts = []
            line_count = 0
            total_size = 0
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        line_count += 1
                        
                        # Format each line's content
                        if line_count <= 100:  # Limit to first 100 lines for content
                            line_content = json.dumps(data, indent=2, ensure_ascii=False)
                            content_parts.append(f"\n[Line {line_num}]:\n{line_content[:500]}")
                        elif line_count == 101:
                            content_parts.append(f"\n... and {line_count - 100} more lines")
                        
                        total_size += len(line)
                        
                    except json.JSONDecodeError as e:
                        print(f"  ⚠️  Invalid JSON at line {line_num}: {e}")
                        continue
            
            content = f"JSONL file with {line_count} lines\n" + "\n".join(content_parts)
            
            metadata = {
                "filename": os.path.basename(file_path),
                "filepath": file_path,
                "type": "jsonl",
                "extension": ".jsonl",
                "size": os.path.getsize(file_path),
                "match_source": "file_content",
                "embedding_type": "ollama",
                "line_count": line_count,
                "total_chars": total_size
            }
            
            return Document(page_content=content, metadata=metadata)
            
        except Exception as e:
            print(f"Error loading JSONL file {file_path}: {e}")
            return None

    def _parse_json_data(self, data: Any, file_path: str) -> tuple[str, Dict[str, Any]]:
        """Parse JSON data and extract content and metadata."""
        
        # Determine JSON structure type
        structure_type = self._detect_structure_type(data)
        
        # Convert to searchable text
        content = self._json_to_text(data, max_depth=3)
        
        # Build metadata
        metadata = {
            "filename": os.path.basename(file_path),
            "filepath": file_path,
            "type": "json",
            "extension": ".json",
            "size": os.path.getsize(file_path),
            "match_source": "file_content",
            "embedding_type": "ollama",
            "structure_type": structure_type,
            "is_large_file": 0,
            "parsing_method": "full_load"
        }
        
        # Add structure-specific metadata
        if structure_type == "array":
            metadata["item_count"] = len(data) if isinstance(data, list) else 0
            if len(data) > 0:
                metadata["first_item_type"] = type(data[0]).__name__
                
        elif structure_type == "object":
            metadata["key_count"] = len(data) if isinstance(data, dict) else 0
            # Store top-level keys as comma-separated string (Chromadb compatible)
            if isinstance(data, dict) and len(data) <= 20:
                metadata["top_keys"] = ",".join(list(data.keys())[:20])
            elif isinstance(data, dict):
                metadata["top_keys"] = ",".join(list(data.keys())[:10]) + ",..."
                
        elif structure_type == "nested":
            metadata["nesting_depth"] = self._get_nesting_depth(data)
            if isinstance(data, dict):
                metadata["key_count"] = len(data)
                # Store first few keys
                metadata["top_keys"] = ",".join(list(data.keys())[:10])
        
        return content, metadata

    def _detect_structure_type(self, data: Any) -> str:
        """Detect the structure type of JSON data."""
        if isinstance(data, list):
            # Check if it's an array of objects
            if len(data) > 0 and isinstance(data[0], (dict, list)):
                return "array_of_objects"
            return "array"
        elif isinstance(data, dict):
            # Check if it's nested
            depth = self._get_nesting_depth(data)
            if depth > 2:
                return "nested"
            return "object"
        else:
            return "primitive"

    def _get_nesting_depth(self, data: Any, current_depth: int = 1) -> int:
        """Calculate maximum nesting depth of JSON structure."""
        if isinstance(data, dict):
            if not data:
                return current_depth
            return max(self._get_nesting_depth(v, current_depth + 1) for v in data.values())
        elif isinstance(data, list):
            if not data:
                return current_depth
            return max(self._get_nesting_depth(item, current_depth + 1) for item in data)
        else:
            return current_depth

    def _json_to_text(self, data: Any, prefix: str = "", max_depth: int = 3, current_depth: int = 0) -> str:
        """Convert JSON to readable text format for search."""
        if current_depth >= max_depth:
            return f"{prefix}[Deep nesting truncated]\n"
        
        lines = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    lines.append(self._json_to_text(value, prefix + "  ", max_depth, current_depth + 1))
                else:
                    # Format primitive values
                    value_str = self._format_value(value)
                    lines.append(f"{prefix}{key}: {value_str}")
                    
        elif isinstance(data, list):
            for idx, item in enumerate(data[:20]):  # Limit to first 20 items
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}[{idx}]:")
                    lines.append(self._json_to_text(item, prefix + "  ", max_depth, current_depth + 1))
                else:
                    value_str = self._format_value(item)
                    lines.append(f"{prefix}[{idx}]: {value_str}")
            
            if len(data) > 20:
                lines.append(f"{prefix}... and {len(data) - 20} more items")
                
        else:
            # Primitive value
            lines.append(f"{prefix}{self._format_value(data)}")
        
        return "\n".join(lines)

    def _format_value(self, value: Any) -> str:
        """Format JSON values for display."""
        if value is None:
            return "null"
        elif isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, str):
            # Truncate long strings
            if len(value) > 200:
                return f'"{value[:200]}..."'
            return f'"{value}"'
        else:
            return str(value)

    def extract_json_paths(self, file_path: str, paths: List[str]) -> Optional[Dict[str, Any]]:
        """
        Extract specific JSON paths from a file.
        
        Args:
            file_path: Path to JSON file
            paths: List of JSON paths (e.g., ["data.users[0].name", "metadata.version"])
        
        Returns:
            Dictionary of extracted values
        """
        try:
            import jsonpath_ng
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            results = {}
            for path_str in paths:
                try:
                    jsonpath_expr = jsonpath_ng.parse(path_str)
                    matches = jsonpath_expr.find(data)
                    if matches:
                        results[path_str] = [match.value for match in matches]
                    else:
                        results[path_str] = None
                except Exception as e:
                    print(f"  ⚠️  Error parsing path '{path_str}': {e}")
                    results[path_str] = None
            
            return results
            
        except ImportError:
            print("  ⚠️  jsonpath-ng not installed. Install with: pip install jsonpath-ng")
            return None
        except Exception as e:
            print(f"Error extracting paths from {file_path}: {e}")
            return None


if __name__ == "__main__":
    # Self-test
    import tempfile
    
    print("JSONLoader self-test:")
    loader = JSONLoader()
    
    # Test can_handle
    print(f"  can_handle('test.json'): {loader.can_handle('test.json')}")
    print(f"  can_handle('test.jsonl'): {loader.can_handle('test.jsonl')}")
    print(f"  can_handle('test.txt'): {loader.can_handle('test.txt')}")
    
    # Test 1: Simple JSON object
    test_data = {
        "name": "Test Document",
        "version": "1.0",
        "metadata": {
            "author": "John Doe",
            "created": "2024-01-01",
            "tags": ["test", "json", "loader"]
        },
        "data": [1, 2, 3, 4, 5]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(test_data, f, indent=2)
        temp_path = f.name
    
    try:
        doc = loader.load_document(temp_path)
        if doc:
            print("  Test 1 (simple JSON): SUCCESS")
            print(f"    Content length: {len(doc.page_content)} chars")
            print(f"    Structure type: {doc.metadata.get('structure_type')}")
            print(f"    Key count: {doc.metadata.get('key_count')}")
            print(f"\n  Content preview:")
            print("    " + doc.page_content[:200].replace('\n', '\n    '))
        else:
            print("  Test 1 (simple JSON): FAILED")
    finally:
        os.unlink(temp_path)
    
    # Test 2: JSON array
    test_array = [
        {"id": 1, "name": "Item 1", "value": 100},
        {"id": 2, "name": "Item 2", "value": 200},
        {"id": 3, "name": "Item 3", "value": 300}
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(test_array, f, indent=2)
        temp_path2 = f.name
    
    try:
        doc = loader.load_document(temp_path2)
        if doc:
            print("\n  Test 2 (JSON array): SUCCESS")
            print(f"    Structure type: {doc.metadata.get('structure_type')}")
            print(f"    Item count: {doc.metadata.get('item_count')}")
        else:
            print("  Test 2 (JSON array): FAILED")
    finally:
        os.unlink(temp_path2)
    
    # Test 3: JSONL format
    jsonl_content = """{"id": 1, "name": "First"}
{"id": 2, "name": "Second"}
{"id": 3, "name": "Third"}"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
        f.write(jsonl_content)
        temp_path3 = f.name
    
    try:
        doc = loader.load_document(temp_path3)
        if doc:
            print("\n  Test 3 (JSONL): SUCCESS")
            print(f"    Type: {doc.metadata.get('type')}")
            print(f"    Line count: {doc.metadata.get('line_count')}")
        else:
            print("  Test 3 (JSONL): FAILED")
    finally:
        os.unlink(temp_path3)
    
    print("\n  For advanced features, install optional dependencies:")
    print("    pip install ijson      # For streaming large JSON files")
    print("    pip install jsonpath-ng # For JSON path extraction")