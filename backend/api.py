#!/usr/bin/env python3
"""
api.py - Add this to your project
Flask part for ocean-search.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from document_processor import DocumentProcessor
import os

app = Flask(__name__)
CORS(app)  # Allow Angular to call this API

# Initialize your existing processor
processor = DocumentProcessor(
    db_path="./chroma_db",
    # embed_model="nomic-embed-text-v2-moe",
    # embeddinggemma:300m has issues with multiple spaces.
    embed_model='bge-m3',
    use_bert=True
)

@app.route('/api/search', methods=['POST'])
def search():
    """Angular will call this endpoint"""
    data = request.json
    query = data.get('query', '')
    k = data.get('k', 100)
    
    # YOUR EXISTING SEARCH LOGIC - unchanged!
    results = processor.search(query, k=k)
    
    # Convert results to JSON for Angular
    formatted_results = []
    for doc, snippets in results:
        formatted_results.append({
            'filename': doc.metadata.get('filename', 'N/A'),
            'filepath': doc.metadata.get('filepath', 'N/A'),
            'content_preview': doc.page_content[:300],
            'match_type': doc.get_match_type_display(),
            'embedding_type': doc.metadata.get('embedding_type', 'ollama'),
            'snippets': snippets[:3]  # Send top 3 snippets
        })
    
    return jsonify(formatted_results)

@app.route('/api/stats', methods=['GET'])
def stats():
    """Get database statistics"""
    if hasattr(processor.vector_store, '_collection'):
        count = processor.vector_store._collection.count()
        return jsonify({
            'document_count': count,
            'db_path': os.path.abspath("./chroma_db")
        })
    return jsonify({'document_count': 0})

@app.route('/api/fill', methods=['POST'])
def fill():
    """Add documents to database"""
    data = request.json
    path = data.get('path', '')
    if os.path.isdir(path):
        processor.add_files_from_directory(path)
        return jsonify({'status': 'success', 'message': f'Added files from {path}'})
    return jsonify({'status': 'error', 'message': 'Invalid directory path'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)

