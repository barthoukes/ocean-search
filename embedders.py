"""
Embedding classes for document search.
Supports BERT for text files and Ollama for other file types.
"""

import numpy as np
from typing import List, Optional, Union
from langchain_ollama import OllamaEmbeddings

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
        """Generate embeddings for multiple documents."""
        if not self.is_available or self.model is None:
            return None
        
        try:
            # Truncate very long texts to avoid memory issues (BERT has 512 token limit)
            truncated_texts = [text[:100000] for text in texts]
            embeddings = self.model.encode(truncated_texts, convert_to_numpy=True)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            print(f"Error generating BERT embeddings: {e}")
            return None
    
    def embed_query(self, query: str) -> Optional[List[float]]:
        """Embed a query string."""
        if not self.is_available or self.model is None:
            return None
        
        try:
            embedding = self.model.encode([query], convert_to_numpy=True)[0]
            return embedding.tolist()
        except Exception as e:
            print(f"Error generating BERT query embedding: {e}")
            return None


class SmartEmbedder:
    """Smart embedder that uses BERT only for natural language text files."""
    
    BERT_FILE_TYPES = {'text'}  # Only text files (.txt, .md, .rst, etc.)
    
    def __init__(self, ollama_embeddings: OllamaEmbeddings, use_bert: bool = True):
        """
        Initialize smart embedder.
        
        Args:
            ollama_embeddings: OllamaEmbeddings instance
            use_bert: Whether to use BERT for text files
        """
        self.ollama = ollama_embeddings
        self.bert = BERTEmbedder() if use_bert else None
        self.use_bert = use_bert and self.bert and self.bert.is_available
        
        if self.use_bert:
            print("🎯 Smart Embedder: Using BERT for natural language text files, Ollama for code and other files")
        else:
            print("📝 Smart Embedder: Using Ollama for all file types")
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query using Ollama for consistency.
        
        Args:
            text: Query text
            
        Returns:
            Embedding vector
        """
        return self.ollama.embed_query(text)
    
    def embed_documents(self, texts: List[str], file_types: List[str] = None) -> List[List[float]]:
        """
        Embed documents, using BERT only for natural language text files.
        
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
            if ftype in self.BERT_FILE_TYPES:
                bert_indices.append(i)
                bert_texts.append(text)
            else:
                ollama_indices.append(i)
                ollama_texts.append(text)
        
        results = [None] * len(texts)
        
        # Process BERT texts
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
        
        # Process Ollama texts
        if ollama_texts:
            ollama_embeddings = self.ollama.embed_documents(ollama_texts)
            for idx, emb in zip(ollama_indices, ollama_embeddings):
                results[idx] = emb
        
        return results

