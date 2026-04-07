#!/usr/bin/env python3
"""
Test script for Ollama embedding models with cosine similarity
"""

import ollama
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time

# List of embedding models you have available (from your Ollama installation)
embedding_models = [
    "all-minilm:latest",           # 45 MB - fastest, lightweight
    "nomic-embed-text:latest",     # 274 MB - good balance
    "mxbai-embed-large:latest",    # 669 MB - best quality
    "embeddinggemma:latest",       # 621 MB - Google's model
    "Qwen3-Embedding:8B",          # 4.7 GB - highest quality, slower
    "bge-m3"                       # 1.2 GB
]

# Test sentences with different semantic meanings
test_sentences = [
    "The ocean is vast and full of mysterious creatures",
    "Bart works for ENTER, a company in Eindhoven",
    "The sea contains many undiscovered species",
    "Machine learning models can generate text embeddings",
    "Vector similarity measures how close two sentences are in meaning",
    "Python is great for data science, Bart and machine learning",
    "The deep sea holds secrets we haven't explored yet"
]

def get_embeddings(model, texts) -> np.ndarray:
    """Get embeddings for a list of texts using specified model
       Convert the test_sentences to an array of many magical numbers that represent the string and
       can be used to compare with other embeddings. """
    embeddings = []
    for text in texts:
        try:
            response = ollama.embeddings(model=model, prompt=text)
            embeddings.append(response['embedding'])
            print(f"Text: '{text[:50]}...' - Embedding: {response['embedding']}")
        except Exception as e:
            print(f"  Error with text '{text[:50]}...': {e}")
            # Return None for failed embeddings
            return None
    return np.array(embeddings)

def print_similarity_analysis(similarity_matrix, sentences, model_name, elapsed_time):
    """Print detailed similarity analysis"""
    print(f"\n{'='*70}")
    print(f"Model: {model_name}")
    print(f"Time taken: {elapsed_time:.2f} seconds")
    print(f"{'='*70}")
    
    # Print matrix
    print("\nCosine Similarity Matrix:")
    print("    " + "".join([f"{i:6d}" for i in range(len(sentences))]))
    for i in range(len(sentences)):
        print(f"{i:2d}: ", end="")
        for j in range(len(sentences)):
            print(f"{similarity_matrix[i][j]:6.3f}", end="")
        print()
    
    # Find most similar pairs (excluding self-similarity)
    print("\nMost Similar Sentence Pairs (excluding self):")
    similarities = []
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            similarities.append((i, j, similarity_matrix[i][j]))
    
    similarities.sort(key=lambda x: x[2], reverse=True)
    for i, j, sim in similarities[:3]:
        print(f"  '{sentences[i][:40]}...'")
        print(f"  ↔ '{sentences[j][:40]}...'")
        print(f"  Similarity: {sim:.4f}\n")
    
    # Find least similar pairs
    print("Least Similar Sentence Pairs:")
    for i, j, sim in similarities[-3:]:
        print(f"  '{sentences[i][:40]}...'")
        print(f"  ↔ '{sentences[j][:40]}...'")
        print(f"  Similarity: {sim:.4f}\n")
    
    # Statistics
    all_sims = [sim for _, _, sim in similarities]
    print(f"Statistics:")
    print(f"  Mean similarity: {np.mean(all_sims):.4f}")
    print(f"  Std deviation: {np.std(all_sims):.4f}")
    print(f"  Min similarity: {np.min(all_sims):.4f}")
    print(f"  Max similarity: {np.max(all_sims):.4f}")

def test_single_model():
    """Quick test with just one model"""
    print("Quick test with a single model...")
    model = "all-minilm:latest"  # Start with the fastest one
    
    texts = [
        "The ocean is blue",
        "The sea is deep",
        "Cats are fluffy"
    ]
    
    print(f"\nGetting embeddings from {model}...")
    start_time = time.time()
    embeddings = get_embeddings(model, texts)
    elapsed = time.time() - start_time
    
    if embeddings is not None:
        print(f"Embedding dimension: {len(embeddings[0])}")
        similarities = cosine_similarity(embeddings)
        print("\nSimilarity matrix:")
        print(np.round(similarities, 3))
        print(f"\nTime taken: {elapsed:.2f} seconds")
    else:
        print("Failed to get embeddings")

def test_all_models():
    """Test all available embedding models"""
    print("Testing all embedding models...")
    print(f"Number of test sentences: {len(test_sentences)}")
    print(f"Test sentences:")
    for i, sent in enumerate(test_sentences):
        print(f"  {i}: {sent}")
    
    results = {}
    
    for model in embedding_models:
        print(f"\n{'='*70}")
        print(f"Processing model: {model}")
        
        start_time = time.time()
        embeddings = get_embeddings(model, test_sentences)
        elapsed_time = time.time() - start_time
        
        if embeddings is not None:
            # Calculate cosine similarity matrix
            similarity_matrix = cosine_similarity(embeddings)
            
            # Store results
            results[model] = {
                'dimension': len(embeddings[0]),
                'time': elapsed_time,
                'similarities': similarity_matrix
            }
            
            # Print analysis
            print_similarity_analysis(similarity_matrix, test_sentences, model, elapsed_time)
            print(f"Embedding dimension: {len(embeddings[0])}")
        else:
            print(f"  Failed to get embeddings for {model}")
    
    # Summary comparison
    if results:
        print(f"\n{'='*70}")
        print("SUMMARY COMPARISON")
        print(f"{'='*70}")
        print(f"{'Model':<25} {'Dimension':<12} {'Time (s)':<10} {'Avg Similarity':<15}")
        print("-" * 70)
        for model, data in results.items():
            avg_sim = np.mean(data['similarities'][np.triu_indices_from(data['similarities'], k=1)])
            print(f"{model:<25} {data['dimension']:<12} {data['time']:<10.2f} {avg_sim:<15.4f}")

def main():
    """Main function"""
    print("Ollama Embedding Model Test Suite")
    print("=" * 70)
    print("This script tests various embedding models from your Ollama installation")
    print("and compares their performance using cosine similarity.\n")
    
    # Check if ollama is available
    try:
        ollama.list()
    except Exception as e:
        print(f"Error: Cannot connect to Ollama. Is it running?")
        print(f"Details: {e}")
        return
    
    # Choose test mode
    print("Select test mode:")
    print("1. Quick test (single model: all-minilm)")
    print("2. Test all embedding models")
    print("3. Test specific models (custom)")
    
    choice = input("\nEnter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        test_single_model()
    elif choice == "2":
        test_all_models()
    elif choice == "3":
        print("\nAvailable models:")
        for i, model in enumerate(embedding_models):
            print(f"  {i}: {model}")
        
        selection = input("Enter model numbers to test (comma-separated, e.g., 0,1,2): ")
        indices = [int(x.strip()) for x in selection.split(",")]
        selected_models = [embedding_models[i] for i in indices if 0 <= i < len(embedding_models)]
        
        if selected_models:
            for model in selected_models:
                print(f"\n{'='*70}")
                print(f"Testing {model}...")
                start_time = time.time()
                embeddings = get_embeddings(model, test_sentences[:3])  # Use first 3 sentences for speed
                elapsed = time.time() - start_time
                
                if embeddings is not None:
                    similarities = cosine_similarity(embeddings)
                    print(f"Embedding dimension: {len(embeddings[0])}")
                    print("Similarity matrix:")
                    print(np.round(similarities, 3))
                    print(f"Time: {elapsed:.2f} seconds")
                else:
                    print(f"Failed to get embeddings for {model}")
        else:
            print("No valid models selected")
    else:
        print("Invalid choice. Running quick test by default.")
        test_single_model()

if __name__ == "__main__":
    main()

