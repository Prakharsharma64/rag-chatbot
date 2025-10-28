"""
Configuration file for the RAG chatbot
"""
import os

# Model configurations - Using CPU for now to avoid memory issues
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Paths
KNOWLEDGE_BASE_DIR = "knowledge_base"
VECTOR_STORE_DIR = "vector_store"
CACHE_DIR = ".cache"

# RAG parameters
CHUNK_SIZE = 1500  # Increased for better context
CHUNK_OVERLAP = 300  # Increased for better continuity
TOP_K = 3  # Increased for better retrieval

# Chat parameters
MAX_NEW_TOKENS = 256  # Reduced for faster responses
TEMPERATURE = 0.3  # Lower temperature for more focused responses
TOP_P = 0.85  # Slightly lower for better quality

# Create necessary directories
os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
