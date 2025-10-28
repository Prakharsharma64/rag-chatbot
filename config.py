
import os 


LLM_MODEL_NAME ="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
EMBEDDING_MODEL_NAME ="sentence-transformers/all-MiniLM-L6-v2"


KNOWLEDGE_BASE_DIR ="knowledge_base"
VECTOR_STORE_DIR ="vector_store"
CACHE_DIR =".cache"


CHUNK_SIZE =1500 
CHUNK_OVERLAP =300 
TOP_K =3 


MAX_NEW_TOKENS =256 
TEMPERATURE =0.3 
TOP_P =0.85 


os .makedirs (KNOWLEDGE_BASE_DIR ,exist_ok =True )
os .makedirs (VECTOR_STORE_DIR ,exist_ok =True )
os .makedirs (CACHE_DIR ,exist_ok =True )
