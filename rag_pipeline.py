"""
RAG pipeline combining retrieval and generation
"""
from typing import List, Dict
import config
from vector_store import VectorStoreManager
from llm_handler import LLMHandler
import os
import shutil


class RAGPipeline:
    """Combines retrieval and generation for RAG"""
    
    def __init__(self, llm_handler=None):
        self.vector_store = VectorStoreManager()
        self.llm_handler = llm_handler or LLMHandler()
        self.conversation_history = []
    
    def initialize(self):
        """Initialize the pipeline by loading vector store and model"""
        print("Initializing RAG pipeline...")
        self.vector_store.load_or_create_store()
        # Load model (will be cached by Streamlit)
        self.llm_handler.load_model()
        print("RAG pipeline initialized")
    
    def build_context(self, query: str) -> str:
        """
        Build context from relevant documents for the query
        """
        if self.vector_store.is_empty():
            return ""
        
        results = self.vector_store.search(query, top_k=config.TOP_K)
        
        if not results:
            return ""
        
        context_parts = [result['text'] for result in results]
        return "\n\n---\n\n".join(context_parts)
    
    def generate_response(
        self,
        query: str,
        use_history: bool = True,
        show_context: bool = False
    ) -> Dict:
        """
        Generate a response using RAG
        
        Returns a dictionary with 'response' and optionally 'context'
        """
        # Build context
        context = self.build_context(query)
        
        # Build the prompt
        if use_history and self.conversation_history:
            history_text = "\n".join([
                f"User: {msg['query']}\nAssistant: {msg['response']}"
                for msg in self.conversation_history[-2:]
            ])
            prompt = f"""Context: {context}

Previous: {history_text}

Question: {query}

Answer:"""
        else:
            if not context:
                prompt = f"""Answer this question briefly:

Question: {query}

Answer:"""
            else:
                prompt = f"""Answer the question using ONLY the context provided.

Context:
{context}

Question: {query}

Answer (based on context only):"""
        
        # Generate response
        response = self.llm_handler.generate_response(
            prompt,
            max_new_tokens=config.MAX_NEW_TOKENS,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P
        )
        
        # Add to history
        self.conversation_history.append({
            'query': query,
            'response': response
        })
        
        result = {'response': response}
        if show_context:
            result['context'] = context
        
        return result
    
    def add_documents(self, chunks: List[Dict]):
        """Add documents to the vector store"""
        self.vector_store.add_documents(chunks)
    
    def save_vector_store(self):
        """Save the vector store"""
        self.vector_store.save()
    
    def get_total_documents(self) -> int:
        """Get total number of documents"""
        return self.vector_store.get_total_documents()

    def get_loaded_pdfs(self) -> List[str]:
        """Return unique PDF filenames currently stored in the vector store"""
        return list({meta.get('source') for meta in self.vector_store.metadata if meta.get('source')})
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def clear_knowledge_base(self):
        """Clear the knowledge base on disk and in memory"""
        # Remove stored FAISS index and metadata files if they exist
        if os.path.exists(self.vector_store.store_path):
            shutil.rmtree(self.vector_store.store_path, ignore_errors=True)
        if os.path.exists(self.vector_store.metadata_path):
            try:
                os.remove(self.vector_store.metadata_path)
            except OSError:
                pass
        # Reset in-memory structures
        self.vector_store.clear_store()
    
    def rebuild_knowledge_base(self, pdf_files: List[str]):
        """Rebuild the knowledge base from PDF files"""
        from pdf_loader import PDFProcessor
        
        self.clear_knowledge_base()
        processor = PDFProcessor(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        
        for pdf_file in pdf_files:
            print(f"Processing {pdf_file}...")
            chunks = processor.process_pdf(pdf_file)
            if chunks:
                self.add_documents(chunks)
        
        self.save_vector_store()
