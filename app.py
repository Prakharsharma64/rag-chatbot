import streamlit as st 
import os 
from pathlib import Path 
from rag_pipeline import RAGPipeline 
from pdf_loader import PDFProcessor 
import config 



st .set_page_config (
page_title ="Local RAG Chatbot",
page_icon ="🤖",
layout ="wide"
)


if 'rag_pipeline'not in st .session_state :
    st .session_state .rag_pipeline =None 
if 'initialized'not in st .session_state :
    st .session_state .initialized =False 
if 'messages'not in st .session_state :
    st .session_state .messages =[]


@st .cache_resource 
def load_llm_handler ():

    from llm_handler import LLMHandler 
    handler =LLMHandler ()
    handler .load_model ()
    return handler 

def initialize_app ():

    if not st .session_state .initialized :
        with st .spinner ("Initializing RAG system..."):
            if st .session_state .rag_pipeline is None :

                llm_handler =load_llm_handler ()
                st .session_state .rag_pipeline =RAGPipeline (llm_handler =llm_handler )
            st .session_state .rag_pipeline .initialize ()
            st .session_state .initialized =True 


def main ():
    st .title ("🤖 RAG Chatbot")
    st .markdown ("*AI-powered document assistant*")


    initialize_app ()


    with st .sidebar :

        if st .session_state .rag_pipeline :
            total_docs =st .session_state .rag_pipeline .get_total_documents ()
            st .info (f"📚 {total_docs} chunks loaded")

            pdf_files_loaded =st .session_state .rag_pipeline .get_loaded_pdfs ()
            if pdf_files_loaded :
                with st .expander ("📂 Knowledge Base PDFs"):
                    for pdf_name in pdf_files_loaded :
                        st .text (pdf_name )
            else :
                st .caption ("No PDFs in knowledge base.")

        st .markdown ("---")


        uploaded_file =st .file_uploader (
        "Upload PDF",
        type ="pdf"
        )

        if uploaded_file is not None :
            if st .button ("Add PDF"):
                with st .spinner ("Processing..."):
                    upload_dir =Path (config .KNOWLEDGE_BASE_DIR )
                    upload_dir .mkdir (exist_ok =True )
                    file_path =upload_dir /uploaded_file .name 

                    with open (file_path ,"wb")as f :
                        f .write (uploaded_file .getbuffer ())

                    processor =PDFProcessor ()
                    chunks =processor .process_pdf (str (file_path ))

                    if chunks :
                        st .session_state .rag_pipeline .add_documents (chunks )
                        st .session_state .rag_pipeline .save_vector_store ()
                        st .success (f"Added {uploaded_file.name}")
                        st .rerun ()

        st .markdown ("---")

        if st .button ("Rebuild Knowledge Base"):
            pdf_processor =PDFProcessor ()
            pdf_files =pdf_processor .get_all_pdfs (config .KNOWLEDGE_BASE_DIR )

            if pdf_files :
                with st .spinner ("Rebuilding..."):
                    st .session_state .rag_pipeline .rebuild_knowledge_base (pdf_files )
                    st .success (f"Rebuilt from {len(pdf_files)} file(s)")
                    st .rerun ()
            else :
                st .warning ("No PDFs found")

        if st .button ("Clear Knowledge Base"):
            with st .spinner ("Clearing knowledge base..."):
                st .session_state .rag_pipeline .clear_knowledge_base ()
                st .session_state .rag_pipeline .save_vector_store ()
                st .success ("Knowledge base cleared")
                st .rerun ()

        if st .button ("Clear History"):
            st .session_state .messages =[]
            st .session_state .rag_pipeline .clear_history ()
            st .rerun ()

        show_sources =st .checkbox ("Show sources",value =False )


    for message in st .session_state .messages :
        with st .chat_message (message ["role"]):
            st .markdown (message ["content"])


    if prompt :=st .chat_input ("Ask a question..."):
        st .session_state .messages .append ({"role":"user","content":prompt })
        with st .chat_message ("user"):
            st .markdown (prompt )

        with st .chat_message ("assistant"):
            with st .spinner ("Thinking..."):
                result =st .session_state .rag_pipeline .generate_response (prompt )
                response =result ['response']
                st .markdown (response )

                if show_sources :
                    results =st .session_state .rag_pipeline .vector_store .search (
                    prompt ,
                    top_k =config .TOP_K 
                    )
                    if results :
                        sources =set ([r ['metadata'].get ('source','Unknown')for r in results ])
                        if sources :
                            with st .expander ("Sources"):
                                for source in sources :
                                    st .text (source )

                st .session_state .messages .append ({
                "role":"assistant",
                "content":response 
                })


if __name__ =="__main__":
    main ()
