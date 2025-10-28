
import os 
import pickle 
from typing import List ,Dict ,Optional 
import faiss 
import numpy as np 
from langchain_community .vectorstores import FAISS 
from langchain_huggingface import HuggingFaceEmbeddings 
from sentence_transformers import SentenceTransformer 
import config 


class VectorStoreManager :


    def __init__ (self ):
        self .embedding_model =SentenceTransformer (config .EMBEDDING_MODEL_NAME )
        self .embeddings =HuggingFaceEmbeddings (model_name =config .EMBEDDING_MODEL_NAME )
        self .vector_store =None 
        self .documents =[]
        self .metadata =[]
        self .store_path =os .path .join (config .VECTOR_STORE_DIR ,"faiss_index")
        self .metadata_path =os .path .join (config .VECTOR_STORE_DIR ,"metadata.pkl")

    def load_or_create_store (self ):

        if os .path .exists (self .store_path )and os .path .exists (self .metadata_path ):
            try :

                with open (self .metadata_path ,'rb')as f :
                    self .metadata =pickle .load (f )


                if os .path .exists (os .path .join (self .store_path ,"index.faiss")):
                    self .vector_store =FAISS .load_local (
                    self .store_path ,
                    self .embeddings ,
                    allow_dangerous_deserialization =True 
                    )
                    self .documents =[meta .get ('text','')for meta in self .metadata ]
                    print (f"Loaded {len(self.documents)} documents from vector store")
                else :
                    print ("FAISS index file not found, creating new store")
                    self ._create_new_store ()
            except Exception as e :
                print (f"Error loading vector store: {e}")
                self ._create_new_store ()
        else :
            self ._create_new_store ()

    def _create_new_store (self ):


        self .documents =[]
        self .metadata =[]
        self .vector_store =None 

    def add_documents (self ,chunks :List [Dict ]):

        if not chunks :
            return 

        texts =[chunk ['text']for chunk in chunks ]
        metadatas =[chunk ['metadata']for chunk in chunks ]


        if self .vector_store is None :
            self .vector_store =FAISS .from_texts (
            texts ,
            self .embeddings ,
            metadatas =metadatas 
            )
        else :
            self .vector_store .add_texts (texts ,metadatas =metadatas )


        self .documents .extend (texts )
        self .metadata .extend (metadatas )

    def search (self ,query :str ,top_k :int =5 )->List [Dict ]:

        if self .vector_store is None or len (self .documents )==0 :
            return []

        try :

            results =self .vector_store .similarity_search_with_score (query ,k =top_k )

            formatted_results =[]
            for doc ,score in results :
                formatted_results .append ({
                'text':doc .page_content ,
                'metadata':doc .metadata ,
                'score':float (score )
                })

            return formatted_results 
        except Exception as e :
            print (f"Error during search: {e}")
            return []

    def save (self ):

        if self .vector_store is not None :
            try :

                self .vector_store .save_local (self .store_path )


                with open (self .metadata_path ,'wb')as f :
                    pickle .dump (self .metadata ,f )

                print (f"Saved {len(self.documents)} documents to vector store")
            except Exception as e :
                print (f"Error saving vector store: {e}")

    def get_total_documents (self )->int :

        return len (self .documents )

    def get_file_hash_set (self )->set :

        return set ([meta .get ('file_hash','')for meta in self .metadata ])

    def clear_store (self ):

        self ._create_new_store ()

    def is_empty (self )->bool :

        return len (self .documents )==0 
