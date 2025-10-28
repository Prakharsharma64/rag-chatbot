
import os 
import fitz 
from typing import List ,Dict 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
import hashlib 


class PDFProcessor :


    def __init__ (self ,chunk_size :int =1000 ,chunk_overlap :int =200 ):
        self .chunk_size =chunk_size 
        self .chunk_overlap =chunk_overlap 
        self .text_splitter =RecursiveCharacterTextSplitter (
        chunk_size =chunk_size ,
        chunk_overlap =chunk_overlap ,
        length_function =len ,
        separators =["\n\n","\n",". "," ",""]
        )

    def extract_text_from_pdf (self ,pdf_path :str )->str :

        try :
            doc =fitz .open (pdf_path )
            text =""
            for page in doc :
                text +=page .get_text ()
            doc .close ()
            return text 
        except Exception as e :
            print (f"Error loading PDF {pdf_path}: {str(e)}")
            return ""

    def chunk_text (self ,text :str ,metadata :Dict =None )->List [Dict ]:

        chunks =self .text_splitter .split_text (text )

        results =[]
        for i ,chunk in enumerate (chunks ):
            chunk_dict ={
            "text":chunk ,
            "chunk_id":i ,
            "metadata":metadata or {}
            }
            results .append (chunk_dict )

        return results 

    def process_pdf (self ,pdf_path :str )->List [Dict ]:

        text =self .extract_text_from_pdf (pdf_path )
        if not text .strip ():
            return []

        filename =os .path .basename (pdf_path )

        file_hash =self ._get_file_hash (pdf_path )

        metadata ={
        "source":filename ,
        "file_path":pdf_path ,
        "file_hash":file_hash 
        }

        chunks =self .chunk_text (text ,metadata )
        return chunks 

    def _get_file_hash (self ,file_path :str )->str :

        hash_md5 =hashlib .md5 ()
        with open (file_path ,"rb")as f :
            for chunk in iter (lambda :f .read (4096 ),b""):
                hash_md5 .update (chunk )
        return hash_md5 .hexdigest ()

    def get_all_pdfs (self ,directory :str )->List [str ]:

        pdf_files =[]
        if os .path .exists (directory ):
            for filename in os .listdir (directory ):
                if filename .lower ().endswith ('.pdf'):
                    pdf_files .append (os .path .join (directory ,filename ))
        return pdf_files 
