import os
import tempfile
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import (TextLoader, WebBaseLoader, DirectoryLoader, PyPDFLoader)

from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv()

def pdf_loader(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    print(f"Loaded {len(documents)} document(s) from the PDF file.")
    for i, doc in enumerate(documents):
        print(f"Document {i+1} Content:")
        print(doc.page_content)  # Print the first 100 characters of the document content
        print(f"Loaded document metadata: {doc.metadata}")


if __name__ == "__main__":
    # Example usage of the pdf_loader function
    pdf_file_path = "./docs/sample.pdf"  # Replace with the actual path to your PDF file
    pdf_loader(pdf_file_path)