import os
import tempfile
from pathlib import Path
# from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import TextLoader
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
load_dotenv()

def load_text_file():
    # Create a temporary text file for demo.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(b"This is a sample text file for testing the TextLoader.")
        temp_file_path = Path(temp_file.name)

    try:
        # Load the text file using TextLoader
        loader = TextLoader(str(temp_file_path))
        documents = loader.load()
        print(f"Loaded {len(documents)} document(s) from the text file.")
        print(f"Loaded document content: {documents[0].page_content[:50]}...")  # Print the first 50 characters of the document content 
        print(f"Loaded document metadata: {documents[0].metadata}")

        # for doc in documents:
        #     print("Document Content")
        #     print(doc)
        #     print(f"Loaded document content: {doc.page_content}")

    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)



if __name__ == "__main__":
    load_text_file()