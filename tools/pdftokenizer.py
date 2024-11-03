# %%
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# %%
# Directory containing PDF files
pdf_directory = "./documents"

# %%
# Initialize an empty list to hold all documents
documents = []

# %%
# Load each PDF file in the directory
for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):
        pdf_loader = PyPDFLoader(file_path=os.path.join(pdf_directory, filename))
        documents.extend(pdf_loader.load())

# %%

# Split the loaded documents into smaller chunks for better embedding
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# %%
# Create embeddings for the document chunks
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')


# %%
# Store the document embeddings in Chroma vectorstore
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_vectorstore")

# Persist the vectorstore for future retrieval
vectorstore.persist()



