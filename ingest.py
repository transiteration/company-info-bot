import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Qdrant

load_dotenv()

START_DATETIME_STR = "2024-06-01 10:30:00"
start_datetime = datetime.strptime(START_DATETIME_STR, "%Y-%m-%d %H:%M:%S")

def filter_documents_by_creation_date(all_docs, s_datetime):
    # This function filters documents by their creation date
    print(f"Filtering {len(all_docs)} documents by creation date (after {s_datetime.strftime('%Y-%m-%d %H:%M:%S')})...")
    f_documents = []
    for doc in all_docs:
        file_path = doc.metadata.get("source")
        if not file_path:
            continue
        try:
            file_creation_timestamp = os.path.getctime(file_path)
            file_creation_datetime = datetime.fromtimestamp(file_creation_timestamp)
            if file_creation_datetime >= s_datetime:
                f_documents.append(doc)
                print(f"  [+] Including '{os.path.basename(file_path)}' (Created: {file_creation_datetime.strftime('%Y-%m-%d %H:%M:%S')})")
            else:
                print(f"  [-] Skipping '{os.path.basename(file_path)}' (Created: {file_creation_datetime.strftime('%Y-%m-%d %H:%M:%S')})")
        except FileNotFoundError:
            print(f"Warning: Could not find file {file_path} to check its timestamp.")
            continue
    return f_documents

def get_all_documents():
    # Loads only text documents from the documents folder
    loaders = [
        ("Text", DirectoryLoader(
            "./documents",
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True
        )),
    ]
    all_docs = []
    for name, loader in loaders:
        try:
            print(f"Loading {name} files...")
            docs = loader.load()
            print(f"  Loaded {len(docs)} {name} documents.")
            all_docs.extend(docs)
        except Exception as e:
            print(f"  Warning: Could not load {name} files: {e}")
    return all_docs

print("Loading Documents...")
# Loads all supported documents from the documents folder
all_documents = get_all_documents()

if not all_documents:
    print("No documents found in the 'documents' folder.")
    exit()

print(f"Found {len(all_documents)} total documents.")

filtered_documents = filter_documents_by_creation_date(all_documents, start_datetime)

if not filtered_documents:
    print("\nNo documents found within the specified date range. Nothing to process.")
    exit()

print(f"\nFinished filtering. {len(filtered_documents)} documents will be processed.")

print("Splitting documents...")
# Splits documents into smaller chunks for embedding
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(filtered_documents)
print(f"Split into {len(docs)} chunks.")

print("Creating embeddings and storing in Qdrant...")
# Makes embeddings and saves them to Qdrant vector database
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

collection_name = "company_docs_collection"

qdrant = Qdrant.from_documents(
    docs,
    embeddings,
    url=os.getenv("QDRANT_URL"),
    prefer_grpc=True,
    collection_name=collection_name,
)

print(f"Successfully stored {len(docs)} chunks in Qdrant collection '{collection_name}'.")