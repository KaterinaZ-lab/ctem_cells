import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma


CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():

    # ---------------------------------------------------
    # Parse CLI arguments
    # ---------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    # ---------------------------------------------------
    # Reset database if requested
    # ---------------------------------------------------

    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # ---------------------------------------------------
    # Load PDF documents
    # ---------------------------------------------------

    documents = load_documents()

    # ---------------------------------------------------
    # Split documents into chunks
    # ---------------------------------------------------

    chunks = split_documents(documents)

    # ---------------------------------------------------
    # Store chunks in Chroma
    # ---------------------------------------------------

    add_to_chroma(chunks)


def load_documents():

    # ---------------------------------------------------
    # Load all PDFs from data directory
    # ---------------------------------------------------

    document_loader = PyPDFDirectoryLoader(DATA_PATH)

    return document_loader.load()


def split_documents(documents: list[Document]):

    # ---------------------------------------------------
    # Create text splitter
    # ---------------------------------------------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )

    # ---------------------------------------------------
    # Split documents into chunks
    # ---------------------------------------------------

    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):

    # ---------------------------------------------------
    # Connect to Chroma database
    # ---------------------------------------------------

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

    # ---------------------------------------------------
    # Generate chunk IDs
    # ---------------------------------------------------

    chunks_with_ids = calculate_chunk_ids(chunks)

    # ---------------------------------------------------
    # Get existing documents in database
    # ---------------------------------------------------

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])

    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # ---------------------------------------------------
    # Identify new chunks
    # ---------------------------------------------------

    new_chunks = []

    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    # ---------------------------------------------------
    # Insert new chunks
    # ---------------------------------------------------

    if len(new_chunks):

        print(f"👉 Adding new documents: {len(new_chunks)}")

        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]

        db.add_documents(new_chunks, ids=new_chunk_ids)

        # db.persist()

    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # ---------------------------------------------------
    # Create unique IDs for each chunk
    # ---------------------------------------------------
    # Format:
    # data/file.pdf:page:chunk
    # Example:
    # data/book.pdf:6:2
    # ---------------------------------------------------

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:

        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")

        current_page_id = f"{source}:{page}"

        # ---------------------------------------------------
        # Check if chunk belongs to same page
        # ---------------------------------------------------

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # ---------------------------------------------------
        # Assign chunk ID
        # ---------------------------------------------------

        chunk_id = f"{current_page_id}:{current_chunk_index}"

        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():

    # ---------------------------------------------------
    # Delete Chroma database folder
    # ---------------------------------------------------

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
