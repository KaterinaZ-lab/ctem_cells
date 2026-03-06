import sys
import time
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are an expert summarizer.

Read the following text extracted from a PDF and provide a clear and concise summary of the document.

{text}

SUMMARY:
"""


def main():
    if len(sys.argv) != 2:
        print("Usage: python summarize_pdf.py <pdf_name>")
        sys.exit(1)

    pdf_name = sys.argv[1]
    summarize_pdf(pdf_name)


def summarize_pdf(pdf_name: str):

    start_total = time.time()

    # ---------------------------------------------------
    # Load embeddings
    # ---------------------------------------------------
    embedding_function = get_embedding_function()

    # ---------------------------------------------------
    # Connect to Chroma
    # ---------------------------------------------------
    start = time.time()

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )

    # ---------------------------------------------------
    # Retrieve relevant chunks
    # ---------------------------------------------------

    start = time.time()

    results = db.similarity_search_with_score(pdf_name, k=8)

    if not results:
        print("No results found.")
        return

    # ---------------------------------------------------
    # Filter chunks from the specific PDF
    # ---------------------------------------------------

    filtered_docs = []

    for doc, score in results:
        source = doc.metadata.get("source", "")
        if pdf_name in source:
            filtered_docs.append(doc)

    if not filtered_docs:
        print("No chunks found for this PDF.")
        return

    # ---------------------------------------------------
    # Combine context
    # ---------------------------------------------------

    context_text = "\n\n---\n\n".join([doc.page_content for doc in filtered_docs])


    # ---------------------------------------------------
    # Build prompt
    # ---------------------------------------------------

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(text=context_text)


    # ---------------------------------------------------
    # Load model
    # ---------------------------------------------------

    model = OllamaLLM(model="mistral")


    # ---------------------------------------------------
    # Generate summary
    # ---------------------------------------------------

    start = time.time()

    summary = model.invoke(prompt)

    # ---------------------------------------------------
    # Output
    # ---------------------------------------------------
    print("\n===== SUMMARY =====\n")
    print(summary)

    print("\n===== DONE =====")
    print(f"Total runtime: {round(time.time()-start_total,2)} seconds")


if __name__ == "__main__":
    main()
