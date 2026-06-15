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
    # Retrieve relevant chunks using multiple queries
    # for broad coverage across the paper
    # ---------------------------------------------------

    start = time.time()

    queries = [
        "background introduction motivation research question",
        "methods experimental protocol materials procedures",
        "results findings data observations",
        "discussion conclusions implications future work",
    ]

    seen_ids = set()
    filtered_docs = []
    scores = []

    for query in queries:
        results = db.similarity_search_with_score(query, k=10)
        for doc, score in results:
            source = doc.metadata.get("source", "")
            chunk_id = doc.metadata.get("id", "")
            if pdf_name in source and chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                filtered_docs.append(doc)
                scores.append(score)

    if not filtered_docs:
        print("No chunks found for this PDF.")
        return

    avg_score = round(sum(scores) / len(scores), 3)
    print(f"Retrieved {len(filtered_docs)} unique chunks (avg retrieval score: {avg_score})")

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
