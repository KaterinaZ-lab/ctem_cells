import argparse
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function


CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():

    # ---------------------------------------------------
    # Parse CLI arguments
    # ---------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--pdf", type=str, default=None, help="Restrict results to a specific PDF.")
    args = parser.parse_args()

    # ---------------------------------------------------
    # Run RAG query
    # ---------------------------------------------------

    query_rag(args.query_text, args.pdf)


def query_rag(query_text: str, pdf_name: str = None):

    # ---------------------------------------------------
    # Load embedding function
    # ---------------------------------------------------

    embedding_function = get_embedding_function()

    # ---------------------------------------------------
    # Connect to Chroma database
    # ---------------------------------------------------

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )

    # ---------------------------------------------------
    # Search for relevant chunks
    # ---------------------------------------------------

    results = db.similarity_search_with_score(query_text, k=5)

    if pdf_name:
        results = [(doc, score) for doc, score in results if pdf_name in doc.metadata.get("source", "")]

    # ---------------------------------------------------
    # Build context from retrieved chunks
    # ---------------------------------------------------

    context_text = "\n\n---\n\n".join(
        [doc.page_content for doc, _score in results]
    )

    # ---------------------------------------------------
    # Build prompt
    # ---------------------------------------------------

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    prompt = prompt_template.format(
        context=context_text,
        question=query_text
    )

    # ---------------------------------------------------
    # Load LLM
    # ---------------------------------------------------

    model = OllamaLLM(model="mistral")

    # ---------------------------------------------------
    # Generate response
    # ---------------------------------------------------

    response_text = model.invoke(prompt)

    # ---------------------------------------------------
    # Collect sources with scores
    # ---------------------------------------------------

    sources = [
        f"{doc.metadata.get('id', None)} (score: {round(score, 3)})"
        for doc, score in results
    ]

    # ---------------------------------------------------
    # Output response
    # ---------------------------------------------------

    formatted_response = f"Response: {response_text}\nSources: {sources}"

    print(formatted_response)

    return response_text


if __name__ == "__main__":
    main()
