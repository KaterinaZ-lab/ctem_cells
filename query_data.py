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
    args = parser.parse_args()

    query_text = args.query_text

    # ---------------------------------------------------
    # Run RAG query
    # ---------------------------------------------------

    query_rag(query_text)


def query_rag(query_text: str):

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
    # Collect sources
    # ---------------------------------------------------

    sources = [doc.metadata.get("id", None) for doc, _score in results]

    # ---------------------------------------------------
    # Output response
    # ---------------------------------------------------

    formatted_response = f"Response: {response_text}\nSources: {sources}"

    print(formatted_response)

    return response_text


if __name__ == "__main__":
    main()
