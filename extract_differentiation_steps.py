import sys
import time
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"


PROMPT_TEMPLATE = """
You are a STRICT biomedical protocol extraction system.

Extract ONLY information explicitly present in the text.

DO NOT infer anything.
DO NOT use external knowledge.
If missing → "Not specified in text".

Return EXACT format:

CELL LINE:
- ...

TARGET CELL:
- ...

DIFFERENTIATION STEPS:

Step X:
- Description:
- Duration:
- Cell markers:
- Growth factors:
- Culture media:
- Matrix:

RULES:
- Do NOT merge steps
- Do NOT rewrite biology
- ONLY extract explicit statements

TEXT:
{context}
"""


# ---------------------------------------------------
# Helper: filter only protocol-like chunks
# ---------------------------------------------------
def is_protocol_chunk(text: str) -> bool:
    keywords = [
        "differentiation", "BMP", "VEGF", "WNT",
        "mesoderm", "cardiac", "stem", "EB",
        "marker", "culture", "media", "day", "stage"
    ]
    return sum(k.lower() in text.lower() for k in keywords) >= 2


def main():
    if len(sys.argv) != 3:
        print("Usage: python extract_protocol_v2.py <topic> <pdf_name>")
        sys.exit(1)

    topic = sys.argv[1]
    pdf_name = sys.argv[2]

    run(topic, pdf_name)


def run(topic: str, pdf_name: str):

    start_total = time.time()

    print("\n===== PROTOCOL EXTRACTION v2 =====")
    print(f"Topic: {topic}")
    print(f"PDF: {pdf_name}")

    # ---------------------------------------------------
    # Load embedding
    # ---------------------------------------------------
    print("\n[1] Loading embeddings...")
    embedding_function = get_embedding_function()

    # ---------------------------------------------------
    # Connect DB
    # ---------------------------------------------------
    print("[2] Connecting DB...")
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )

    # ---------------------------------------------------
    # BIG retrieval (important)
    # ---------------------------------------------------
    print("[3] Vector search...")

    results = db.similarity_search_with_score(topic, k=40)

    print(f"[3] Retrieved {len(results)} chunks")

    # ---------------------------------------------------
    # Filter by PDF
    # ---------------------------------------------------
    print("[4] Filtering PDF + protocol content...")

    filtered = []
    for doc, score in results:
        source = doc.metadata.get("source", "")
        text = doc.page_content

        if pdf_name in source and is_protocol_chunk(text):
            filtered.append(text)

    print(f"[4] Protocol chunks: {len(filtered)}")

    if not filtered:
        print("No protocol-related content found.")
        return

    # ---------------------------------------------------
    # LIMIT CONTEXT (VERY IMPORTANT FOR SPEED)
    # ---------------------------------------------------
    filtered = filtered[:8]

    context = "\n\n---\n\n".join(filtered)

    print(f"[5] Context size: {len(context)} chars")

    # ---------------------------------------------------
    # Prompt
    # ---------------------------------------------------
    print("[6] Building prompt...")

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    formatted = prompt.format(context=context)

    # ---------------------------------------------------
    # Model
    # ---------------------------------------------------
    print("[7] Loading model...")

    model = OllamaLLM(
        model="mistral",
        temperature=0
    )

    print("[7] Model ready")

    # ---------------------------------------------------
    # Run extraction
    # ---------------------------------------------------
    print("[8] Extracting structured protocol...")

    start = time.time()
    output = model.invoke(formatted)
    end = time.time()

    print(f"[8] Done in {round(end - start, 2)} sec")

    # ---------------------------------------------------
    # Output
    # ---------------------------------------------------
    print("\n===== FINAL PROTOCOL =====\n")
    print(output)

    print("\n===== DONE =====")
    print(f"Total runtime: {round(time.time() - start_total, 2)} sec")


if __name__ == "__main__":
    main()