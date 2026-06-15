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

GENERIC_KEYWORDS = [
    "differentiation", "stem", "marker", "culture",
    "media", "day", "stage", "protocol", "factor",
    "cell line", "progenitor", "pluripotent"
]

PAPER_KEYWORDS = {
    "yang2008": [
        "BMP", "VEGF", "WNT", "DKK", "mesoderm",
        "cardiac", "EB", "embryoid", "KDR", "cardiomyocyte",
        "activin", "bFGF", "TBX", "MLC2", "CTNT"
    ],
    "kikuchi2017": [
        "dopaminergic", "midbrain", "Dlk1", "neuron",
        "FGF8", "SHH", "floor plate", "substantia nigra",
        "ventral", "forebrain", "neural", "FOXA2", "LMX1"
    ],
}


def is_protocol_chunk(text: str, pdf_name: str) -> bool:
    generic_hits = sum(k.lower() in text.lower() for k in GENERIC_KEYWORDS)

    specific_keywords = next(
        (v for k, v in PAPER_KEYWORDS.items() if k in pdf_name), []
    )
    specific_hits = sum(k.lower() in text.lower() for k in specific_keywords)

    return generic_hits >= 1 or specific_hits >= 1


def main():
    if len(sys.argv) != 2:
        print("Usage: python extract_differentiation_steps.py <pdf_name>")
        sys.exit(1)

    pdf_name = sys.argv[1]
    run("differentiation steps protocol", pdf_name)


def run(topic: str, pdf_name: str):

    start_total = time.time()

    print(f"\n===== PROTOCOL EXTRACTION | {pdf_name} =====")

    embedding_function = get_embedding_function()

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )

    results = db.similarity_search_with_score(topic, k=40)

    filtered = []
    for doc, score in results:
        source = doc.metadata.get("source", "")
        text = doc.page_content
        if pdf_name in source and is_protocol_chunk(text, pdf_name):
            filtered.append(text)

    if not filtered:
        print("No protocol-related content found.")
        return

    filtered = filtered[:8]
    context = "\n\n---\n\n".join(filtered)

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    formatted = prompt.format(context=context)

    model = OllamaLLM(model="mistral", temperature=0)
    output = model.invoke(formatted)

    # ---------------------------------------------------
    # Output
    # ---------------------------------------------------
    print("\n===== FINAL PROTOCOL =====\n")
    print(output)

    # ---------------------------------------------------
    # Completeness scoring
    # ---------------------------------------------------
    step_fields = ["Description", "Duration", "Cell markers", "Growth factors", "Culture media", "Matrix"]
    steps = [line for line in output.splitlines() if line.strip().startswith("Step ")]
    n_steps = len(steps)

    total_fields = n_steps * len(step_fields)
    missing = output.count("Not specified in text")
    filled = total_fields - missing
    pct = round((filled / total_fields) * 100) if total_fields > 0 else 0

    print("\n===== COMPLETENESS SCORE =====")
    print(f"Steps found: {n_steps}")
    print(f"Fields filled: {filled}/{total_fields} ({pct}%)")
    print(f"Missing fields: {missing}")

    print("\n===== DONE =====")
    print(f"Total runtime: {round(time.time() - start_total, 2)} sec")


if __name__ == "__main__":
    main()