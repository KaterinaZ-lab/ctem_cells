import sys
import time
import numpy as np
from sklearn.cluster import KMeans
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"
N_CLUSTERS = 3

STEP_PROMPT = """
You are a STRICT biomedical protocol extraction system.

Extract ONLY differentiation steps explicitly present in the text below.

DO NOT infer anything.
DO NOT use external knowledge.
If a field is missing → "Not specified in text".

Return EXACT format:

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
- If no steps are present in this text, return "No steps found"

TEXT:
{context}
"""

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
        print("Usage: python extract_steps_clustered.py <pdf_name>")
        sys.exit(1)

    pdf_name = sys.argv[1]
    run(pdf_name)


def run(pdf_name: str):

    start_total = time.time()

    print(f"\n===== CLUSTERED PROTOCOL EXTRACTION | {pdf_name} =====")

    embedding_function = get_embedding_function()

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )

    results = db.similarity_search_with_score("differentiation steps protocol", k=40)

    chunks = []
    for doc, _ in results:
        source = doc.metadata.get("source", "")
        if pdf_name in source and is_protocol_chunk(doc.page_content, pdf_name):
            chunks.append(doc.page_content)

    if not chunks:
        print("No protocol-related content found.")
        return

    vectors = np.array(embedding_function.embed_documents(chunks))

    n_clusters = min(N_CLUSTERS, len(chunks))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(vectors)

    clusters = {i: [] for i in range(n_clusters)}
    for text, label in zip(chunks, labels):
        clusters[label].append(text)

    model = OllamaLLM(model="mistral", temperature=0)
    prompt_template = ChatPromptTemplate.from_template(STEP_PROMPT)

    all_outputs = []

    for cluster_id, cluster_texts in clusters.items():
        context = "\n\n---\n\n".join(cluster_texts)
        formatted = prompt_template.format(context=context)
        output = model.invoke(formatted)

        if "No steps found" not in output:
            all_outputs.append(output)

    # ---------------------------------------------------
    # Merge + renumber steps
    # ---------------------------------------------------
    print("\n===== FINAL PROTOCOL =====\n")
    merged = "\n\n".join(all_outputs)

    step_counter = 0
    renumbered_lines = []
    for line in merged.splitlines():
        if line.strip().startswith("Step "):
            step_counter += 1
            renumbered_lines.append(f"Step {step_counter}:")
        else:
            renumbered_lines.append(line)

    merged = "\n".join(renumbered_lines)
    print(merged)

    # ---------------------------------------------------
    # Completeness scoring
    # ---------------------------------------------------
    step_fields = ["Description", "Duration", "Cell markers", "Growth factors", "Culture media", "Matrix"]
    steps = [line for line in merged.splitlines() if line.strip().startswith("Step ")]
    n_steps = len(steps)

    total_fields = n_steps * len(step_fields)
    missing = merged.count("Not specified in text")
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
