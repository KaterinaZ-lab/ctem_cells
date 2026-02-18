import chromadb
from pypdf import PdfReader

# Chroma client open comunication
client = chromadb.Client()

# Collection
# collection_history = client.get_or_create_collection("history")
# collection_biology = client.get_or_create_collection('biology')


collection_pdfs = client.get_or_create_collection('pdfs')

# read PDF
reader = PdfReader("pdf_Test.pdf")

texts = []
for page in reader.pages:
    txt = page.extract_text()
    if txt:
        texts.append(txt)

# save
collection_pdfs.add(
    documents=texts,
    ids=[str(i) for i in range(len(texts))]
)

print("✅ Αποθηκεύτηκε στη Chroma Κατερίνα!")


# view saved
data = collection_pdfs.get()

print("\n📄 Περιεχόμενα του pdf:")
for doc in data["documents"]:
    print("-", doc)  