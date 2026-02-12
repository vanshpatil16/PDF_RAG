from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

def chunk_sections(sections):
    documents = []
    for sec in sections:
        chunks = splitter.split_text(sec["content"])
        for i, chunk in enumerate(chunks):
            documents.append({
                "text": chunk,
                "metadata": {
                    "section": sec["title"],
                    "page": sec.get("page"),
                    "chunk_id": i
                }
            })
    return documents
