from src.chunking import get_fixed_size_chunks
from src.embeddings import EmbeddingModel
from src.retriever import VectorRetriever
from src.eval import calculate_mrr, calculate_recall_at_k
import os
from pathlib import Path

def load_documents(data_dir: str) -> list:
    """Read all *.txt files from *data_dir* and return a list of their text content.
    Each file is treated as a separate document."""
    texts = []
    for txt_path in Path(data_dir).glob("*.txt"):
        with open(txt_path, encoding="utf-8") as f:
            texts.append(f.read())
    return texts

def main():
    print("--- Retrieval Evaluation Pipeline with real law data ---")
    # Data directory is inside the project root (RAG/data)
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    documents = load_documents(data_dir)
    if not documents:
        print("No documents found in the data directory.")
        return
    print(f"Loaded {len(documents)} document(s) from {data_dir}")

    # Chunking
    all_chunks = []
    doc_ids = []
    for idx, doc in enumerate(documents, start=1):
        chunks = get_fixed_size_chunks(doc, chunk_size=500, chunk_overlap=50)
        all_chunks.extend(chunks)
        doc_ids.extend([f"doc{idx}_{i}" for i in range(len(chunks))])
    print(f"Created {len(all_chunks)} chunks.")

    # Embeddings
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedder = EmbeddingModel(model_name=model_name, device="cpu")
    doc_embeddings = embedder.encode_documents(all_chunks)

    # Retriever (FAISS)
    retriever = VectorRetriever(embedding_dim=doc_embeddings.shape[1])
    retriever.add_documents(doc_embeddings, all_chunks, doc_ids)

    # Simple evaluation with synthetic queries
    queries = [
        "Какие гарантии предоставляются гражданам после службы?",
        "Какие изменения внесены в закон об образовании?",
        "Что регулирует статья 50‑1 о воинской службе?"
    ]
    ground_truth = []
    for q in queries:
        kws = q.lower().split()
        relevant = [doc_ids[i] for i, chunk in enumerate(all_chunks) if any(k in chunk.lower() for k in kws)]
        ground_truth.append(relevant)
    query_emb = embedder.encode_queries(queries)
    _, indices = retriever.search(query_emb, k=3)
    predicted = [[retriever.get_doc_id(idx) for idx in row] for row in indices]
    mrr = calculate_mrr(ground_truth, predicted)
    recall = calculate_recall_at_k(ground_truth, predicted, k=3)
    print("Results (k=3):")
    print(f"MRR: {mrr:.4f}, Recall@3: {recall:.4f}")

if __name__ == "__main__":
    main()
