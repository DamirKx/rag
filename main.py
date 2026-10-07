from src.chunking import get_fixed_size_chunks
from src.embeddings import EmbeddingModel
from src.retriever import VectorRetriever
from src.eval import calculate_mrr, calculate_recall_at_k
import os
from pathlib import Path

def load_documents(data_dir: str) -> list:
    """Read all *.txt files from *data_dir* and return a list of their text content.
    Each file is treated as a separate document.
    """
    texts = []
    for txt_path in Path(data_dir).glob("*.txt"):
        with open(txt_path, encoding="utf-8") as f:
            texts.append(f.read())
    return texts

def main():
    print("--- Demonstration of Retrieval Evaluation Pipeline with real data ---")

    # 1. Load documents from the data folder (the user added a law text here)
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    documents = load_documents(data_dir)
    if not documents:
        print("No documents found in the data directory. Exiting.")
        return
    print(f"Loaded {len(documents)} document(s) from {data_dir}")

    # 2. Chunking – we keep the fixed‑size strategy for simplicity.
    print("\n[1/4] Chunking documents...")
    all_chunks = []
    doc_ids = []
    for idx, doc in enumerate(documents, start=1):
        chunks = get_fixed_size_chunks(doc, chunk_size=500, chunk_overlap=50)
        all_chunks.extend(chunks)
        doc_ids.extend([f"doc{idx}_{i}" for i in range(len(chunks))])
    print(f"Created {len(all_chunks)} chunks total.")

    # 3. Embeddings – use a lightweight multilingual model that works on CPU.
    print("\n[2/4] Initializing embedding model (multilingual MiniLM)...")
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedder = EmbeddingModel(model_name=model_name, device="cpu")
    print("Vectorising the knowledge base...")
    doc_embeddings = embedder.encode_documents(all_chunks)

    # 4. Retriever (FAISS)
    print("\n[3/4] Building FAISS index and adding documents...")
    embedding_dim = doc_embeddings.shape[1]
    retriever = VectorRetriever(embedding_dim=embedding_dim)
    retriever.add_documents(doc_embeddings, all_chunks, doc_ids)

    # 5. Evaluation – synthetic queries that should hit the law text.
    print("\n[4/4] Running a quick evaluation...")
    queries = [
        "Какие гарантии предоставляются гражданам после службы?",
        "Какие изменения внесены в закон об образовании?",
        "Что регулирует статья 50‑1 о воинской службе?"
    ]
    # Build a naive ground‑truth: any chunk containing a keyword from the query is considered relevant.
    ground_truth_relevant = []
    for q in queries:
        keywords = q.lower().split()
        relevant = [doc_ids[i] for i, chunk in enumerate(all_chunks) if any(kw in chunk.lower() for kw in keywords)]
        ground_truth_relevant.append(relevant)

    query_embeddings = embedder.encode_queries(queries)
    distances, indices = retriever.search(query_embeddings, k=3)
    predicted_docs = []
    for q_idx in range(len(queries)):
        pred = [retriever.get_doc_id(idx) for idx in indices[q_idx]]
        predicted_docs.append(pred)
        print(f"\nQuery: '{queries[q_idx]}'")
        print(f"Top‑3 chunks: {pred}")

    mrr = calculate_mrr(ground_truth_relevant, predicted_docs)
    recall_at_3 = calculate_recall_at_k(ground_truth_relevant, predicted_docs, k=3)
    print("\n--- Results (k=3) ---")
    print(f"MRR: {mrr:.4f}")
    print(f"Recall@3: {recall_at_3:.4f}")

if __name__ == "__main__":
    main()

from src.embeddings import EmbeddingModel
from src.retriever import VectorRetriever
from src.eval import calculate_mrr, calculate_recall_at_k
import numpy as np

def main():
    print("--- Демонстрация Pipeline'а оценки Retrieval-части RAG ---")
    
    # 1. Загрузка/создание dummy документов (нормативные документы вуза)
    # Представим, что у нас есть русский и английский тексты
    doc1 = "Студенты обязаны посещать не менее 80% занятий. В случае пропусков требуется справка."
    doc2 = "Students must attend at least 80% of classes. A medical certificate is required for absences."
    doc3 = "Курсовая работа сдается до 15 мая. Задержка влечет снижение оценки."
    
    # 2. Чанкинг
    print("\n[1/4] Чанкинг документов...")
    chunks1 = get_fixed_size_chunks(doc1, chunk_size=50, chunk_overlap=10)
    chunks2 = get_fixed_size_chunks(doc2, chunk_size=50, chunk_overlap=10)
    chunks3 = get_fixed_size_chunks(doc3, chunk_size=50, chunk_overlap=10)
    
    all_chunks = chunks1 + chunks2 + chunks3
    # Генерируем ID для чанков (в реальности они должны быть привязаны к исходному документу)
    doc_ids = [f"doc1_{i}" for i in range(len(chunks1))] + \
              [f"doc2_{i}" for i in range(len(chunks2))] + \
              [f"doc3_{i}" for i in range(len(chunks3))]

    # 3. Эмбеддинги
    print("\n[2/4] Инициализация модели эмбеддингов (LaBSE / E5 / MiniLM)...")
    # Для теста берем легкую мультиязычную модель
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedder = EmbeddingModel(model_name=model_name, device="cpu")
    
    print("Векторизация базы знаний...")
    doc_embeddings = embedder.encode_documents(all_chunks)
    
    # 4. Ретривер (FAISS)
    print("\n[3/4] Добавление в векторную БД (FAISS)...")
    embedding_dim = doc_embeddings.shape[1]
    retriever = VectorRetriever(embedding_dim=embedding_dim)
    retriever.add_documents(doc_embeddings, all_chunks, doc_ids)
    
    # 5. Оценка (Evaluation)
    print("\n[4/4] Оценка качества поиска (Retrieval Evaluation)...")
    queries = [
        "What happens if I submit my coursework late?", # Ожидаем релевантность doc3
        "Сколько занятий можно пропустить?" # Ожидаем релевантность doc1 или doc2
    ]
    
    # Ground truth (qrels). Для query 1 -> doc3, для query 2 -> doc1/doc2
    ground_truth_relevant = [
        [id for id in doc_ids if "doc3" in id], 
        [id for id in doc_ids if "doc1" in id or "doc2" in id]
    ]
    
    query_embeddings = embedder.encode_queries(queries)
    distances, indices = retriever.search(query_embeddings, k=2)
    
    # Извлекаем предсказанные ID
    predicted_docs = []
    for q_idx in range(len(queries)):
        pred = [retriever.get_doc_id(idx) for idx in indices[q_idx]]
        predicted_docs.append(pred)
        print(f"Запрос: '{queries[q_idx]}'")
        print(f"Найдены чанки: {pred}")
        
    mrr = calculate_mrr(ground_truth_relevant, predicted_docs)
    recall_at_2 = calculate_recall_at_k(ground_truth_relevant, predicted_docs, k=2)
    
    print(f"\n--- Результаты (k=2) ---")
    print(f"MRR: {mrr:.4f}")
    print(f"Recall@2: {recall_at_2:.4f}")

if __name__ == "__main__":
    main()
