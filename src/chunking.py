def get_fixed_size_chunks(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """Split *text* into overlapping chunks of *chunk_size* characters.
    The next chunk starts ``chunk_size - chunk_overlap`` characters after the
    previous chunk's start, ensuring the specified overlap.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non‑negative and less than chunk_size")
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks


def get_semantic_chunks(text: str, model_name: str = "intfloat/multilingual-e5-base", chunk_size: int = 256) -> list:
    """Placeholder for semantic chunking.
    The original implementation relied on ``langchain_text_splitters`` which is
    not installed in the environment. Implementing a full semantic splitter is
    out of scope for this demo, so we fall back to the fixed‑size chunker.
    """
    # Reuse the simple fixed‑size splitter as a fallback.
    return get_fixed_size_chunks(text, chunk_size=chunk_size, chunk_overlap=30)
    """Split *text* into overlapping chunks of *chunk_size* characters.
    The next chunk starts ``chunk_size - chunk_overlap`` characters after the
    previous chunk's start, ensuring the specified overlap.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non‑negative and less than chunk_size")
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks


def get_semantic_chunks(text: str, model_name: str = "intfloat/multilingual-e5-base", chunk_size: int = 256) -> list:
    """Placeholder for semantic chunking.
    The original implementation relied on ``langchain_text_splitters`` which is
    not installed in the environment. Implementing a full semantic splitter is
    out of scope for this demo, so we fall back to the fixed‑size chunker.
    """
    # Reuse the simple fixed‑size splitter as a fallback.
    return get_fixed_size_chunks(text, chunk_size=chunk_size, chunk_overlap=30)
    """Split *text* into overlapping chunks of *chunk_size* characters.
    The next chunk starts ``chunk_size - chunk_overlap`` characters after the
    previous chunk's start, ensuring the specified overlap.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non‑negative and less than chunk_size")
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks


def get_semantic_chunks(text: str, model_name: str = "intfloat/multilingual-e5-base", chunk_size: int = 256) -> list:
    """Placeholder for semantic chunking.
    The original implementation relied on ``langchain_text_splitters`` which is
    not installed in the environment. Implementing a full semantic splitter is
    out of scope for this demo, so we fall back to the fixed‑size chunker.
    """
    # Reuse the simple fixed‑size splitter as a fallback.
    return get_fixed_size_chunks(text, chunk_size=chunk_size, chunk_overlap=30)

def get_fixed_size_chunks(text: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Разбивает текст на чанки фиксированного размера символов с перекрытием.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non‑negative and less than chunk_size")
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

def get_semantic_chunks(text: str, model_name: str = "intfloat/multilingual-e5-base", chunk_size: int = 256):
    """
    Разбивает текст на основе токенов конкретной модели (с учетом ее лимита токенов).
    Полезно для избежания обрезки текста эмбеддинг-моделью.
    """
    text_splitter = SentenceTransformersTokenTextSplitter(
        model_name=model_name,
        chunk_overlap=30,
        tokens_per_chunk=chunk_size
    )
    return text_splitter.split_text(text)

# Дополнительно можно добавить SemanticChunker (на основе косинусного сходства предложений)
