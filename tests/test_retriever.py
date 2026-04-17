from app.ingestion.chunking import ChunkingPipeline
from app.services.retriever import Retriever


def test_retriever_search_returns_relevant_chunk():
    chunker = ChunkingPipeline()
    chunks = chunker.chunk("kb", "# Refunds\n\nRefunds are processed within 5 business days.")
    retriever = Retriever()
    retriever.index(chunks)

    results = retriever.search("How long do refunds take?", top_k=1)

    assert results
    assert results[0].chunk.section == "Refunds"
