from app.ingestion.chunking import ChunkingPipeline
from app.services.retriever import Retriever
from app.services.vector_store import SQLiteVectorStore


def test_vector_store_persists_chunks(tmp_path):
    database_path = tmp_path / "vectors.sqlite3"
    store = SQLiteVectorStore(str(database_path))
    retriever = Retriever(store=store)
    retriever.index(ChunkingPipeline().chunk("kb", "# Billing\n\nInvoices are emailed monthly."))

    reloaded = Retriever(store=SQLiteVectorStore(str(database_path)))
    results = reloaded.search("When are invoices sent?", top_k=1)

    assert results
    assert results[0].chunk.section == "Billing"
