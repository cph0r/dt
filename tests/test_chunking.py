from app.ingestion.chunking import ChunkingPipeline


def test_chunking_tracks_heading_metadata():
    chunks = ChunkingPipeline().chunk("help", "# Billing\n\nInvoices are emailed monthly.\n\n# Access\n\nReset your password.")

    assert chunks[0].section == "Billing"
    assert chunks[0].metadata["source"] == "help"
