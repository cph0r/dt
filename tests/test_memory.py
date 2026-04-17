from app.services.memory import ConversationMemory


def test_memory_sliding_window_and_summary_rollup():
    memory = ConversationMemory(window_size=4, summary_trigger=5, max_summary_chars=300)
    conversation_id = "conv-memory"

    for idx in range(8):
        memory.append(conversation_id, "user", f"question-{idx}")
        memory.append(conversation_id, "assistant", f"answer-{idx}")

    snapshot = memory.get_snapshot(conversation_id)

    assert len(snapshot.turns) == 4
    assert snapshot.summary.startswith("Summary:")
    assert "question-0" in snapshot.summary
    assert snapshot.turns[-1]["content"] == "answer-7"
