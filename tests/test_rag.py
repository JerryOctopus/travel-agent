from travel_agent.rag import retrieve_destination_knowledge


def test_retrieve_destination_knowledge_returns_city_chunks() -> None:
    chunks = retrieve_destination_knowledge("北京", "历史 美食 雨天")

    assert chunks
    assert chunks[0].city == "北京"
    assert chunks[0].score > 0
    assert "北京" in chunks[0].source or "beijing" in chunks[0].source


def test_retrieve_destination_knowledge_returns_empty_for_unknown_city() -> None:
    assert retrieve_destination_knowledge("不存在", "历史") == []
