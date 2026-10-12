from ai_glasses_memory_assistant.memory_store import EventMemoryStore


def test_mechanical_han_budget_and_english_unchanged():
    payload=EventMemoryStore._fallback_term_payload('甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥')
    assert len(payload['terms'])==33 and payload['truncated']
    assert payload['terms'][1:4]==['甲乙丙丁','乙丙丁戊','丙丁戊己']
    assert len(set(payload['terms']))==len(payload['terms'])
    assert EventMemoryStore._fallback_terms('the copper receipt')==['copper','receipt']
    assert EventMemoryStore._fallback_terms('氮')==['氮']
    assert EventMemoryStore._fallback_terms('甲乙 x 丙丁')==['甲乙','x','丙丁']
    assert EventMemoryStore._fallback_terms('甲乙丙丁')==['甲乙丙丁','甲乙丙','乙丙丁','甲乙','乙丙','丙丁']


def test_shared_document_and_tag_search(tmp_path,monkeypatch):
    monkeypatch.setenv("AI_GLASSES_HOME",str(tmp_path/"home"))
    s=EventMemoryStore(tmp_path/"db")
    try:
        m=s.add_memory("u","旧记录",tags=["凝胶黏度"])
        d=s.add_document("u",filename="test.md",title="检测",summary="材料",content="凝胶黏度检测",source="test",ingestion_id="synthetic")
        s.install_logical_tag_projection(with_fts=False)
        s._conn.execute("DROP TABLE memories_fts")
        s._conn.execute("DROP TABLE documents_fts")
        assert m.id in [x.id for x in s.search_with_ranking("u","请查凝胶黏度结果").memories]
        assert d.id in [x.id for x in s.search_documents("u","请查凝胶黏度结果")]
    finally:s._conn.close()
