"""Concurrent reads/writes on one store must finish and preserve both users."""
from concurrent.futures import ThreadPoolExecutor
from ai_glasses_memory_assistant.memory_store import EventMemoryStore


def test_shared_connection_complete_operations(tmp_path,monkeypatch):
    monkeypatch.setenv('AI_GLASSES_HOME',str(tmp_path/'home'))
    s=EventMemoryStore(tmp_path/'db')
    try:
        def operation(i):
            user=f'u{i%2}'
            if i%3==0:return user,s.add_memory(user,f'anchor record {i}').id
            assert all(m.user_id==user for m in s.search_with_ranking(user,'anchor').memories)
            return None
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures=[executor.submit(operation,i) for i in range(90)]
            written=[value for f in futures if (value:=f.result(timeout=10))]
        assert len(written)==30
        for user,memory_id in written:assert s.get_memory(user,memory_id) is not None
        assert s._conn.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        s._conn.execute("INSERT INTO memories_fts(memories_fts,rank) VALUES('integrity-check',1)")
    finally:s._conn.close()
