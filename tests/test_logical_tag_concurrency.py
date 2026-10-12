"""Derived indexes must commit consistently with simultaneous store writers."""
import json
from concurrent.futures import ThreadPoolExecutor
import pytest
from ai_glasses_memory_assistant.memory_store import EventMemoryStore
from ai_glasses_memory_assistant.logical_tag_search import TABLE,FTS,PROJECTION

@pytest.mark.parametrize('fts',[False,True])
def test_concurrent_tag_writers_and_index_integrity(tmp_path,fts):
    store=EventMemoryStore(tmp_path/'events.db')
    try:
        store.install_logical_tag_projection(with_fts=fts)
        def write(i):
            user=f'u{i%2}'
            event=store.add_memory(user,f'evidence {i}',tags=[f'cedar{i:03d}','凝胶黏度'])
            assert event.id in [m.id for m in store.search(user,f'cedar{i:03d}')]
            assert not store.search(f'u{1-i%2}',f'cedar{i:03d}')
            store.record_memory_access(user,[event.id])
            return event.id
        with ThreadPoolExecutor(max_workers=4) as pool:
            ids=list(pool.map(write,range(50)))
        assert len(set(ids))==50
        for row in store._conn.execute(f'SELECT rowid,tags,{PROJECTION} FROM memories'):
            expected=[x for x in json.loads(row['tags']) if isinstance(x,str)]
            actual=[r[0] for r in store._conn.execute(f'SELECT tag FROM {TABLE} WHERE memory_rowid=? ORDER BY ordinal',(row['rowid'],))]
            assert actual==expected and row[PROJECTION]=='\x1f'.join(expected)
        if fts:
            store._conn.execute(f"INSERT INTO {FTS}({FTS},rank) VALUES('integrity-check',1)")
        store._conn.execute("INSERT INTO memories_fts(memories_fts,rank) VALUES('integrity-check',1)")
        assert store._conn.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    finally:
        store._conn.close()
