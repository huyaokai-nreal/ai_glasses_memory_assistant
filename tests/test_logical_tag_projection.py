"""Normal-store integration: explicit installation, legacy writers, max rank."""
import json,sqlite3
import pytest
from ai_glasses_memory_assistant.memory_store import EventMemoryStore
from ai_glasses_memory_assistant.logical_tag_search import TABLE,FTS,PROJECTION,TRIGGERS
from tests.helpers import CoreChatService,pre_reply_recall

@pytest.fixture
def store(tmp_path,monkeypatch):
    monkeypatch.setenv('AI_GLASSES_HOME',str(tmp_path/'home'))
    s=EventMemoryStore(tmp_path/'db');yield s;s._conn.close()

@pytest.mark.parametrize('fts',[True,False])
def test_logical_tags_and_shared_cjk_document_helper(store,fts):
    target=store.add_memory('u','archive',tags=['凝胶黏度','quote"fern','ivy\\path'],created_at=1)
    store.add_memory('u','other',tags=['ice','cream'])
    store.add_memory('u','other',tags=['杉'])
    store.install_logical_tag_projection(with_fts=fts)
    if not fts:
        for n in ('memories_fts_insert','memories_fts_update','memories_fts_delete'):store._conn.execute(f'DROP TRIGGER {n}')
        store._conn.execute('DROP TABLE memories_fts')
    for query in ['请查凝胶黏度结果','quote"fern','ivy\\path']:
        assert target.id in [m.id for m in store.search_with_ranking('u',query).memories]
    assert not store.search_with_ranking('u','icecream').memories
    assert not store.search_with_ranking('u','u6749').memories
    d=store.add_document('u',filename='synthetic.md',title='材料',summary='检测',content='凝胶黏度检测',source='test',ingestion_id='fixed')
    store._conn.execute('DROP TABLE documents_fts')
    assert d.id in [x.id for x in store.search_documents('u','请查凝胶黏度结果')]

@pytest.mark.parametrize('fault_index',range(10))
def test_atomic_install_failure_leaves_original_data(store,fault_index):
    m=store.add_memory('u','kept',tags=['杉'])
    def fault(index,conn):
        if index==fault_index:raise RuntimeError('injected')
    with pytest.raises(RuntimeError):store.install_logical_tag_projection(fault=fault)
    assert not store._table_exists(TABLE)
    assert PROJECTION not in {r[1] for r in store._conn.execute('PRAGMA table_info(memories)')}
    assert store.get_memory('u',m.id).content=='kept'


def test_no_auto_install_old_writer_and_no_access_rebuild(store):
    assert not store._table_exists(TABLE)
    m=store.add_memory('u','kept',tags=['雪松']);store.install_logical_tag_projection()
    writer=sqlite3.connect(store.db_path)
    try:
        writer.execute('UPDATE memories SET tags=? WHERE id=?',(json.dumps(['槐树']),m.id));writer.commit()
        assert store.search_with_ranking('u','槐树').memories[0].id==m.id
        before=store._conn.total_changes
        store.record_memory_access('u',[m.id])
        assert store._conn.total_changes-before==1
        writer.execute('UPDATE memories SET deleted_at=1 WHERE id=?',(m.id,));writer.commit()
        assert not store.search_with_ranking('u','槐树').memories
        writer.execute('UPDATE memories SET deleted_at=NULL WHERE id=?',(m.id,));writer.commit()
        assert store.search_with_ranking('u','槐树').memories
        store.rollback_logical_tag_projection()
        assert not store._table_exists(TABLE) and store.get_memory('u',m.id).content=='kept'
    finally:writer.close()

@pytest.mark.parametrize('fts',[True,False])
def test_rewrite_max_before_budget_with_subject_scope(store,fts):
    subject=store.create_named_subject('u','Mira')
    own=store.ensure_self_subject('u')
    target=store.add_memory('u','kept',tags=['cedar receipt'],created_at=1,subject_id=subject.id)
    candidates=[target]
    for i in range(80):candidates.append(store.add_memory('u',f'birch invoice {i}',subject_id=subject.id))
    for i in range(25):store.add_memory('u','cedar receipt '+str(i),subject_id=own.id)
    store.install_logical_tag_projection(with_fts=fts)
    if not fts:
        for n in ('memories_fts_insert','memories_fts_update','memories_fts_delete'):store._conn.execute(f'DROP TRIGGER {n}')
        store._conn.execute('DROP TABLE memories_fts')
    queries=['cedar receipt','birch invoice']
    result=store.search_with_ranking('u','unrelated wording',search_queries=queries,subject_ids=[subject.id],limit=5)
    reference=store._rank_memory_search_results(candidates,'unrelated wording',5,search_queries=queries,logical_fields=True)
    assert [m.id for m in result.memories]==[m.id for m in reference.memories]
    assert len(result.candidate_ranking)<=20 and all(m.subject_id==subject.id for m in result.memories)


def test_actual_service_consumer_after_explicit_install(tmp_path,monkeypatch):
    monkeypatch.setenv('AI_GLASSES_HOME',str(tmp_path/'home'))
    service=CoreChatService(str(tmp_path))
    try:
        service.memory_store.install_logical_tag_projection()
        target=service.memory_store.add_memory('u','archived cjk target',tags=['凝胶黏度'],created_at=1)
        for i in range(30):service.memory_store.add_memory('u',f'unrelated new entry {i}')
        service.fake_agent.pre_reply=pre_reply_recall()
        result=service.chat('请查凝胶黏度结果',user_id='u')
        assert target.id in [m['id'] for m in result['recalled_memories']]
        assert result['debug']['memory']['event_recall']['candidate_trace']['query_term_budgets']
    finally:service.close()
