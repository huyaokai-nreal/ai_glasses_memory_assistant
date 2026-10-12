"""Typed capsule evidence obeys the applied subject contract at consumption."""
import json
import tempfile
import pytest
from tests.helpers import CoreChatService,FakeAgent,isolated_app_home,pre_reply_recall

@pytest.mark.parametrize('scope',['self','named','all','unresolved'])
@pytest.mark.parametrize('source',['recent','query'])
@pytest.mark.parametrize('merged',[False,True])
def test_capsule_scopes_same_bounded_candidates_without_changing_routing(scope,source,merged):
    with tempfile.TemporaryDirectory() as folder,isolated_app_home(folder):
        service=CoreChatService(folder)
        store=service.memory_store
        own=store.add_memory('u1','My Sable tray marker is A643.',created_at=1 if source=='query' else 100)
        named=store.add_memory('u1','Iris has Sable tray marker N572.',subject_type='named',subject_name='Iris',created_at=2 if source=='query' else 101)
        other=store.add_memory('u1','Emre has Sable tray marker E861.',subject_type='named',subject_name='Emre',created_at=3 if source=='query' else 102)
        if source=='query':
            for n in range(10):store.add_memory('u1',f'Unrelated record {n}.',created_at=100+n)
        original=service._build_recent_context_capsule(user_id='u1',query='Sable tray marker')
        assert all(m.content in original['text'] for m in [own,named,other])
        initial=original['text']
        if merged:
            original=service._merge_ambient_context_into_capsule(original,{'text':'Ambient independent background.','debug':{}})
        allowed=[own.subject_id] if scope=='self' else [named.subject_id] if scope=='named' else None if scope=='all' else []
        sql=[];store._conn.set_trace_callback(sql.append)
        qualified=service._qualify_recent_context_capsule(original,subject_ids=allowed)
        store._conn.set_trace_callback(None)
        assert not sql
        for memory in [own,named,other]:
            assert (memory.content in qualified['text']) == (allowed is None or memory.subject_id in allowed)
        assert qualified['debug']['structured_subject_candidates']>=3
        assert (('Ambient independent background.' in qualified['text'])==merged)
        assert initial in original['text']
        assert '_parts' in qualified
        json.dumps(qualified['debug'])  # Only counts, no MemoryEvent objects in public debug.
        service.close()


def test_capsule_consumption_preserves_ppd_background_and_unrelated_sections():
    with tempfile.TemporaryDirectory() as folder,isolated_app_home(folder):
        agent=FakeAgent(pre_reply={**pre_reply_recall(),'temporal_query':{'has_expression':False}})
        service=CoreChatService(folder,agent)
        own=service.memory_store.add_memory('u1','我的 Azure 工具编号是 V381。')
        other=service.memory_store.add_memory('u1','若谷的 Azure 工具编号是 F964。',subject_type='named',subject_name='若谷')
        result=service.chat('我的 Azure 工具编号是什么？',user_id='u1')
        ppd=next(c for c in agent.calls if c['system_message'] and 'unified pre-reply decision classifier' in c['system_message'])
        main=next(c for c in reversed(agent.calls) if not c['system_message'])
        assert other.content in ppd['message']
        assert own.content in main['message'] and other.content not in main['message']
        assert result['debug']['recent_context_capsule']['structured_subject_filtered']>=1
        json.dumps(result,ensure_ascii=False)
        service.close()
