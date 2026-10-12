"""Deterministic plumbing tests; these do not measure real PPD quality."""
import pytest
from ai_glasses_memory_assistant.memory_store import EventMemoryStore
from ai_glasses_memory_assistant.memory_recall_arbitration import normalize_memory_search_queries
from ai_glasses_memory_assistant.turn_semantic_classifier import _decision_from_payload
from tests.helpers import pre_reply_recall, CoreChatService


def test_rewrites_fail_closed_and_do_not_coerce_values():
    assert normalize_memory_search_queries('cedar') == []
    assert normalize_memory_search_queries([None, 'x'*129, {'q':'cedar'}, 'cedar']) == []
    assert normalize_memory_search_queries([' cedar ', 'cedar', 'birch', 'fourth']) == ['cedar','birch']
    payload = pre_reply_recall()
    payload['memory_search_queries'] = ['cedar receipt']
    decision = _decision_from_payload(payload, raw='{}', backend='test')
    assert decision.memory_search_queries == ['cedar receipt']
    payload.update(memory_action='none', needs_event_memory=False, needs_profile_memory=False,
                   memory_recall_type='none', recall_type='none')
    assert _decision_from_payload(payload, raw='{}', backend='test').memory_search_queries == []


@pytest.mark.parametrize('without_fts',[False,True])
@pytest.mark.parametrize('kind',['event','profile'])
def test_union_max_score_budget_and_scope(tmp_path,monkeypatch,without_fts,kind):
    now=1778131200.
    monkeypatch.setattr('ai_glasses_memory_assistant.memory_store.time.time',lambda:now)
    store=EventMemoryStore(tmp_path/'db')
    try:
        target=store.add_memory('u','cedar receipt supporting detail '*30,kind=kind,
                                created_at=now-365*86400)
        eligible=[target]
        for i in range(85):
            eligible.append(store.add_memory('u',f'birch invoice item {i}',kind=kind,created_at=now-i))
        store.add_memory('other','cedar receipt',kind=kind)
        store.add_memory('u','cedar receipt',kind='profile' if kind=='event' else 'event')
        store.add_memory('u','cedar receipt',kind=kind,status='superseded')
        if without_fts:store._conn.execute('DROP TABLE memories_fts')
        queries=['cedar receipt','birch invoice']
        assert store.search_with_ranking('u','unrelated wording',kind=kind).memories==[]
        result=store.search_with_ranking('u','unrelated wording',kind=kind,search_queries=queries,limit=5)
        reference=store._rank_memory_search_results(eligible,'unrelated wording',5,now=now,search_queries=queries)
        assert [m.id for m in result.memories]==[m.id for m in reference.memories]
        assert len(result.candidate_ranking)<=20
        assert result.ranking==reference.ranking
        assert all(m.user_id=='u' and m.kind==kind and m.status=='active' for m in result.memories)
    finally:store._conn.close()


def test_service_passes_rewrite_to_real_search_and_reader(tmp_path,monkeypatch):
    monkeypatch.setenv("AI_GLASSES_HOME",str(tmp_path/"home"))
    service=CoreChatService(str(tmp_path))
    try:
        payload=pre_reply_recall()
        payload['memory_search_queries']=['cedar receipt']
        service.fake_agent.pre_reply=payload
        target=service.memory_store.add_memory('u','cedar receipt was stored in the blue folder',created_at=1)
        for i in range(30):service.memory_store.add_memory('u',f'new unrelated entry {i}')
        response=service.chat(user_id='u',message='Where was that proof of purchase?')
        assert target.id in [m['id'] for m in response['recalled_memories']]
        main=[c for c in service.fake_agent.calls if 'cedar receipt was stored' in str(c['message'])]
        assert main
        assert sum('unified pre-reply decision classifier' in str(c['system_message']) for c in service.fake_agent.calls)==1
    finally:service.close()
