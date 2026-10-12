"""Explicit derived-index installation and its bounded lexical consumer.

The normal store never installs this schema automatically. Persistent triggers
use SQLite built-ins so older writers can maintain it without Python UDFs.
"""
from __future__ import annotations
import json
import struct
from types import SimpleNamespace
from . import memory_store as core

TABLE = "memory_logical_tags_v1"
FTS = "memory_logical_tags_fts_v1"
PROJECTION = "memory_tags_search_v1"
TRIGGERS = ("memory_logical_tags_insert_v1", "memory_logical_tags_delete_v1",
            "memory_logical_tags_update_v1", "memory_logical_tags_fts_insert_v1",
            "memory_logical_tags_fts_delete_v1")


def _array(expression):
    return f"CASE WHEN json_valid({expression}) THEN CASE WHEN json_type({expression})='array' THEN {expression} ELSE '[]' END ELSE '[]' END"


def _projection(expression):
    return f"COALESCE((SELECT group_concat(value,char(31)) FROM json_each({_array(expression)}) WHERE type='text'),'')"


def available(store):
    return store._table_exists(TABLE)


def _tags(raw):
    if isinstance(raw,list):return [tag for tag in raw if isinstance(tag,str)]
    try:values=json.loads(raw or '[]')
    except (ValueError,TypeError):return []
    return [tag for tag in values if isinstance(tag,str)] if isinstance(values,list) else []


def coverage_score(content,raw,query):
    patterns=core._memory_search_patterns(query)
    if not patterns:return 0.
    text=(content or '').lower()
    missing=[pattern for pattern in patterns if pattern.search(text) is None]
    if missing:
        for tag in _tags(raw):
            missing=[pattern for pattern in missing if pattern.search(tag.lower()) is None]
            if not missing:break
    return (len(patterns)-len(missing))/len(patterns)


def _logical_match(queries_json,raw):
    queries=json.loads(queries_json)
    patterns=[p for query in queries for p in core._memory_search_patterns(query)]
    return int(any(p.search(tag.lower()) for tag in _tags(raw) for p in patterns))


def _rank_key(content,raw,start,occurred,created,updated,accessed,kind,strength,count,query,now):
    text=coverage_score(content,raw,query)
    memory=SimpleNamespace(memory_type=kind or '',strength=strength,last_accessed_at=accessed,
                           updated_at=updated or created,created_at=created)
    values=core._search_rank_values(text,core._recency_score(start or occurred or created,now),
        core._clamp_score(core._effective_strength_policy(memory,now)['effective_strength']),
        core._access_score(count),created)
    packed=struct.pack('>5d',*values)
    tail=bytes(b^0xff for b in packed[32:]) if created<0 else bytes([packed[32]^0x80])+packed[33:]
    return packed[:32]+tail


def _register(store):
    if not getattr(store,'_logical_search_functions_registered',False):
        store._conn.create_function('memory_logical_tag_match',2,_logical_match)
        store._conn.create_function('memory_logical_rank_key',12,_rank_key)
        store._logical_search_functions_registered=True


def install(store,*,with_fts=True,fault=None):
    """Explicit only: call on a temporary copy, or an approved target database."""
    conn=store._conn
    columns={r[1] for r in conn.execute('PRAGMA table_info(memories)')}
    if available(store) or PROJECTION in columns:
        required=TRIGGERS if with_fts else TRIGGERS[:3]
        if not available(store) or PROJECTION not in columns or any(not conn.execute("SELECT 1 FROM sqlite_master WHERE type='trigger' AND name=?",(n,)).fetchone() for n in required) or (with_fts and not store._table_exists(FTS)):
            raise RuntimeError('Incomplete logical-tag schema; explicit rollback required')
        return False
    conn.execute("SELECT value FROM json_each('[\"probe\"]')").fetchone()
    conn.execute('SAVEPOINT logical_tag_install')
    try:
        statements=[f'CREATE TABLE {TABLE}(id INTEGER PRIMARY KEY,memory_rowid INTEGER NOT NULL,ordinal INTEGER NOT NULL,tag TEXT NOT NULL,UNIQUE(memory_rowid,ordinal))',
                    f"ALTER TABLE memories ADD COLUMN {PROJECTION} TEXT NOT NULL DEFAULT ''"]
        if with_fts:
            statements += [f"CREATE VIRTUAL TABLE {FTS} USING fts5(tag,content='{TABLE}',content_rowid='id')",
                f'CREATE TRIGGER {TRIGGERS[3]} AFTER INSERT ON {TABLE} BEGIN INSERT INTO {FTS}(rowid,tag) VALUES(new.id,new.tag); END',
                f"CREATE TRIGGER {TRIGGERS[4]} AFTER DELETE ON {TABLE} BEGIN INSERT INTO {FTS}({FTS},rowid,tag) VALUES('delete',old.id,old.tag); END"]
        statements += [
            f"CREATE TRIGGER {TRIGGERS[0]} AFTER INSERT ON memories BEGIN INSERT INTO {TABLE}(memory_rowid,ordinal,tag) SELECT new.rowid,key,value FROM json_each({_array('new.tags')}) WHERE type='text' AND new.deleted_at IS NULL; UPDATE memories SET {PROJECTION}={_projection('new.tags')} WHERE rowid=new.rowid; END",
            f'CREATE TRIGGER {TRIGGERS[1]} AFTER DELETE ON memories BEGIN DELETE FROM {TABLE} WHERE memory_rowid=old.rowid; END',
            f"CREATE TRIGGER {TRIGGERS[2]} AFTER UPDATE OF tags,deleted_at ON memories WHEN old.tags IS NOT new.tags OR old.deleted_at IS NOT new.deleted_at BEGIN DELETE FROM {TABLE} WHERE memory_rowid=old.rowid; INSERT INTO {TABLE}(memory_rowid,ordinal,tag) SELECT new.rowid,key,value FROM json_each({_array('new.tags')}) WHERE type='text' AND new.deleted_at IS NULL; UPDATE memories SET {PROJECTION}={_projection('new.tags')} WHERE rowid=new.rowid; END",
            f'UPDATE memories SET {PROJECTION}={_projection("tags")}',
            f"INSERT INTO {TABLE}(memory_rowid,ordinal,tag) SELECT m.rowid,j.key,j.value FROM memories m,json_each({_array('m.tags')}) j WHERE m.deleted_at IS NULL AND j.type='text'"]
        for i,sql in enumerate(statements):
            conn.execute(sql)
            if fault:fault(i,conn)
        conn.execute('RELEASE logical_tag_install')
    except BaseException:
        conn.execute('ROLLBACK TO logical_tag_install');conn.execute('RELEASE logical_tag_install');raise
    return True


def rollback(store):
    conn=store._conn;conn.execute('SAVEPOINT logical_tag_rollback')
    try:
        for name in TRIGGERS:conn.execute(f'DROP TRIGGER IF EXISTS {name}')
        conn.execute(f'DROP TABLE IF EXISTS {FTS}');conn.execute(f'DROP TABLE IF EXISTS {TABLE}')
        if PROJECTION in {r[1] for r in conn.execute('PRAGMA table_info(memories)')}:conn.execute(f'ALTER TABLE memories DROP COLUMN {PROJECTION}')
        conn.execute('RELEASE logical_tag_rollback')
    except BaseException:
        conn.execute('ROLLBACK TO logical_tag_rollback');conn.execute('RELEASE logical_tag_rollback');raise


def search(store,user_id,queries,*,limit,subject_ids,kind,exclude_memory_types):
    _register(store)
    subjects=store._canonical_subject_ids(user_id,subject_ids)
    if subject_ids is not None and not subjects:return core.MemorySearchResult(memories=[],ranking=[])
    scope=['m.user_id=?','m.status=?','m.deleted_at IS NULL'];params=[user_id,core.ACTIVE_MEMORY_STATUS]
    if subject_ids is not None:scope.append(f"m.subject_id IN ({','.join('?' for _ in subjects)})");params.extend(subjects)
    if kind:scope.append('m.kind=?');params.append(core.normalize_memory_kind(kind))
    excluded=sorted({core.normalize_memory_type(t) for t in exclude_memory_types if t})
    if excluded:scope.append(f"m.memory_type NOT IN ({','.join('?' for _ in excluded)})");params.extend(excluded)
    content=[];tags=[];content_params=[];tag_params=[]
    for term in dict.fromkeys(term for q in queries for term in store._fallback_terms(q)):
        content.append('(m.content LIKE ? AND memory_term_match(?,m.content))');content_params.extend([f'%{term}%',core._search_term_pattern(term)])
        tags.append(f'm.{PROJECTION} LIKE ?');tag_params.append(f'%{term}%')
    branches=[];branch_params=[]
    for query in queries:
        fts=store._fts_query(query)
        if store._table_exists('memories_fts'):
            branches.append('SELECT rowid FROM memories_fts WHERE memories_fts MATCH ?');branch_params.append('content : ('+fts+')')
        if store._table_exists(FTS):
            branches.append(f'SELECT memory_rowid AS rowid FROM {TABLE} WHERE id IN (SELECT rowid FROM {FTS} WHERE {FTS} MATCH ?)');branch_params.append(fts)
    qualification=f"(({' OR '.join(content) or '0'}) OR (({' OR '.join(tags) or '0'}) AND memory_logical_tag_match(?,m.tags)))"
    branches.append(f"SELECT m.rowid FROM memories m WHERE {' AND '.join(scope)} AND {qualification}")
    now=core.time.time()
    keys=['memory_logical_rank_key(m.content,m.tags,m.start_at,m.occurred_at,m.created_at,m.updated_at,m.last_accessed_at,m.memory_type,m.strength,m.access_count,?,?)' for q in queries]
    rank=keys[0] if len(keys)==1 else 'MAX('+','.join(keys)+')'
    rows=store._conn.execute(f"SELECT m.* FROM ({' UNION '.join(branches)}) qualified CROSS JOIN memories m WHERE m.rowid=qualified.rowid AND {' AND '.join(scope)} ORDER BY {rank} DESC,m.rowid ASC LIMIT ?",(*branch_params,*params,*content_params,*tag_params,json.dumps(queries),*params,*[v for q in queries for v in (q,now)],4*limit)).fetchall()
    return store._rank_memory_search_results([store._row_to_event(r) for r in rows],queries[0],limit,now=now,search_queries=queries[1:],logical_fields=True)
