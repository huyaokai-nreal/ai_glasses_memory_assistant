"""Serial, fail-closed acceptance runner. All datasets here are synthetic."""
from __future__ import annotations
import argparse, hashlib, json, os, secrets, shutil, subprocess, sys, threading, time, urllib.request
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = Path(os.environ.get("AI_GLASSES_ACCEPTANCE_RUN_DIR", ROOT / ".planning/2026-10-12-live-closeout")).resolve()
OLD = ROOT / '.planning/2026-10-12-continuous-recall-closeout'
AUTH = 'http://127.0.0.1:27901'
MODEL_URL = 'http://10.252.17.5:11438/v1'
CONFIG = dict(provider='llama_cpp', model='qwen3.8-27b-32k', base_url=MODEL_URL,
              api_key='ollama', transport='stdlib_http', api_mode='chat_completions')
SERIAL = os.environ.get('AI_GLASSES_ACCEPTANCE_DEVICE_SERIAL', '10.2.35.115:39865')
ADB = '/opt/homebrew/share/android-commandlinetools/platform-tools/adb'
CASES = json.loads((OLD / 'model-conditions.json').read_text()) if (OLD / 'model-conditions.json').exists() else []
# Frozen before live requests. Expected content/hints are never inserted into prompts.
EXPERIENCE = [
 dict(id='generic', question='请用一句话解释为什么天空看起来是蓝色的。'),
 dict(id='old_event', question='铜色收据放在哪里？', content='铜色收据放在黄色文件夹中。', kind='event', created_at=1, noise_rows=30),
 dict(id='dated', question='昨天完成了哪项检查？', content='10月11日完成了焊料检查。', stored_start_iso='2026-10-11T09:00:00+08:00'),
 dict(id='profile', question='我喜欢什么休闲活动？', content='我喜欢在周末做陶艺。', kind='profile', memory_type='preference', created_at=1, noise_rows=30),
 dict(id='sync_write', question='请记住：我喜欢在周六上午整理书架。'),
 dict(id='background_write', question='我刚为工作室买了一台蓝色落地灯，下周要放到阅读角。'),
]

def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def token(): return (RUN / 'authority-token').read_text().strip()

def request(path, body=None, base=AUTH, timeout=15, authenticated=True):
    headers = {'Content-Type': 'application/json'}
    if authenticated: headers['X-Acceptance-Token'] = token()
    req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None, headers=headers)
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=timeout) as res:
        return json.load(res)

def variants():
    values = [f'quality/{c["id"]}/{arm}' for c in CASES for arm in ['before', 'after']]
    for platform in ['web', 'android']:
        for c in EXPERIENCE:
            for repeat in range(3):
                values.extend(f'{platform}/{c["id"]}/{repeat}/{arm}' for arm in ['before','after'])
        values.extend(f'{platform}/generic/cold/{arm}' for arm in ['before','after'])
    return values

def freeze():
    if (RUN/'freeze.json').exists(): raise RuntimeError('Already frozen; refuse overwrite')
    (RUN/'authority-token').write_text(secrets.token_hex(32))
    os.chmod(RUN/'authority-token', 0o600)
    for arm in ['before','after']:
        dest = RUN/'projects'/arm
        source = OLD/'experience/baseline-instrumented-project' if arm=='before' else ROOT
        for name in ['ai_glasses_memory_assistant','static','android']:
            shutil.copytree(source/name, dest/name, ignore=shutil.ignore_patterns('build','.gradle','__pycache__','*.pyc','.DS_Store'))
        for name in ['llm_client.py','llm_runtime.py','acceptance_runtime.py','android_runtime.py']:
            shutil.copy2(ROOT/'ai_glasses_memory_assistant'/name, dest/'ai_glasses_memory_assistant'/name)
        for name in ['app/build.gradle.kts','app/src/main/java/com/aiglasses/memoryassistant/demo/MainActivity.kt',
                     'app/src/main/java/com/aiglasses/memoryassistant/demo/PythonRuntime.kt']:
            shutil.copy2(ROOT/'android'/name, dest/'android'/name)
        hashes = {str(p.relative_to(dest)):digest(p) for p in dest.rglob('*') if p.is_file()}
        dump(RUN/'sources'/f'{arm}.json', hashes)
    assert len(variants())==124
    dump(RUN/'freeze.json', dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        config=CONFIG, model_authorized=True, device_authorized=True, real_database_install_authorized=False,
        limit=640, per_variant=5, variants=variants(), conditions=CASES, experience=EXPERIENCE,
        conditions_sha256=digest(OLD/'model-conditions.json'), case_freeze='after implementation; not blind holdout',
        p95_method='nearest-rank', quality_gates=dict(positive=11,zero_overlap=7,holdout_positive=5,wrong_flips=0),
        experience_gates=dict(normal_failures=0,p95_ratio=1.1,controls_ready_ms=100),
        ordering='before then after, serial; no re-run; all background drained before variant change'))

class Ledger:
    def __init__(self, directory=RUN, allowed=None, *, prior_ledger=None, local_limit=640):
        self.directory=directory; self.path=directory/'requests.jsonl'; self.lock=threading.RLock()
        self.allowed=set(allowed or variants()); self.rows=[]; self.current=None; self.used=set(); self.pending=set()
        self.count=Counter(); self.ids=set()
        self.prior_total=0; self.prior_variants=0; self.prior_ledger=prior_ledger; self.local_limit=local_limit
        if prior_ledger is not None:
            path, expected_hash = prior_ledger
            if digest(path) != expected_hash: raise ValueError('Archived ledger hash mismatch')
            archived=[json.loads(line) for line in path.read_text().splitlines()]
            reserves=[row for row in archived if row['type']=='reserve']
            keys={row['request_id'] for row in reserves}
            finished={row['request_id'] for row in archived if row['type']=='finish'}
            if len(keys)!=len(reserves) or keys-finished or (archived and archived[-1]['type']!='end'):
                raise ValueError('Archived ledger is not fully sealed')
            prior_counts=Counter(row['variant'] for row in reserves)
            if len(reserves)>640 or any(count>5 for count in prior_counts.values()):
                raise ValueError('Archived ledger exceeded budget')
            self.prior_total=len(reserves)
            self.prior_variants=len({row['variant'] for row in archived if row['type']=='begin'})
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                row=json.loads(line); self.rows.append(row)
                if row['type']=='begin': self.used.add(row['variant']); self.current=row['variant']
                if row['type']=='reserve':
                    self.count[row['variant']]+=1; self.pending.add(row['request_id']); self.ids.add(row['request_id'])
                if row['type']=='finish': self.pending.discard(row['request_id'])
                if row['type']=='end': self.current=None
    def append(self, row):
        with self.path.open('a') as file:
            file.write(json.dumps(dict(row,at=time.time()),ensure_ascii=False)+'\n'); file.flush(); os.fsync(file.fileno())
        self.rows.append(row)
    def dispatch(self, path, body):
        with self.lock:
            if path=='/status': return dict(current=self.current,total=self.prior_total+sum(self.count.values()),local_total=sum(self.count.values()),prior_total=self.prior_total,counts=dict(self.count),pending=sorted(self.pending),used=sorted(self.used))
            if path=='/begin':
                variant=body['variant']
                if self.current is not None or self.pending or variant in self.used or variant not in self.allowed:
                    raise ValueError('Variant cannot start: active, pending, used, or not frozen')
                if self.prior_variants+len(self.used)>=124:
                    raise ValueError('Global variant budget exhausted')
                self.append(dict(type='begin',variant=variant)); self.current=variant; self.used.add(variant)
                return dict(variant=variant)
            if path=='/reserve':
                if self.prior_ledger is not None and digest(self.prior_ledger[0])!=self.prior_ledger[1]:
                    raise ValueError('Archived ledger changed; stop before sending')
                if (not self.current or self.count[self.current]>=5 or sum(self.count.values())>=self.local_limit
                        or self.prior_total+sum(self.count.values())>=640):
                    raise ValueError('Generation budget exhausted or no active variant')
                client=body['client']
                if client!=dict(provider='llama_cpp',model=CONFIG['model'],base_url=MODEL_URL) or body['request'].get('model')!=CONFIG['model']:
                    raise ValueError('Wrong generation destination/model')
                request_id=secrets.token_hex(12)
                self.append(dict(type='reserve',variant=self.current,request_id=request_id,**body))
                self.count[self.current]+=1; self.pending.add(request_id); self.ids.add(request_id)
                return dict(request_id=request_id)
            if path=='/finish':
                key=body['request_id']
                if key not in self.pending: raise ValueError('Unknown or already finished request')
                self.append(dict(type='finish',variant=self.current,**body)); self.pending.remove(key)
                return dict(ok=True)
            if path=='/end':
                if self.pending or not self.current: raise ValueError('Cannot end pending/inactive variant')
                self.append(dict(type='end',variant=self.current,**body)); self.current=None
                return dict(ok=True)
            raise ValueError('Unknown authority operation')

def authority_server(ledger=None):
    ledger=ledger if ledger is not None else Ledger()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self): self.handle_request()
        def do_POST(self): self.handle_request()
        def log_message(self,*args): pass
        def handle_request(self):
            code=200
            try:
                if not secrets.compare_digest(self.headers.get('X-Acceptance-Token',''),token()): raise ValueError('Unauthorized')
                body=json.loads(self.rfile.read(int(self.headers.get('Content-Length',0))) or b'{}')
                result=ledger.dispatch(self.path,body)
            except Exception as exc: code=409; result=dict(error=str(exc))
            data=json.dumps(result).encode(); self.send_response(code); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
    print('authority ready; no generation forwarding',flush=True)
    ThreadingHTTPServer(('127.0.0.1',27901),Handler).serve_forever()

def environment(arm, home):
    env=os.environ.copy()
    for key,value in CONFIG.items(): env['AI_GLASSES_LLM_'+key.upper()]=value
    env.update(AI_GLASSES_ACCEPTANCE_LEDGER_URL=AUTH, AI_GLASSES_ACCEPTANCE_TOKEN=token(),
               AI_GLASSES_HOME=str(home),AI_GLASSES_STATIC_DIR=str(RUN/'projects'/arm/'static'),
               NO_PROXY='*',PYTHONPATH=str(RUN/'projects'/arm))
    return env

def quality_worker(arm, case_id):
    sys.path.insert(0,str(RUN/'projects'/arm))
    from ai_glasses_memory_assistant.agent_bridge import GlassesChatService
    from ai_glasses_memory_assistant.acceptance_runtime import prepare
    case=next(c for c in CASES if c['id']==case_id)
    user='acceptance_'+case_id.replace('-','_')
    service=GlassesChatService()
    output=dict(id=case_id,arm=arm,condition=case)
    try:
        seeded=prepare(service,dict(user_id=user,condition=case,projection=arm=='after'))
        output['seed']=seeded
        started=time.perf_counter()
        output['result']=service.chat(case['question'],user_id=user,defer_memory_writes=False)
        output['elapsed_seconds']=time.perf_counter()-started
    except Exception as exc: output['error']=dict(type=type(exc).__name__,detail=str(exc))
    finally:
        service.close(timeout=65)
        output['audit']=[json.loads(line) for line in service.audit_path.read_text().splitlines()] if service.audit_path.exists() else []
        dump(RUN/'quality'/case_id/f'{arm}.json',output)

def quality():
    for case in CASES:
        for arm in ['before','after']:
            variant=f'quality/{case["id"]}/{arm}'
            if variant in request('/status')['used']: continue
            request('/begin',dict(variant=variant))
            home=RUN/'homes'/variant
            result=subprocess.run([sys.executable,__file__,'quality-worker','--arm',arm,'--case',case['id']],
                                  env=environment(arm,home),cwd=RUN/'projects'/arm,capture_output=True,text=True,timeout=390)
            (RUN/'quality'/case['id']/f'{arm}.log').write_text(result.stdout+result.stderr)
            status=request('/status')
            if status['pending']: raise RuntimeError('Incomplete transport records; stop')
            request('/end',dict(exit_code=result.returncode))
            print(variant, 'requests',status['counts'].get(variant,0),'exit',result.returncode,flush=True)

def web_server(arm,port):
    sys.path.insert(0,str(RUN/'projects'/arm))
    from ai_glasses_memory_assistant.server import GlassesHandler,ExclusiveThreadingHTTPServer
    from ai_glasses_memory_assistant.agent_bridge import GlassesChatService
    from ai_glasses_memory_assistant.acceptance_runtime import acceptance_handler
    handler=acceptance_handler(GlassesHandler)
    handler.service=GlassesChatService(); server=ExclusiveThreadingHTTPServer(('127.0.0.1',port),handler)
    try: server.serve_forever()
    finally: server.server_close();handler.service.close(timeout=65)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','authority','quality','quality-worker','web-server']);p.add_argument('--arm');p.add_argument('--case');p.add_argument('--port',type=int)
    args=p.parse_args()
    if args.action=='freeze': freeze()
    elif args.action=='authority': authority_server()
    elif args.action=='quality': quality()
    elif args.action=='quality-worker': quality_worker(args.arm,args.case)
    elif args.action=='web-server': web_server(args.arm,args.port)
if __name__=='__main__': main()
