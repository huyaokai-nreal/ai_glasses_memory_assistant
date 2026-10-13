"""Drive actual isolated Chrome/Android WebView, never the original app."""
import importlib.util,json,os,subprocess,sys,time,urllib.request
from pathlib import Path
import run_live_memory_closeout as live

spec=importlib.util.spec_from_file_location('device_tools',live.ROOT/'android/tools/run_device_acceptance.py')
device=importlib.util.module_from_spec(spec);spec.loader.exec_module(device)

class CDP(device.DevToolsSocket):
    def command(self,method,params=None):
        key=self.next_id;self.next_id+=1
        self._send_json(dict(id=key,method=method,params=params or {}))
        while True:
            payload=self._receive_json()
            if payload.get('id')==key:
                if payload.get('error'): raise RuntimeError(str(payload['error']))
                return payload.get('result')

    def evaluate(self,expression):
        # Avoid a long await in CDP: send submits asynchronously, then poll the page.
        return super().evaluate(expression)

def adb(*args,check=True):
    result = subprocess.run([live.ADB,'-s',live.SERIAL,*args],capture_output=True,text=True)
    if check and result.returncode:
        # Never include launch arguments: they contain the acceptance credential.
        raise RuntimeError('ADB command failed: ' + (result.stderr or result.stdout).strip())
    return result

def android_connect(arm,cold=False):
    package='com.aiglasses.memoryassistant.demo.acceptance.'+arm
    if cold: adb('shell','am','force-stop',package)
    # A surviving process can have no live WebView after Activity destruction.
    # Always foreground the isolated Activity before discovering its page.
    adb('shell','am','start','--user','0','-f','0x00020000','-n',package+'/com.aiglasses.memoryassistant.demo.MainActivity',
        '--ez','acceptance_timing','true','--es','acceptance_token',live.token())
    deadline=time.monotonic()+40
    forwarded=None
    while time.monotonic()<deadline:
        rows=adb('shell','ps','-A').stdout.splitlines()
        row=next((r for r in rows if r.split() and r.split()[-1]==package),None)
        if row:
            pid=row.split()[1]
            if not forwarded: forwarded=adb('forward','tcp:0','localabstract:webview_devtools_remote_'+pid).stdout.strip()
            try:
                targets=live.request('/json',base='http://127.0.0.1:'+forwarded,authenticated=False)
                page=next((t for t in targets if t.get('type')=='page' and t.get('url','').startswith('http://127.0.0.1:')),None)
                if page:
                    url='ws://127.0.0.1:'+forwarded+'/'+page['webSocketDebuggerUrl'].split('/',3)[3]
                    connection=CDP(url)
                    if connection.evaluate('typeof aiGlassesAcceptance === "object" && document.readyState === "complete"'):
                        return connection,forwarded
                    connection.close()
            except Exception: pass
        time.sleep(.4)
    if forwarded: adb('forward','--remove','tcp:'+forwarded,check=False)
    raise RuntimeError('Isolated Android WebView not ready')

def chrome_connect(arm,port):
    profile=live.RUN/'chrome-profile'/arm
    profile.mkdir(parents=True,exist_ok=True)
    cmd=['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
         '--user-data-dir='+str(profile),'--remote-debugging-port='+str(port),
         '--no-first-run','--no-default-browser-check','--autoplay-policy=no-user-gesture-required',
         'http://127.0.0.1:'+str(28770 if arm=='before' else 28771)]
    log=(live.RUN/f'chrome-{arm}.log').open('a')
    process=subprocess.Popen(cmd,stdout=log,stderr=log)
    deadline=time.monotonic()+40
    while time.monotonic()<deadline:
        try:
            targets=live.request('/json',base='http://127.0.0.1:'+str(port),authenticated=False)
            page=next(t for t in targets if t.get('type')=='page' and t.get('url','').startswith('http://127.0.0.1:2877'))
            connection=CDP(page['webSocketDebuggerUrl'])
            if connection.evaluate('typeof aiGlassesAcceptance === "object" && document.readyState === "complete"'):
                return connection,process
            connection.close()
        except Exception: pass
        time.sleep(.4)
    process.terminate();raise RuntimeError('Isolated Chrome page not ready')

def page_get(c,path):
    return c.evaluate('(async()=>{let r=await fetch('+json.dumps(path)+');return {status:r.status,body:await r.json()}})()')

def prepare(c,variant,condition,arm):
    user='acceptance_'+variant.replace('/','_')
    body=dict(user_id=user,condition=condition,projection=arm=='after')
    expression='''(async()=>{
      let r=await fetch('/api/acceptance/prepare',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(BODY)});
      let seed=await r.json();if(!r.ok)throw Error(JSON.stringify(seed));
      await selectUser(seed.user_id);state.sessionId=null;state.voiceEnabled=true;
      window.aiGlassesAcceptance.enable(true,PRESERVE);
      window.__acceptanceSeed=seed;window.__acceptanceResult=null;window.__acceptanceError=null;
      window.__acceptanceFetch=window.fetch;
      window.fetch=async function(url,opts){const r=await window.__acceptanceFetch.apply(this,arguments);
        if(String(url)==='/api/chat'){const copy=r.clone();copy.json().then(v=>window.__acceptanceResult=v).catch(()=>{});}return r;};
      return seed;
    })()'''.replace('BODY',json.dumps(body,ensure_ascii=False)).replace('PRESERVE','true' if '/cold/' in variant else 'false')
    return c.evaluate(expression)

def submit(c,question):
    # Actual existing form event/input controls, with request payload observed.
    c.evaluate('''(()=>{let i=document.querySelector('#message-input');i.value=QUESTION;
      document.querySelector('#send-button').click();return true;})()'''.replace('QUESTION',json.dumps(question,ensure_ascii=False)))

def wait_result(c,deadline=300):
    end=time.monotonic()+deadline
    result=None
    while time.monotonic()<end:
        payload=c.evaluate('({result:window.__acceptanceResult,events:aiGlassesAcceptance.snapshot(),disabled:document.querySelector("#message-input").disabled,toast:document.querySelector("#toast")?.textContent})')
        result=payload.get('result')
        if result and not payload['disabled'] and any(e.get('phase')=='reply_visible' for e in payload['events']):
            return payload
        if not payload['disabled'] and not result and any(e.get('phase')=='controls_ready' for e in payload['events']):
            return dict(payload,error='chat failed; no result')
        time.sleep(.2)
    return dict(result=result,error='actual UI timed out',events=c.evaluate('aiGlassesAcceptance.snapshot()'))

def collect(c,payload,condition):
    deadline=time.monotonic()+90
    status={}
    while time.monotonic()<deadline:
        status=page_get(c,'/api/acceptance/status')
        if not status['body'].get('pending'): break
        time.sleep(.3)
    payload['background']=status
    if status['body'].get('pending'): payload['error']='background not drained'
    all_audit=status['body'].get('audit',[])
    user=c.evaluate('state.userId')
    audit=[r for r in all_audit if r.get('user_id')==user]
    payload['audit']=audit
    result=payload.get('result') or {}
    memories=result.get('saved_memories',[])
    for row in audit:
        memories.extend(row.get('saved_memories',[]))
    unique={m['id']:m for m in memories if isinstance(m,dict) and m.get('id')}
    # Job public state exposes saved IDs on some older variants.
    job=result.get('debug',{}).get('memory_processing',{}).get('job_id')
    if job:
        public=page_get(c,'/api/memory/jobs?job_id='+job+'&user_id='+user)
        payload['job']=public
        for m in public.get('body',{}).get('job',{}).get('saved_memories',[]):
            if isinstance(m,dict) and m.get('id'): unique[m['id']]=m
    saved_ids=set()
    for row in audit: saved_ids.update(row.get('saved_memory_ids',[]))
    if job: saved_ids.update(public.get('body',{}).get('job',{}).get('saved_memory_ids',[]))
    if saved_ids:
        listing=page_get(c,'/api/memories?user_id='+user+'&limit=100')
        for m in listing.get('body',{}).get('memories',[]):
            if m.get('id') in saved_ids: unique[m['id']]=m
    if unique:
        m=next(iter(unique.values()))
        ids={'turn_id':result.get('debug',{}).get('timeline',{}).get('turn_id'),'job_id':job}
        passed=c.evaluate('(async()=>aiGlassesAcceptance.probeSearch('+json.dumps(m.get('content',''))+','+json.dumps([m['id']])+','+json.dumps(ids)+'))()')
        payload['search_probe']=dict(passed=passed,expected_ids=[m['id']])
    deadline=time.monotonic()+25
    while time.monotonic()<deadline:
        events=c.evaluate('aiGlassesAcceptance.snapshot()')
        if any(e.get('phase')=='tts_start' for e in events): break
        time.sleep(.25)
    payload['events']=events
    payload['controls_disabled']=c.evaluate('document.querySelector("#message-input").disabled')
    c.evaluate('window.fetch=window.__acceptanceFetch;true')
    return payload

def faults(c,platform,arm):
    if (live.RUN/'experience'/platform/f'faults-{arm}.json').exists():
        return
    output=[]
    for mode in ['failure','timeout','disconnect']:
        c.evaluate('''(()=>{aiGlassesAcceptance.enable(true);window.__originalFaultFetch=fetch;
          window.fetch=async function(u,o){if(String(u)==='/api/chat'){
            if(MODE==='failure')return new Response(JSON.stringify({detail:'验收故障注入'}),{status:503,headers:{'Content-Type':'application/json'}});
            throw MODE==='timeout'?new DOMException('验收超时注入','TimeoutError'):new TypeError('验收连接中断注入');
          }return window.__originalFaultFetch.apply(this,arguments);};return true;})()'''.replace('MODE',json.dumps(mode)))
        submit(c,'故障测试，不调用模型。')
        time.sleep(.5)
        output.append(dict(mode=mode,events=c.evaluate('aiGlassesAcceptance.snapshot()'),
            controls_disabled=c.evaluate('document.querySelector("#message-input").disabled'),
            boundary='actual UI with fetch-layer fault injection; not physical network outage'))
        c.evaluate('window.fetch=window.__originalFaultFetch;true')
    live.dump(live.RUN/'experience'/platform/f'faults-{arm}.json',output)

def main(platform):
    handles={};processes=[]
    arms=os.environ.get('AI_GLASSES_ACCEPTANCE_ARMS','before,after').split(',')
    if not arms or any(arm not in ['before','after'] for arm in arms):
        raise ValueError('Only frozen before/after arms are allowed')
    if platform=='android':
        adb('reverse','tcp:27901','tcp:27901')
    else:
        for arm,port in [(a,28770 if a=='before' else 28771) for a in arms]:
            log=(live.RUN/f'web-server-{arm}.log').open('a')
            process=subprocess.Popen([sys.executable,str(Path(live.__file__).resolve()),'web-server','--arm',arm,'--port',str(port)],
                env=live.environment(arm,live.RUN/'homes'/'web'/arm),stdout=log,stderr=log,cwd=live.RUN/'projects'/arm)
            processes.append(process)
        time.sleep(2)
    try:
        for arm in arms:
            if platform=='android': c,handle=android_connect(arm)
            else: c,handle=chrome_connect(arm,28870 if arm=='before' else 28871)
            handles[arm]=(c,handle)
            probe=page_get(c,'/api/acceptance/models')
            live.dump(live.RUN/'preflight'/f'{platform}-{arm}-models.json',probe)
            if probe['status']!=200 or 'qwen3.8-27b-32k' not in json.dumps(probe): raise RuntimeError('Model unreachable from '+platform)
        scenarios=[(condition,rep) for condition in live.EXPERIENCE for rep in range(3)]+[(live.EXPERIENCE[0],'cold')]
        for condition,repeat in scenarios:
            for arm in arms:
                variant=f'{platform}/{condition["id"]}/{repeat}/{arm}'
                if variant in live.request('/status')['used']: continue
                c,handle=handles[arm]
                if platform=='android':
                    package='com.aiglasses.memoryassistant.demo.acceptance.'+arm
                    adb('shell','am','start','--user','0','-f','0x00020000','-n',
                        package+'/com.aiglasses.memoryassistant.demo.MainActivity',
                        '--es','acceptance_token',live.token())
                else: c.command('Page.bringToFront')
                if repeat=='cold':
                    c.close()
                    if platform=='android':
                        adb('forward','--remove','tcp:'+handle,check=False); c,handle=android_connect(arm,True)
                    else:
                        handle.terminate();handle.wait(timeout=15);c,handle=chrome_connect(arm,28870 if arm=='before' else 28871)
                    handles[arm]=(c,handle)
                live.request('/begin',dict(variant=variant))
                try:
                    seeded=prepare(c,variant,condition,arm)
                    submit(c,condition['question'])
                    payload=collect(c,wait_result(c),condition)
                    payload.update(variant=variant,condition=condition,seed=seeded,input_route='actual page text form')
                    payload['navigation']=c.evaluate('({time_origin:performance.timeOrigin,entries:performance.getEntriesByType("navigation").map(x=>x.toJSON())})')
                    live.dump(live.RUN/'experience'/variant.removeprefix(platform+'/')/f'{platform}.json',payload)
                    if payload.get('background',{}).get('body',{}).get('pending'): raise RuntimeError('Background still active')
                    live.request('/end',dict(outcome='failed' if payload.get('error') else 'recorded'))
                    print(variant,'failed' if payload.get('error') else 'recorded',flush=True)
                    # Stop this failed UI version; untouched other platform can continue.
                    if payload.get('error'): raise RuntimeError('Formal UI failure; version stopped, denominator retained')
                except Exception as exc:
                    live.dump(live.RUN/'experience'/'errors'/f'{platform}-{condition["id"]}-{repeat}-{arm}.json',dict(variant=variant,error=str(exc)))
                    if not live.request('/status')['pending'] and live.request('/status')['current']:
                        live.request('/end',dict(outcome='failed',error=str(exc)))
                    raise
        for arm,(c,_) in handles.items(): faults(c,platform,arm)
    finally:
        for c,handle in handles.values():
            c.close()
            if platform=='android': adb('forward','--remove','tcp:'+handle,check=False)
            else: handle.terminate()
        for p in processes: p.terminate()
if __name__=='__main__': main(sys.argv[1])
