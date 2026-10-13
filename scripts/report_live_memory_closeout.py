"""Read-only verified live acceptance report. No model calls or overwritten runs."""
from __future__ import annotations
import argparse,hashlib,json,math,time
from collections import Counter
from pathlib import Path

def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def percentile(values,p):
    return sorted(values)[max(0,math.ceil(len(values)*p)-1)] if values else None

def generate(root):
    freeze=read(root/'freeze.json');ledger=[json.loads(x) for x in (root/'requests.jsonl').read_text().splitlines()]
    reservations=[r for r in ledger if r['type']=='reserve'];counts=Counter(r['variant'] for r in reservations)
    finishes={r['request_id']:r for r in ledger if r['type']=='finish'}
    assert len(counts)<=124 and len(reservations)<=640 and all(n<=5 for n in counts.values())
    assert len({r['request_id'] for r in reservations})==len(reservations)
    for r in reservations:
        assert r['variant'] in freeze['variants'] and r['client']=={k:freeze['config'][k] for k in ['provider','model','base_url']}
        assert r['request']['model']==freeze['config']['model']
    verified={}
    for arm in ['before','after']:
        hashes=read(root/'sources'/f'{arm}.json'); project=root/'projects'/arm
        for name,expected in hashes.items(): assert sha(project/name)==expected, name
        verified[arm]=len(hashes)
    builds=read(root/'build-install.json')
    for build in builds: assert sha(root/'apks'/f'{build["arm"]}.apk')==build['apk_sha256']
    quality=read(root/'quality-chain.json') if (root/'quality-chain.json').exists() else []
    reviews=read(root/'answer-review.json') if (root/'answer-review.json').exists() else []
    expected={(c['id'],arm) for c in freeze['conditions'] for arm in ['before','after']}
    assert len({(r['id'],r['arm']) for r in quality})==len(quality)
    assert len({(r['id'],r['arm']) for r in reviews})==len(reviews)
    assert {(r['id'],r['arm']) for r in quality} <= expected
    assert {(r['id'],r['arm']) for r in reviews} <= expected
    for review in reviews:
        evidence=(root/review['evidence']).resolve()
        assert evidence.is_relative_to(root.resolve())
        raw=read(evidence)
        assert (raw['id'],raw['arm'])==(review['id'],review['arm'])
        answer=raw.get('result',{}).get('reply','')
        assert hashlib.sha256(answer.encode()).hexdigest()==review['answer_sha256']
    for row in quality:
        raw=read(root/'quality'/row['id']/f'{row["arm"]}.json')
        assert row['reply']==raw.get('result',{}).get('reply','')
    reviewed={(r['id'],r['arm']):r for r in reviews}
    metrics={'generation_requests':len(reservations),'request_limit':640,'per_variant_max':max(counts.values(),default=0),
             'requests_pending':len(set(r['request_id'] for r in reservations)-set(finishes)),
             'requests_outcome_unknown':sum(r.get('error',{}).get('type')=='OutcomeUnknown' for r in finishes.values() if r.get('error')),
             'transport_errors':[dict(variant=r['variant'],error=r['error']) for r in finishes.values() if r.get('error')],
             'source_files_verified':verified,'builds':builds,'overall_complete':False,'real_database_installed':False}
    if (root/'ui-session-stop-20261013.json').exists():
        metrics['ui_observation_limits']=read(root/'ui-session-stop-20261013.json')
        metrics['speed_gain_attributable']=False
    metrics['quality']={}
    for arm in ['before','after']:
        rows=[r for r in quality if r['arm']==arm];pos=[r for r in rows if r['condition'] in ['zero_overlap','paraphrase']]
        zeros=[r for r in pos if r['condition']=='zero_overlap'];holdout=[r for r in pos if r['group']=='holdout']
        def hits(values,key):return sum(bool(r.get(key)) for r in values)
        time_rows=[r for r in rows if r['stage']=='time'];time_pass=sum(reviewed.get((r['id'],arm),{}).get('passed') is True for r in time_rows)
        failures=Counter(r.get('failure_layer','not_executed') for r in pos if not r.get('final_recall_at5'))
        for r in pos:
            if r.get('final_recall_at5') and reviewed.get((r['id'],arm),{}).get('passed') is False:failures['answer_use_failed']+=1
        original=[r for r in rows if r['condition']=='original_correct'];neg=[r for r in rows if r['condition']=='no_evidence']
        answer_positive=sum(reviewed.get((r['id'],arm),{}).get('passed') is True for r in pos)
        passed=(hits(pos,'candidate_recall_at5')>=11 and hits(zeros,'candidate_recall_at5')>=7 and hits(holdout,'candidate_recall_at5')>=5
            and all(reviewed.get((r['id'],arm),{}).get('passed') is True for r in original+neg)
            and time_pass==8 and sum(r['executed'] for r in rows)==24)
        metrics['quality'][arm]=dict(executed=sum(r['executed'] for r in rows),denominator=24,
            candidate_recall_at5={'positive':hits(pos,'candidate_recall_at5'),'denominator':12,'zero':hits(zeros,'candidate_recall_at5'),'zero_denominator':8,'holdout':hits(holdout,'candidate_recall_at5'),'holdout_denominator':6},
            final_recall_at5={'positive':hits(pos,'final_recall_at5'),'denominator':12,'zero':hits(zeros,'final_recall_at5'),'zero_denominator':8,'holdout':hits(holdout,'final_recall_at5'),'holdout_denominator':6},
            positive_answers_correct=answer_positive,time_answers_passed=time_pass,time_denominator=8,
            original_correct=sum(reviewed.get((r['id'],arm),{}).get('passed') is True for r in original),
            negative_safe=sum(reviewed.get((r['id'],arm),{}).get('passed') is True for r in neg),
            failures_by_layer=dict(failures),passed=passed,
            gate_passed_without_answer_gate=hits(pos,'candidate_recall_at5')>=11 and hits(zeros,'candidate_recall_at5')>=7 and hits(holdout,'candidate_recall_at5')>=5)
    metrics['experience']={}
    for platform in ['android','web']:
        output={}; all_rows=[]
        for p in (root/'experience').glob(f'*/*/*/{platform}.json'):
            row=read(p)
            if row.get('variant','').startswith(platform+'/'):all_rows.append(row)
        recorded_variants={row['variant'] for row in all_rows}
        for p in (root/'experience'/'errors').glob(f'{platform}-*.json'):
            row=read(p);variant=row.get('variant','')
            if variant in recorded_variants:continue
            assert variant in freeze['variants'] and variant.startswith(platform+'/')
            condition=next(c for c in freeze['experience'] if c['id']==variant.split('/')[1])
            all_rows.append(dict(variant=variant,condition=condition,error=row['error'],events=[],result_recorded=False))
            recorded_variants.add(variant)
        for arm in ['before','after']:
            rows=[r for r in all_rows if r['variant'].endswith('/'+arm)]
            output[arm]={'executed':len(rows),'denominator':19,'results_recorded':sum(r.get('result_recorded',True) for r in rows),
                         'failures':sum(bool(r.get('error')) for r in rows),'missing':Counter(),'hot':{},'cold':[]}
            buckets={key:[] for key in ['reply_visible_ms','tts_start_ms','memory_searchable_ms','controls_after_reply_ms']}
            for row in rows:
                events=row.get('events',[])
                def at(phase):
                    matching=[e for e in events if e.get('phase')==phase]
                    return matching[0].get('mapped_web_ms') if matching and matching[0].get('clock')=='android_elapsed' else matching[0].get('at_ms') if matching else None
                start=at('input_end');reply=at('reply_visible');tts=at('tts_start');controls=at('controls_ready');search=at('memory_searchable')
                vals={'reply_visible_ms':reply-start if reply is not None and start is not None else None,
                      'tts_start_ms':tts-start if tts is not None and start is not None else None,
                      'memory_searchable_ms':search-start if search is not None and start is not None else None,
                      'controls_after_reply_ms':max(0,controls-reply) if controls is not None and reply is not None else None}
                row['derived_metrics']=vals
                if '/cold/' in row['variant']:output[arm]['cold'].append(dict(variant=row['variant'],metrics=vals,navigation=row.get('navigation'),activity_start=at('activity_start')))
                else:
                    for key,value in vals.items():
                        if value is not None:buckets[key].append(value)
                        elif key!='memory_searchable_ms' or row['condition']['id'] in ['sync_write','background_write']:output[arm]['missing'][key]+=1
                if row['condition']['id']=='sync_write':
                    if not row.get('result',{}).get('saved_memories'):output[arm]['missing']['sync_write_path_not_observed']+=1
                if row['condition']['id']=='background_write':
                    if not row.get('result',{}).get('debug',{}).get('memory_processing',{}).get('job_id'):output[arm]['missing']['background_write_path_not_observed']+=1
            output[arm]['missing']=dict(output[arm]['missing'])
            output[arm]['hot']={key:dict(n=len(values),p50=percentile(values,.5),p95=percentile(values,.95)) for key,values in buckets.items()}
        output['comparisons']={}
        for key in ['reply_visible_ms','tts_start_ms','memory_searchable_ms']:
            before=output['before']['hot'][key]['p95'];after=output['after']['hot'][key]['p95']
            output['comparisons'][key]=dict(before_p95=before,after_p95=after,ratio=after/before if before and after is not None else None,passed=bool(before and after is not None and after<=before*1.1))
        output['faults']={arm:read(root/'experience'/platform/f'faults-{arm}.json') if (root/'experience'/platform/f'faults-{arm}.json').exists() else None for arm in ['before','after']}
        output['passed']=all(output[arm]['executed']==19 and output[arm]['failures']==0 and not output[arm]['missing'] and (output[arm]['hot']['controls_after_reply_ms']['p95'] or 0)<=100 for arm in ['before','after']) and all(c['passed'] for c in output['comparisons'].values())
        output['boundary']='actual text UI and actual TTS events; no physical microphone/audio-end, Bluetooth or radio-outage claim'
        metrics['experience'][platform]=output
    directory=root/'reports'/time.strftime('%Y%m%d-%H%M%S')
    directory.mkdir(parents=True,exist_ok=False)
    (directory/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
    lines=['# 联机模型与双端验收报告','',f'实际模型请求 {len(reservations)}/640；每变体最多 {max(counts.values(),default=0)}/5。','',
           '整体仍未完成：真实库未安装；召回、时间与两端体验分别按结果判定。','',
           '## 用户问题 → 修改 → 同口径前后结果','',
           '实现了请求前预算、无隐式重试的本地模型入口，以及与原应用并存的独立验收包。']
    for arm,q in metrics['quality'].items():
        lines.extend(['',f'- {arm}：执行 {q["executed"]}/24；候选目标召回 {q["candidate_recall_at5"]["positive"]}/12、最终召回 {q["final_recall_at5"]["positive"]}/12；零词重合最终召回 {q["final_recall_at5"]["zero"]}/8；正例答案 {q["positive_answers_correct"]}/12；时间条件 {q["time_answers_passed"]}/8。',f'  失败归因：{q["failures_by_layer"]}。'])
    lines+=['','召回仍未达标：候选≥11/12、零词重合≥7/8、留出≥5/6的三个门槛均未通过。改述有时没有给出能找到目标的具体检索词；另有1条漏开门控、1条已找到记录却被原话证据仲裁丢弃。',
            '时间有改善但未通过全部8条：仍有“昨天”回答未完成、带日期的收件查询回答0、按记录日期排除正文事件的问题。offset ISO对照保留了正确UTC日期，但未完成冻结要求的当地时区表达，不能称为原绝对时间解析错误。',
            '逐例原始回答与事件/人物/日期/证据复核见本证据目录quality/及answer-review.json；固定正确合同和字符串命中未用于替代真实PPD准确率。']
    lines+=['','## 双端结果与失败','']
    for platform,result in metrics['experience'].items():
        lines.append(f'- {platform}：before {result["before"]["executed"]}/19、after {result["after"]["executed"]}/19；阶段通过={result["passed"]}。')
        for key,c in result['comparisons'].items():lines.append(f'  {key} p95：{c["before_p95"]} → {c["after_p95"]} ms；通过={c["passed"]}。')
        for arm in ['before','after']:
            lines.append(f'  {arm} 缺测/路径未满足：{result[arm]["missing"]}；正常失败 {result[arm]["failures"]}。')
            for key,values in result[arm]['hot'].items():
                lines.append(f'  {arm} {key}：n={values["n"]}，p50={values["p50"]}、p95={values["p95"]} ms。')
            lines.append(f'  {arm} 冷启动配对原始值：{result[arm]["cold"]}。')
    lines+=['','## 证据边界与下一步','',
            '24 条件在实现后冻结，不能称为盲测；答案由逐例事件/人物/日期/证据人工复核，不以关键词出现判正确。未执行条件继续留在完整分母中。',
            '本轮输入为实际页面文字表单，TTS 为实际播放事件。麦克风、物理声音结束、蓝牙及实际无线断网未测；故障材料属于页面 fetch 层注入。',
            '失败版本不重跑。根据失败归因另行冻结通用修复；真实库派生索引仍需审批。#2/#3/#5/#8 后续排期，本轮不改判八项总体完成。']
    if 'ui_observation_limits' in metrics:
        lines+=['', '本轮计算机操作工具因当前浏览器URL不允许访问而终止操作会话，UI验收已停止，未绕过限制。Web sync_write/1/before观察到窗口hidden，屏幕绘制标记延迟；原值与分母保留，不能将其前后时延差归因优化收益。冷启动配对未完成，Android正常体验未完成。']
    (directory/'report.md').write_text('\n'.join(lines)+'\n')
    files={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and not any(x in p.parts for x in ['homes','projects','chrome-profile','__pycache__']) and p.name!='authority-token' and 'reports' not in p.parts}
    (directory/'evidence-manifest.json').write_text(json.dumps(files,indent=2)+'\n')
    print(directory);return metrics
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]/'.planning/2026-10-12-live-closeout');generate(p.parse_args().root)
