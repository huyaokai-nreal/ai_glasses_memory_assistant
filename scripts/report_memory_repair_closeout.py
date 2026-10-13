"""Combine sealed quality and Web evidence, without any model or UI calls."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import report_memory_quality_revision as quality
import run_memory_quality_revision as revision
import run_live_memory_closeout as live


def load(path):
    return json.loads(path.read_text())


def percentiles(values):
    values = sorted(values)
    return {f"p{p}": values[math.ceil(len(values) * p / 100) - 1] if values else None
            for p in (50, 95)}


def generate(run):
    protocol = revision.configure(run)
    revision.verify(run)
    ledger = revision.ledger(run, protocol)
    assert not ledger.pending and ledger.current is None
    reviews = load(run / "answer-review.json")
    rows = quality.chains(run, protocol, ledger)
    assert len(reviews) == 24 and len({r['id'] for r in reviews}) == 24
    import hashlib
    for review in reviews:
        row = next(r for r in rows if r['id'] == review['id'])
        assert hashlib.sha256(row['reply'].encode()).hexdigest() == review['answer_sha256']
    after = quality.summary(rows, reviews)
    parent = live.ROOT / protocol['parent_run']
    before = quality.summary([r for r in load(parent / 'quality-chain.json') if r['arm'] == 'after'],
                             [r for r in load(parent / 'answer-review.json') if r['arm'] == 'after'])
    web = run / 'web'
    wp = load(web / 'protocol.json')
    for name, expected in wp['parent_hashes'].items():
        assert live.digest(live.ROOT / name) == expected
    for name, expected in load(web / 'freeze.json').items():
        assert live.digest(web / name) == expected
    assert wp['source_manifest_sha256'] == live.digest(run / 'sources/after.json')
    # Check that the merged historical ledger preserves every row and attempt.
    archived = []
    for index, name in enumerate(wp['parent_hashes']):
        for line in (live.ROOT / name).read_text().splitlines():
            row = json.loads(line)
            for key in ('variant', 'request_id'):
                if key in row: row[key] = f'archive{index}/' + row[key]
            archived.append(row)
    assert archived == [json.loads(line) for line in (web / 'parent-requests.jsonl').read_text().splitlines()]
    wl = live.Ledger(web, wp['variants'], prior_ledger=(web / 'parent-requests.jsonl', wp['parent_copy_sha256']), local_limit=35)
    status = wl.dispatch('/status', {})
    assert status['current'] is None and not status['pending']
    assert wl.prior_total == 328 and wl.prior_variants == 102 and len(wl.used) == 7
    assert status['total'] <= 640 and wl.prior_variants + len(wl.used) <= 124
    assert all(count <= 5 for count in wl.count.values())
    captures = load(web / 'browser-events.json')
    assert [r['id'] for r in captures] == [c['id'] for c in wp['conditions'][:6]]
    reply_ms, tts_ms, controls_ms, search_upper_bounds = [], [], [], []
    for row in captures:
        events = {e['phase']: e for e in row['events']}
        assert not row['disabled'] and row['reply']
        assert not load(web / f"{row['id']}-background.json")['pending']
        for phase in ('input_end', 'request_start', 'reply_visible', 'controls_ready', 'tts_start'):
            assert phase in events
        iid = events['request_start']['interaction_id']
        assert all(events[p]['interaction_id'] == iid for p in ('reply_visible', 'controls_ready', 'tts_start'))
        # The visible browser reply must match the actual isolated server audit.
        audit = load(web / f"{row['id']}-background.json")['audit']
        source = next(a for a in audit if a.get('timeline_turn_id') == events['reply_visible']['turn_id'])
        assert source['reply'] == row['reply']
        begin = events['input_end']['at_ms']  # text submit boundary, not physical audio
        reply_ms.append(events['reply_visible']['at_ms'] - begin)
        tts_ms.append(events['tts_start']['at_ms'] - begin)
        controls_ms.append(max(0, events['controls_ready']['at_ms'] - events['reply_visible']['at_ms']))
        probe = row.get('searchProbe')
        if probe:
            saved = load(web / f"{row['id']}-memories.json")['memories']
            assert probe['passed'] and set(probe['expected_ids']) <= {m['id'] for m in saved}
            search_upper_bounds.append(events['memory_searchable']['at_ms'] - begin)
    cold = load(web / 'generic_cold-failure.json')
    assert cold['generation_requests'] == 0 and not cold['retried']
    metrics = dict(quality_before=before, quality_after=after, requests=status,
        attempted_variants=wl.prior_variants + len(wl.used), web=dict(planned=7, ui_executed=6,
            ui_response_success=6, tts_observed=6, semantic_write_success=1, semantic_write_denominator=2,
            cold_start_failure=cold, visible_reply_ms=percentiles(reply_ms),
            actual_tts_start_ms=percentiles(tts_ms), controls_after_visible_ms=percentiles(controls_ms),
            search_probe_upper_bounds_ms=search_upper_bounds, first_searchable_latency_measured=False,
            physical_input_end_measured=False, paired_comparison=False, full_experience_passed=False),
        android_resumed=False, real_database_installed=False, speed_gain_attributable=False, overall_complete=False)
    output = run / 'reports' / ('closeout-' + time.strftime('%Y%m%d-%H%M%S'))
    output.mkdir(parents=True, exist_ok=False)
    live.dump(output / 'metrics.json', metrics)
    paths = [run / 'protocol.json', run / 'conditions.json', run / 'freeze.json', run / 'requests.jsonl',
             run / 'answer-review.json', run / 'sources/after.json', run / 'runner-hashes.json',
             live.ROOT / '.planning/2026-10-09-a-group/checklist.md', Path(__file__).resolve()]
    paths += [run / r['evidence'] for r in rows]
    paths += [p for p in web.glob('*.json*') if p.is_file()] + [web / 'run_web.py', web / 'server-cold.log', web / 'page-proof.jpg']
    live.dump(output / 'evidence-manifest.json', {str(p.relative_to(live.ROOT)):live.digest(p) for p in paths})
    text = f'''# 本轮召回、时间与Web检查

## 用户问题与修改

修复昨天等日历范围消费、服务时区报时、清单与总数分离、事实与原话仲裁；同次PPD改述保持3×128预算。没有新向量模型、判断调用、存储字段或真实库安装。

## 同口径前后指标

检查点ce08c81结果封存，只运行新版本24条件一次，已逐条人工复核；历史留出已被查看，不是盲测。

| 指标 | 检查点 | 本轮 |
|---|---|---|
| 候选正例/12 | {before['candidate']['positive']} | {after['candidate']['positive']} |
| 零词重合/8 | {before['candidate']['zero']} | {after['candidate']['zero']} |
| 历史留出/6 | {before['candidate']['holdout']} | {after['candidate']['holdout']} |
| 最终正确正例/12 | {before['positive_answers']} | {after['positive_answers']} |
| 时间严格条件/8 | {before['time_pass']} | {after['time_pass']} |
| 原正确保持/2 | {before['original_correct']} | {after['original_correct']} |
| 无证据负例安全/2 | {before['negative_safe']} | {after['negative_safe']} |

质量新增67请求，Web新增{status['local_total']}请求；累计{status['total']}/640、{metrics['attempted_variants']}/124变体、0待结束。旧1次未知结果保守计入。所有生成走指定本地Qwen，最多5请求/变体。

## 失败与退化

质量门槛未通过。召回剩余5个候选遗漏；time-1漏掉UTC与服务当地日期区分，是原正确退化；time-6的complete-set证据集合为空，未列出未知日期相关记录，未进入Reader，不能归因为Reader忽略证据。

Web六类热运行各一次，6/7取得可见回复并恢复控件、6/7取得实际TTS开始。两条写入中1条语义正确；自然陈述的落地灯记录只保存next_week，原正文丢失。PPD候选仍完整，现有writer在temporal_text为空时通过空串包含检查，将时间范围标签normalized_text替换正文；ce08c81已有相同逻辑。本轮保留该冻结失败，不临时改提示或补跑。

冷启动服务绑定报Address already in use，准备请求被拒绝连接；0模型请求，不重试、不换端口替代样本。它计入7条完整分母，不能称正常路径零失败。

## Web计时与证据边界

六条热运行的文字提交到可见回复p50/p95为{metrics['web']['visible_reply_ms']['p50']/1000:.3f}/{metrics['web']['visible_reply_ms']['p95']/1000:.3f}秒；到实际TTS起声为{metrics['web']['actual_tts_start_ms']['p50']/1000:.3f}/{metrics['web']['actual_tts_start_ms']['p95']/1000:.3f}秒。六条控件均在回复可见前恢复，回复显示后的等待为0ms。

这不是原38变体的前后配对验收：仅本轮新版本7变体，不能判断10%退化或可归因提速。两个实际搜索探针能返回保存ID，但人工较晚发起，时间只是上界，不报告为首次可检索p95。没有麦克风物理输入结束、录音或Android证据。sync_write是冻结条件名称，实际文字链路仍由后台job保存。

Web访问曾被自动审批拒绝；用户明确授权两个独立localhost地址后工具允许访问，全程使用同一浏览器工具，没有绕过访问限制。只对临时合成用户启用计时和响应观察，结束前恢复并关闭页面及自建服务。正式应用与真实库未动。

## 下一步与所需支持

保留召回、时间回答和整体状态未完成。下一稳定节点应先保护时间范围标签不覆盖记忆正文，再处理UTC回答与未知日期证据集合；召回剩余是有界词法路线覆盖不足，不能靠题目专用词表补分。当前失败版本不重跑；新实验需另冻协议与请求预算。

完整Python908通过/2跳过，相关时间70及召回138通过，编译和空白检查通过；没有当前HEAD LongMemEval数字。标签/CJK/并发旧结果保留，底层实现本轮未改；真实库安装、Android体验与速度目标仍待验。#2/#3/#5/#8本轮不处理。未自动stage、commit或push。
'''
    (output / 'report.md').write_text(text)
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    generate(parser.parse_args().run.resolve())
