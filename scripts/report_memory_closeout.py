"""Read-only, model-free closeout report. Never overwrite prior report folders."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(root: Path, evidence: Path) -> dict:
    manifest = json.loads((evidence / 'delivery-manifest.json').read_text())
    for item in manifest['files']:
        path = (root / item['path']).resolve()
        if not path.is_relative_to(root) or sha(path) != item['sha256']:
            raise ValueError(f"Evidence hash mismatch: {item['path']}")
    status = json.loads((evidence / 'closeout-status.json').read_text())
    checklist = (root / status['checklist']).read_text()
    if set(re.findall(r'^\| #([1-8]) ', checklist, re.M)) != set('12345678'):
        raise ValueError('Unified checklist must contain all eight items')
    tags = json.loads((evidence / 'integration/paired-quality.json').read_text())
    if len(tags) != 672:
        raise ValueError('Tag denominator changed')
    passed = lambda r: (r['versions'][1]['target_selected'] == r['expected']
                         and r['versions'][1]['target_actual_input'] == r['expected'])
    p95 = lambda xs: sorted(xs)[math.ceil(.95 * len(xs)) - 1]
    config = json.loads((evidence / 'integration/manifest.json').read_text())['config']
    perf = [json.loads(f.read_text()) for f in sorted((evidence / 'integration/benchmark').glob('*-timing.json'))]
    if len(perf) != 78 or len({(r['size'], r['without_fts'], r['workload']) for r in perf}) != 78:
        raise ValueError('Tag performance denominator changed')
    for row in perf:
        for version in ('before', 'after'):
            arms = [r for r in row['arms'] if r['version'] == version]
            if len(arms) != 2 or any(len(r['samples_ms']) != config['query_samples_per_arm'] for r in arms):
                raise ValueError('Tag sample denominator changed')
            if p95([x for arm in arms for x in arm['samples_ms']]) != row['p95_' + version]:
                raise ValueError('Tag p95 mismatch')
        ceiling = row['p95_before'] * config['correct_gate_ratio'] + config['correct_gate_extra_ms'] if row['classification'] == 'correct' else config['restored_gate_ms']
        if ceiling != row['ceiling_ms'] or row['passed'] != (row['p95_after'] <= ceiling):
            raise ValueError('Tag gate mismatch')
        classification = json.loads((evidence / 'integration/benchmark' / f"n{row['size']}-{int(row['without_fts'])}-{row['workload']}-classification.json").read_text())
        for version in ('before', 'after', 'oracle'):
            ids = classification[version + '_ids']
            digest = hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest()
            if digest != classification['hashes'][version] or len(ids) != classification['counts'][version]:
                raise ValueError('Tag qualification hash/count mismatch')
        if classification['after_ids'] != classification['oracle_ids']:
            raise ValueError('Tag qualification does not match oracle')
    tag_metrics = dict(regression_denominator=384, regression_passed=sum(passed(r) for r in tags[:384]),
                       all_denominator=672, all_passed=sum(passed(r) for r in tags),
                       performance_denominator=78, performance_passed=sum(r['passed'] for r in perf),
                       worst_after_p95_ms=max(r['p95_after'] for r in perf), installed=False,
                       preserved_failures={'V3': '77/78', 'V4': '465/480 quality', 'V5': '77/78 performance'})
    cjk = json.loads((evidence / 'integration/results.json').read_text())
    if len(cjk) != 192:
        raise ValueError('CJK denominator changed')
    cells = [json.loads(f.read_text()) for f in sorted((evidence / 'integration/scale').glob('*-*-*.json'))]
    if len(cells) != 18 or len({(r['size'], r['concurrency'], r['fts']) for r in cells}) != 18 or sum(r['read_count'] for r in cells) != 3600 or sum(r['write_count'] for r in cells) != 900:
        raise ValueError('Scale denominator changed')
    for row in cells:
        for mode, count in (('read', 200), ('write', 50)):
            if len(row['samples_ms'][mode]) != count or p95(row['samples_ms'][mode]) != row[f'{mode}_p95_ms']:
                raise ValueError('Scale p95/count mismatch')
        valid = (not row['errors'] and row['stored'] == row['size'] + 50 and row['integrity'] == 'ok'
                 and row['write_isolation'] and row['index_consistent'] and row['projection_consistent']
                 and row['read_p95_ms'] <= 100 and row['write_p95_ms'] <= 200)
        if row['passed'] != valid:
            raise ValueError('Scale gate mismatch')
    freeze = json.loads((evidence / 'integration/formal-source-freeze.json').read_text())
    for filename, digest in freeze.items():
        if sha(root / filename) != digest:
            raise ValueError('Integrated formal source changed')
    if status['models']['new_real_requests'] != 0:
        raise ValueError('This delivery has no new model authorization')
    ledger = root / status['models']['old_ledger']
    if len(ledger.read_text().splitlines()) != 230 or sha(ledger) != status['models']['old_ledger_sha256']:
        raise ValueError('Sealed old model ledger changed')
    return dict(status=status, hashes_verified=len(manifest['files']),
                tag=tag_metrics,
                cjk=dict(denominator=192, passed=sum(r['passed'] for r in cjk),
                         before_correct=sum(r['versions'][0]['target_selected'] == r['expected'] for r in cjk[:96]),
                         after_correct=sum(r['versions'][1]['target_selected'] == r['expected'] for r in cjk[:96]),
                         installed_in_source=True, real_projection_installed=False),
                scale=dict(formal_denominator=18, passed=sum(r['passed'] for r in cells),
                           reads=3600, writes=900,
                           worst_read_p95_ms=max(r['read_p95_ms'] for r in cells),
                           worst_write_p95_ms=max(r['write_p95_ms'] for r in cells),
                           formal_complete=True),
                real_quality=dict(denominator=24, executed=0, blinded=False),
                device=dict(web_actual_samples=0, android_actual_samples=0))


def write_report(root: Path, evidence: Path, out: Path) -> Path:
    metrics = summarize(root.resolve(), evidence.resolve())
    out.mkdir(parents=True, exist_ok=False)
    (out / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n')
    s = metrics['status']
    text = ['# AI眼镜记忆系统收尾报告', '', '当前四项和八项总体均未完成。真实模型调用0，未迁移真实库、未安装设备、未提交Git。', '']
    for stage in s['stages']:
        text += [f"## {stage['name']}", '', f"用户问题：{stage['problem']}", '', f"修改：{stage['change']}", '',
                 f"同口径指标：{stage['metrics']}", '', f"失败/退化：{stage['failures']}", '',
                 f"证据边界：{stage['boundary']}", '', f"下一步与所需支持：{stage['next']}", '']
    text += ['## 证据入口', '',
             f"[八项统一清单]({root / s['checklist']}) · [兼容与回退]({evidence / 'compatibility.md'}) · [指标JSON]({out / 'metrics.json'})", '',
             f"[运行组合原始结果]({evidence / 'integration/formal-result.json'}) · [冻结协议]({evidence / 'protocol.json'}) · [源码与证据哈希]({evidence / 'delivery-manifest.json'})", '',
             f"[前后APK及源码绑定]({evidence / 'experience/timed-build-results.json'}) · [双端测试材料]({evidence / 'experience/acceptance-materials.md'})", '',
             '阶段差量：' + ' · '.join(f"[{name}]({evidence / 'nodes' / name / 'node.diff'})" for name in ('temporal', 'rewrite', 'tags', 'cjk', 'scale', 'integration', 'experience')), '']
    text += ['## 校验与交付', '',
             f"已核对{metrics['hashes_verified']}个证据/源码哈希、完整分母与原始p95。Python最终{s['validation']['python_passed']}通过、{s['validation']['python_skipped']}跳过；初次失败及修复证据保留。相同观察器的前后APK各61本地单测通过，ADB列表为空，实际体验样本仍为0。", '',
             '阶段兼容及回退见 evidence根目录/compatibility.md；八项统一清单保留未完成状态。新报告目录不覆盖旧报告，报告生成不导入service、不调用模型。', '']
    (out / 'report.md').write_text('\n'.join(text))
    return out / 'report.md'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, default=Path('.planning/2026-10-12-continuous-recall-closeout'))
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    evidence = (root / args.evidence).resolve()
    out = args.out or evidence / 'reports' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    print(write_report(root, evidence, out.resolve()))
