#!/usr/bin/env python3
"""Eval_Ali 端到端音频评测 harness.

把 Eval_Ali 的 WAV 经过 ffmpeg 降混成 16kHz 单声道, 按 <=30s 切片, 进程内调用
GlassesChatService.process_audio_segment 走真实本地 ASR -> 时间线 -> 再用
chat(ambient_capture_id=...) 触发记忆抽取, 最终汇总每文件的转写与记忆写入.

必须在 hermes 环境运行 (已装 funasr/torch/soundfile), 且已设置:
  AI_GLASSES_ASR_MODEL_DIR  本地 SenseVoice/Paraformer 模型目录
  AI_GLASSES_LLM_*          本地 qwen (记忆抽取零费用, 见 AGENTS.md)

用法:
  conda run -n hermes python scripts/eval_ali_audio.py \
      --root /Users/huyaokai/Downloads/Eval_Ali \
      --out reports/eval_ali_audio \
      --user-id eval-ali-01 \
      --limit-files 1
"""
from __future__ import annotations

import os
import sys

# 让脚本在 scripts/ 下也能 import 仓库根下的 ai_glasses_memory_assistant 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import base64
import io
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
CHUNK_SECONDS = 30  # 必须 <= offline.MAX_AUDIO_DURATION_MS (30000)
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS


def decode_to_mono16k(wav_path: Path, work_dir: Path) -> tuple[np.ndarray, int]:
    """用 ffmpeg 把任意声道/采样率的 WAV 降混成 16k 单声道 PCM, 返回 (float32[-1,1], sr)."""
    tmp = work_dir / (wav_path.stem + ".mono16k.wav")
    cmd = [
        "ffmpeg", "-y", "-i", str(wav_path),
        "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "wav", str(tmp),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    data, sr = sf.read(str(tmp), dtype="float32")
    if data.ndim > 1:  # 降混后的单声道通常已是一维; 兜底取首通道并压平
        data = data[:, 0]
    tmp.unlink(missing_ok=True)
    return data, int(sr)


def chunk_to_wav_base64(chunk: np.ndarray) -> str:
    """把 30s 单声道 float32 包成 WAV, 返回 base64 (符合 offline.ALLOWED_AUDIO_MIME_TYPES: audio/wav)."""
    pcm = np.clip(chunk, -1.0, 1.0)
    buf = io.BytesIO()
    sf.write(buf, pcm, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def discover_wavs(root: Path) -> list[Path]:
    return sorted(root.rglob("audio_dir/*.wav"))


def run_file(service, *, user_id: str, wav_path: Path, work_dir: Path,
             max_seconds: int | None) -> dict:
    audio, sr = decode_to_mono16k(wav_path, work_dir)
    assert sr == SAMPLE_RATE, f"unexpected sr {sr}"
    total_samples = len(audio)
    max_samples = (max_seconds * SAMPLE_RATE) if max_seconds else total_samples

    capture = service.start_capture(user_id=user_id, source="ambient_audio")
    capture_id = capture["capture_id"]
    transcripts: list[str] = []
    n_chunks = 0
    t0 = time.time()
    idx = 0
    while idx < min(total_samples, max_samples):
        end = min(idx + CHUNK_SAMPLES, total_samples, max_samples)
        chunk = audio[idx:end]
        if chunk.size < SAMPLE_RATE:  # 丢弃 <1s 的尾片
            break
        b64 = chunk_to_wav_base64(chunk)
        res = service.process_audio_segment(
            user_id=user_id,
            capture_id=capture_id,
            audio_base64=b64,
            audio_mime_type="audio/wav",
            audio_duration_ms=int(len(chunk) / SAMPLE_RATE * 1000),
            source_type="ambient_audio",
        )
        txt = (res.get("transcript") or "").strip()
        if txt:
            transcripts.append(txt)
        n_chunks += 1
        idx = end
    elapsed = time.time() - t0

    # 触发记忆抽取: 复用 evals/runner 的 ambient_capture 路径
    svc_chat = getattr(service, "chat", None)
    recall = None
    if svc_chat is not None:
        try:
            chat_res = svc_chat(
                "请基于我们刚才的对话更新你的长期记忆。",
                user_id=user_id,
                ambient_capture_id=capture_id,
                skip_reply_synthesis=True,
            )
            recall = chat_res
        except Exception as exc:  # noqa: BLE001
            recall = {"error": str(exc)}

    full_transcript = "\n".join(transcripts)
    return {
        "file": str(wav_path),
        "capture_id": capture_id,
        "duration_sec": round(min(total_samples, max_samples) / SAMPLE_RATE, 1),
        "chunks": n_chunks,
        "asr_seconds": round(elapsed, 1),
        "transcript_chars": len(full_transcript),
        "transcript_head": full_transcript[:500],
        "chat_recall": recall,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/Users/huyaokai/Downloads/Eval_Ali")
    ap.add_argument("--out", default="reports/eval_ali_audio")
    ap.add_argument("--user-id", default="eval-ali-01")
    ap.add_argument("--limit-files", type=int, default=0, help="0=全部")
    ap.add_argument("--max-seconds", type=int, default=0, help="0=整文件; 否则每文件只跑前 N 秒(验证用)")
    ap.add_argument("--home", default="", help="AI_GLASSES_HOME 隔离目录")
    args = ap.parse_args()

    if not os.environ.get("AI_GLASSES_ASR_MODEL_DIR"):
        print("ERROR: 先设置 AI_GLASSES_ASR_MODEL_DIR 指向本地 ASR 模型目录", file=sys.stderr)
        return 2
    if not os.environ.get("AI_GLASSES_LLM_BASE_URL"):
        print("WARN: 未设置 AI_GLASSES_LLM_* , 记忆抽取可能回退到 .env 的 deepseek(产生费用)", file=sys.stderr)

    root = Path(args.root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = out / "_tmp_audio"
    work.mkdir(parents=True, exist_ok=True)

    if args.home:
        os.environ["AI_GLASSES_HOME"] = str(Path(args.home).expanduser().resolve())
    os.environ.setdefault("AI_GLASSES_LLM_TRANSPORT", "stdlib_http")

    from ai_glasses_memory_assistant.agent_bridge import GlassesChatService

    service = GlassesChatService()

    wavs = discover_wavs(root)
    if args.limit_files:
        wavs = wavs[: args.limit_files]
    print(f"发现 {len(wavs)} 个 WAV, 开始处理 (ASR={os.environ['AI_GLASSES_ASR_MODEL_DIR']})")

    results = []
    for i, wav in enumerate(wavs, 1):
        print(f"[{i}/{len(wavs)}] {wav.name} ...", flush=True)
        try:
            rec = run_file(
                service, user_id=args.user_id, wav_path=wav, work_dir=work,
                max_seconds=args.max_seconds or None,
            )
        except Exception as exc:  # noqa: BLE001
            rec = {"file": str(wav), "error": str(exc)}
        results.append(rec)
        with open(out / "per_file.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"    -> chunks={rec.get('chunks')} asr={rec.get('asr_seconds')}s "
              f"transcript_chars={rec.get('transcript_chars')} "
              f"chat_err={rec.get('chat_recall', {}).get('error') if isinstance(rec.get('chat_recall'), dict) else None}")

    # 汇总记忆写入
    try:
        mems = service.memory_store.list_memories(user_id=args.user_id, limit=1000)
        mem_summary = [{"kind": m.kind, "type": getattr(m, "memory_type", ""), "content": m.content[:200]} for m in mems]
    except Exception as exc:  # noqa: BLE001
        mem_summary = [{"error": str(exc)}]
    summary = {"user_id": args.user_id, "files": len(results), "memories": mem_summary}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完成. 输出: {out}/per_file.jsonl, {out}/summary.json ; 记忆条数={len(mem_summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
