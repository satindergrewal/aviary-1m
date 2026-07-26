#!/usr/bin/env python3
"""Stage (b) of the GLM calibration corpus: model-self-generated shares.

Generates the Chinese (15%) and chat-template/reasoning (15%) shares by sampling the
PRODUCTION GLM-5.2 IQ1_KT under the tuned DRY serve config (spec ruling: dry_multiplier
0.8, dry_base 1.75, dry_allowed_length 1, temperature 0.7, --jinja, reasoning on), then
post-filtering every sample through the loop detector so the corpus is provably
loop-free regardless of sampler behavior.

Serve first (both cards), then run this against the endpoint. Think blocks are RETAINED
in the chat share (that is the point: production output contains them). Multi-turn
shapes are mixed per spec.
"""
import argparse
import hashlib
import json
import random
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from loop_rate import detect_loop  # noqa: E402  (aviary-1m tools/loop_rate.py)

ZH_TOPICS = [
    "分布式系统中的一致性协议", "Go语言的并发模型与通道", "点对点网络的NAT穿透技术",
    "大语言模型的量化方法", "Rust的所有权与借用检查器", "操作系统的虚拟内存管理",
    "密码学中的椭圆曲线签名", "数据库事务的隔离级别", "编译器的中间表示与优化",
    "网络协议栈中TCP拥塞控制", "容器技术与命名空间隔离", "GPU上矩阵乘法的优化",
    "哈希表的冲突处理策略", "异步编程中的事件循环", "版本控制系统的合并算法",
    "内存屏障与缓存一致性", "语音识别的声学模型", "推荐系统的协同过滤",
    "区块链的共识机制", "函数式编程的不可变数据结构",
]
ZH_FORMS = [
    "写一篇技术说明文章，解释{t}的核心原理和常见实现。",
    "以工程师的口吻，详细介绍{t}，包括实际应用中的注意事项。",
    "写一份面向初学者的教程，主题是{t}，配合具体例子。",
    "分析{t}的优缺点，并比较两种主流方案。",
    "写一篇关于{t}的深入综述，涵盖历史发展和当前趋势。",
]
CHAT_SEEDS = [
    ["Debug this: my Go service leaks goroutines when clients disconnect mid-stream. Where do I look first?",
     "The leak persists after adding context cancellation. The goroutines block on a channel send."],
    ["Explain the tradeoffs between mmap and pread for a database storage engine.",
     "Which one behaves better under memory pressure on Linux, and why?"],
    ["Write a Rust function that merges overlapping intervals, then explain its complexity.",
     "Now make it work on streaming input without holding the full set in memory."],
    ["Why does my quantized LLM repeat tokens at low bit-width while perplexity looks fine?",
     "What sampler settings mitigate it, and where do they stop helping?"],
    ["Design a reconnect strategy for a P2P mesh where peers churn every few minutes.",
     "How do exponential backoff and jitter interact with hole punching here?"],
    ["Walk me through how speculative decoding verification preserves the target distribution.",
     "What changes when the drafter proposes a whole block instead of one token?"],
    ["My CUDA kernel is memory-bound at 40% occupancy. What do I profile first?",
     "Shared memory tiling helped. Why does the next bottleneck appear at the L2?"],
    ["Compare Raft and gossip protocols for membership in a 10k-node cluster.",
     "What failure modes does each hide during network partitions?"],
]

def chat(endpoint: str, messages, max_tokens: int, timeout: int = 600):
    body = json.dumps({
        "messages": messages, "temperature": 0.7, "top_p": 0.95,
        "max_tokens": max_tokens, "stream": False,
        "samplers": "top_k;top_p;min_p;temperature;dry;typ_p;xtc",
        "dry_multiplier": 0.8, "dry_base": 1.75, "dry_allowed_length": 1,
    }).encode()
    req = urllib.request.Request(endpoint + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    msg = r["choices"][0]["message"]
    think = msg.get("reasoning_content") or ""
    content = msg.get("content") or ""
    return think, content

def clean(think: str, content: str, keep_think: bool):
    """Loop/empty filter per spec ruling. Returns (text, reason): text None when dropped,
    reason 'loop' | 'short' | None."""
    for part in (think, content):
        if part:
            verdict, _ = detect_loop(part)
            if verdict == "LOOP":
                return None, "loop"
    if len(content.split()) < 30:
        return None, "short"
    if keep_think and think:
        return f"<think>{think}</think>\n{content}", None
    return content, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8641")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--zh-mb", type=float, default=7.5)
    ap.add_argument("--chat-mb", type=float, default=7.5)
    ap.add_argument("--max-tokens", type=int, default=1600)
    ap.add_argument("--seed", type=int, default=41)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats = {"zh": {"kept": 0, "dropped_loop": 0, "dropped_short": 0}, "chat": {"kept": 0, "dropped_loop": 0, "dropped_short": 0}}

    # Chinese share: single-turn technical/general prose, content only (prose corpus)
    zh_path = out / "chinese.txt"
    with zh_path.open("w", encoding="utf-8") as f:
        total = 0
        while total < args.zh_mb * 1e6:
            t = rng.choice(ZH_TOPICS)
            prompt = rng.choice(ZH_FORMS).format(t=t)
            try:
                think, content = chat(args.endpoint, [{"role": "user", "content": prompt}], args.max_tokens)
            except Exception as e:
                print(f"zh request error: {e}", flush=True)
                continue
            text, reason = clean(think, content, keep_think=False)
            if text is None:
                key = "dropped_" + reason
                stats["zh"][key] += 1
                print(f"zh DROP {reason} #{stats['zh'][key]} (think={len(think)} content={len(content.split())}w)", flush=True)
                with (out / "dropped_zh.txt").open("a", encoding="utf-8") as df:
                    df.write(f"\n\n==== DROP {reason} {stats['zh'][key]} ====\nTHINK:\n{think}\nCONTENT:\n{content}\n")
                continue
            f.write(f"\n\n==== {t} ====\n\n" + text)
            f.flush()
            total += len(text)
            stats["zh"]["kept"] += 1
            if stats["zh"]["kept"] % 20 == 0:
                print(f"zh: {total/1e6:.2f}/{args.zh_mb} MB ({stats['zh']})", flush=True)

    # Chat share: multi-turn, jinja-shaped, THINK RETAINED
    chat_path = out / "chat_think.txt"
    with chat_path.open("w", encoding="utf-8") as f:
        total = 0
        while total < args.chat_mb * 1e6:
            seed = rng.choice(CHAT_SEEDS)
            messages = [{"role": "user", "content": seed[0]}]
            sample_parts = [f"USER: {seed[0]}"]
            ok = True
            for turn in range(1 + rng.randint(0, 1)):
                try:
                    think, content = chat(args.endpoint, messages, args.max_tokens)
                except Exception as e:
                    print(f"chat request error: {e}", flush=True)
                    ok = False
                    break
                text, reason = clean(think, content, keep_think=True)
                if text is None:
                    key = "dropped_" + reason
                    stats["chat"][key] += 1
                    print(f"chat DROP {reason} #{stats['chat'][key]} (think={len(think)} content={len(content.split())}w)", flush=True)
                    with (out / "dropped_chat.txt").open("a", encoding="utf-8") as df:
                        df.write(f"\n\n==== DROP {reason} {stats['chat'][key]} ====\nTHINK:\n{think}\nCONTENT:\n{content}\n")
                    ok = False
                    break
                sample_parts.append(f"ASSISTANT: {text}")
                messages.append({"role": "assistant", "content": content})
                if turn + 1 < len(seed):
                    messages.append({"role": "user", "content": seed[turn + 1]})
                    sample_parts.append(f"USER: {seed[turn + 1]}")
            if not ok:
                continue
            sample = "\n\n".join(sample_parts)
            f.write("\n\n==== CHAT ====\n\n" + sample)
            f.flush()
            total += len(sample)
            stats["chat"]["kept"] += 1
            if stats["chat"]["kept"] % 20 == 0:
                print(f"chat: {total/1e6:.2f}/{args.chat_mb} MB ({stats['chat']})", flush=True)

    manifest_path = out / "manifest_stage_b.json"
    manifest = {"generator": "GLM-5.2 IQ1_KT production artifact, tuned DRY config (spec ruling)",
                "filter": "loop_rate.detect_loop on think AND content; LOOP/EMPTY dropped",
                "stats": stats, "shares": {}}
    for name, p in [("chinese", zh_path), ("chat_think", chat_path)]:
        data = p.read_text(encoding="utf-8")
        manifest["shares"][name] = {"bytes": len(data),
                                    "sha256": hashlib.sha256(data.encode()).hexdigest()}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print("STAGE-B-COMPLETE", stats, flush=True)

if __name__ == "__main__":
    main()
