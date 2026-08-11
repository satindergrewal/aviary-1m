#!/usr/bin/env python3
"""Convert glm-calib chat_think.txt into K3-native XTML calibration text.

Why: the imatrix we built on 2026-07-31 used chat-SHAPED text (USER:/ASSISTANT:
transcripts). The 2.11 bpw artifact answers correctly but cannot close a THINK
block (<|close|>think<|sep|> is never emitted) - and our calibration contained
zero instances of that exact structure. This renders the same conversations in
K3's native format so the imatrix sees the think-block regime.

Native structure (from tools/k3/chat/k3_chat_template_minja_fixed.jinja):
  <|open|>message role="user"<|sep|>X<|close|>message<|sep|><|end_of_msg|>
  <|open|>message role="assistant"<|sep|>
  <|open|>think<|sep|>T<|close|>think<|sep|>
  <|open|>response<|sep|>R<|close|>response<|sep|>
  <|close|>message<|sep|><|end_of_msg|>

Input : <BOX>/glm-calib/corpus/chat_think.txt   (==== CHAT ==== / USER: / ASSISTANT:)
Output: stdout (redirect to file)
"""
import re, sys

SRC = "<BOX>/glm-calib/corpus/chat_think.txt"
THINK_EFFORT_SYS = ('<|open|>message role="system" type="thinking-effort"<|sep|>'
                    "`thinking_effort` guides on how much to think in the response."
                    "<|close|>message<|sep|><|end_of_msg|>\n")

def msg(role, content):
    return (f'<|open|>message role="{role}"<|sep|>{content}'
            f'<|close|>message<|sep|><|end_of_msg|>\n')

def assistant(content):
    # content holds GLM-style <think>...</think> inline; map to native blocks
    m = re.match(r"\s*<think>(.*?)</think>\s*(.*)", content, re.S)
    if m:
        think, resp = m.group(1).strip(), m.group(2).strip()
    else:
        think, resp = "", content.strip()
    out = '<|open|>message role="assistant"<|sep|>\n'
    if think:
        out += f'<|open|>think<|sep|>{think}<|close|>think<|sep|>\n'
    out += f'<|open|>response<|sep|>{resp}<|close|>response<|sep|>\n'
    out += '<|close|>message<|sep|><|end_of_msg|>\n'
    return out

def main():
    text = open(SRC, encoding="utf-8", errors="replace").read()
    convos = [c for c in text.split("==== CHAT ====") if c.strip()]
    n_turns = n_thinkblocks = 0
    for c in convos:
        # turns begin with "USER:" or "ASSISTANT:" at line start
        parts = re.split(r"(?m)^(USER:|ASSISTANT:)\s*", "\n" + c.strip())
        # parts: ['', 'USER:', 'text', 'ASSISTANT:', 'text', ...]
        out = [THINK_EFFORT_SYS]
        i = 1
        while i + 1 < len(parts):
            tag, body = parts[i], parts[i + 1]
            body = body.strip()
            if not body:
                i += 2
                continue
            if tag == "USER:":
                out.append(msg("user", body))
            else:
                out.append(assistant(body))
                if "<think>" in body:
                    n_thinkblocks += 1
            n_turns += 1
            i += 2
        sys.stdout.write("".join(out))
    sys.stderr.write(f"conversations={len(convos)} turns={n_turns} think_blocks={n_thinkblocks}\n")

if __name__ == "__main__":
    main()
