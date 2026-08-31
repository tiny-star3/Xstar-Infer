import httpx

BASE = "http://127.0.0.1:8000/generate"
SH = (
    "Once upon a time in a land far away there lived a wise old king who ruled over "
) * 6


def send(prompt, mt):
    with httpx.stream(
        "POST", BASE, json={"prompt": prompt, "max_tokens": mt}, timeout=None
    ) as r:
        for line in r.iter_lines():
            if line.startswith("data:") and line[6:] == "[DONE]":
                return


send(SH + " the seed chapter.", 32)  # 种一棵前缀节点(matched=0)
send(
    SH + " the kingdom of chapter 0 under the mountain", 64
)  # matched>0: 命中+pin+adopt
send(SH + " the kingdom of chapter 1 under the mountain", 64)  # matched>0: 再命中
