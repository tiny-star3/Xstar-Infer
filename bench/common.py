import httpx
import json

URL = "http://127.0.0.1:8000/generate"


def make_payload(prompt, max_tokens):
    return {"prompt": prompt, "max_tokens": max_tokens}


async def stream_one(client, payload):
    tokens = []
    async with client.stream("POST", URL, json=payload) as resp:
        async for line in resp.aiter_lines():
            # 跳空行(SSE 双换行分隔)
            if not line.strip():
                continue
            # 非 data 行跳过
            if not line.startswith("data:"):
                continue
            body = line[len("data:") :].strip()
            # 结束信号,不是 token
            if body == "[DONE]":
                break
            tokens.append(json.loads(body)["token"])
    return tokens
