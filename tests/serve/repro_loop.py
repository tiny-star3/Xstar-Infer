import httpx, asyncio

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


async def batch(n, mt):
    tails = [
        f" the kingdom of chapter {i} under the great mountain of stone and steel and iron"
        for i in range(n)
    ]

    async def one(c, i):
        async with c.stream(
            "POST", BASE, json={"prompt": SH + tails[i], "max_tokens": mt}, timeout=None
        ) as r:
            k = 0
            async for line in r.aiter_lines():
                if line.startswith("data:") and line[6:] != "[DONE]":
                    k += 1
            return i, k

    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=200)) as c:
        return await asyncio.gather(*[one(c, i) for i in range(n)])


for r in range(1, 21):
    send(SH + " the seed chapter.", 32)
    asyncio.run(batch(16, 80))
    send(SH + " the seed chapter.", 32)
    r2 = asyncio.run(batch(12, 256))
    bad = [t for t in r2 if isinstance(t[1], str)]
    print(f"round {r}: {'ERR '+str(bad) if bad else 'OK ' + str(r2[:3])}", flush=True)
print("ALL ROUNDS DONE")
